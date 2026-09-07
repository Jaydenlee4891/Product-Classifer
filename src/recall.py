"""Build step 2b: the retrieval gate.

Embeds items and label documents with a FROZEN off-the-shelf encoder and reports
recall@k per label-document variant, per frequency stratum.

This is the ceiling on the whole cascade: no downstream stage recovers a leaf the
retriever never surfaced. If recall@20 is already >= 0.95 overall and >= 0.85 on the
tail strata, the retriever needs no fine-tuning — which on a 6 GB card is the outcome
you want anyway.

Item embeddings are cached to .npy. They depend only on the model and the split, so
the expensive pass runs once and every later experiment reads it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen3-Embedding-0.6B"
KS = (1, 5, 20, 50)
STRATA = ["head", "torso", "tail", "few_shot", "zero_shot"]
# Qwen3-Embedding is instruction-aware: the asymmetric task needs an instruction on the
# query side only. Documents are embedded bare.
INSTRUCTION = "Given a product listing, retrieve the product category it belongs to"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str) -> SentenceTransformer:
    # fp16 on CUDA: a 6 GB Turing card holds 0.6B in half precision with room to spare.
    # No bf16 below compute capability 8.0, so fp16 is the only half-precision option here.
    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    print(f"Loading {MODEL} on {device}" + (" (fp16)" if kwargs else ""))
    return SentenceTransformer(MODEL, device=device, model_kwargs=kwargs)


def assert_finite(a: np.ndarray, what: str) -> None:
    """NaN embeddings do not crash anything — they rank as misses and quietly deflate
    every score. Check rather than trust.

    Note: numpy 2.x on macOS (Accelerate BLAS) can raise 'divide by zero' / 'overflow' /
    'invalid value' RuntimeWarnings from matmul even when inputs and outputs are finite.
    That is why the warnings are silenced at the matmul below and the actual invariant —
    finite in, finite out — is asserted instead.
    """
    bad = int((~np.isfinite(a).all(axis=1)).sum())
    if bad:
        raise SystemExit(
            f"\n{bad} of {len(a)} {what} contain NaN or inf.\n"
            "Scores computed from these would be silently wrong. Remedies, in order:\n"
            "  1. delete data/emb_items_*.npy (a bad cache is reused otherwise)\n"
            "  2. re-run with --device cpu (MPS and fp16 are the usual culprits)\n"
        )


def item_text(row: pd.Series) -> str:
    parts = [row.title]
    if row.brand:
        parts.append(f"Brand: {row.brand}")
    if row.bullets:
        parts.append(row.bullets)
    return " ".join(parts)[:1000]


def embed_items(model, df: pd.DataFrame, cache: Path, batch_size: int) -> np.ndarray:
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(df):
            print(f"Loaded cached item embeddings from {cache}")
            return emb
        print(f"Cache size {len(emb)} != {len(df)} rows; re-embedding.")
    print(f"Embedding {len(df):,} items (one-time, cached afterwards)")
    emb = model.encode(
        [item_text(r) for r in df.itertuples()],
        prompt=f"Instruct: {INSTRUCTION}\nQuery: ",
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


def recall_table(
    item_emb: np.ndarray, label_emb: np.ndarray, leaves: list[str], df: pd.DataFrame
) -> pd.DataFrame:
    assert_finite(item_emb, "item embeddings")
    assert_finite(label_emb, "label embeddings")

    idx = {leaf: i for i, leaf in enumerate(leaves)}
    gold = df.product_type.map(idx).to_numpy()
    max_k = min(max(KS), label_emb.shape[0])
    # Chunked so the N x n_leaves similarity matrix is never materialised in full.
    ranks = np.empty((len(df), max_k), dtype=np.int32)
    for i in range(0, len(item_emb), 4096):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = item_emb[i : i + 4096] @ label_emb.T
        if not np.isfinite(sims).all():
            raise SystemExit(
                "Similarity matrix is non-finite despite finite inputs — stop and "
                "investigate the BLAS backend before trusting any score."
            )
        part = np.argpartition(-sims, max_k - 1, axis=1)[:, :max_k]
        ordered = np.take_along_axis(part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), 1)
        ranks[i : i + 4096] = ordered
    hit = ranks == gold[:, None]

    res = pd.DataFrame({"leaf": df.product_type.values, "stratum": df.stratum.values})
    for k in KS:
        if k <= max_k:
            res[f"hit{k}"] = hit[:, :k].any(1)

    # micro = per item, macro = per leaf. They diverge hard here: one ABO leaf is 44% of
    # the corpus, so a micro average is close to a report on that single class. Macro is
    # the number that describes the taxonomy.
    rows = {}
    for stratum in STRATA + ["ALL"]:
        sub = res if stratum == "ALL" else res[res.stratum == stratum]
        if sub.empty:
            continue
        d = {"items": len(sub), "leaves": int(sub.leaf.nunique())}
        for k in KS:
            if f"hit{k}" not in sub:
                continue
            d[f"micro@{k}"] = float(sub[f"hit{k}"].mean())
            d[f"macro@{k}"] = float(sub.groupby("leaf")[f"hit{k}"].mean().mean())
        rows[stratum] = d
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    # Rare leaves often have a single item, and split() sends that one item to test — so
    # `val` alone covers only 30 of the 90 zero-shot leaves. Nothing is fitted in this
    # script (frozen encoder, no thresholds), so val+test is legitimate here and is the
    # only way to score every leaf. Use val alone once you start tuning on the result.
    ap.add_argument("--split", default="val+test", choices=["val", "test", "val+test"])
    ap.add_argument("--batch-size", type=int, default=16, help="16 suits a 6 GB card")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate a random subsample (0 = all). Use ~2000 for a smoke test; "
        "the rare strata get too few items to trust, so re-run without it for real numbers.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.data / "items.parquet")
    df = df[df.split.isin(args.split.split("+"))].reset_index(drop=True)
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=17).reset_index(drop=True)
        print(f"SMOKE TEST: subsampled to {len(df):,} items. Per-stratum numbers are noisy.\n")
    docs = json.loads((args.data / "label_docs.json").read_text())
    leaves = sorted(docs)
    variants = list(next(iter(docs.values())))

    device = pick_device(args.device)
    model = load_model(device)

    tag = args.split.replace("+", "_") + (f"_n{args.limit}" if args.limit else "")
    item_emb = embed_items(model, df, args.data / f"emb_items_{tag}.npy", args.batch_size)

    rows = []
    for variant in variants:
        label_emb = model.encode(
            [docs[leaf][variant] for leaf in leaves],
            batch_size=args.batch_size,
            normalize_embeddings=True,
        ).astype(np.float32)
        tbl = recall_table(item_emb, label_emb, leaves, df)
        tbl.insert(0, "variant", variant)
        tbl.index.name = "stratum"
        rows.append(tbl.reset_index())
        print(f"\n=== {variant} ===")
        print(tbl.to_string(float_format=lambda v: f"{v:.3f}"))

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(args.data / "recall.csv", index=False)
    print(f"\nWrote {args.data / 'recall.csv'}")

    pivot = result.pivot(index="stratum", columns="variant", values="macro@20").reindex(
        [s for s in STRATA + ["ALL"] if s in set(result.stratum)]
    )
    print("\nmacro-averaged recall@20 — the table to show\n")
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n(macro = averaged over leaves. The micro column in recall.csv is averaged")
    print(" over items, where one leaf is ~44% of ABO and dominates the number.)")

    best = variants[-1]
    overall = result[(result.stratum == "ALL") & (result.variant == best)]["macro@20"]
    tails = result[(result.stratum.isin(["few_shot", "zero_shot"])) & (result.variant == best)]["macro@20"]
    if not len(overall):
        return
    worst_tail = float(tails.min()) if len(tails) else float("nan")
    ok = overall.iloc[0] >= 0.95 and (worst_tail >= 0.85 if len(tails) else True)
    k50 = result[(result.stratum == "ALL") & (result.variant == best)].get("macro@50")
    print(f"\nGATE ({best}, macro-averaged): {'PASS' if ok else 'FAIL'}")
    print(f"  overall macro@20      {overall.iloc[0]:.3f}   (target 0.95)")
    if len(tails):
        print(f"  worst tail macro@20   {worst_tail:.3f}   (target 0.85)")
    if k50 is not None and len(k50):
        print(f"  overall macro@50      {float(k50.iloc[0]):.3f}   (widening K is free here)")
    print(
        "  -> retriever stays frozen; skip fine-tuning entirely."
        if ok
        else "  -> before fine-tuning: widen K, then improve the label documents of the\n"
        "     worst leaves. Both are cheaper than training, and the thresholds above are\n"
        "     design-doc guesses — a miss of 0.005 is not a reason to train anything."
    )


if __name__ == "__main__":
    main()
