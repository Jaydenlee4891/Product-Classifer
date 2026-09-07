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
KS = (1, 5, 20)
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
    gold = np.array([leaves.index(pt) for pt in df.product_type])
    max_k = max(KS)
    # Chunked so the N x 576 similarity matrix is never materialised in full.
    ranks = np.empty((len(df), max_k), dtype=np.int32)
    for i in range(0, len(item_emb), 4096):
        sims = item_emb[i : i + 4096] @ label_emb.T
        part = np.argpartition(-sims, max_k - 1, axis=1)[:, :max_k]
        ordered = np.take_along_axis(part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), 1)
        ranks[i : i + 4096] = ordered

    hit = ranks == gold[:, None]
    out = {}
    for stratum in STRATA + ["ALL"]:
        mask = np.ones(len(df), bool) if stratum == "ALL" else (df.stratum == stratum).values
        if mask.sum() == 0:
            continue
        out[stratum] = {f"r@{k}": float(hit[mask, :k].any(1).mean()) for k in KS}
        out[stratum]["n"] = int(mask.sum())
    return pd.DataFrame(out).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=16, help="16 suits a 6 GB card")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    df = pd.read_parquet(args.data / "items.parquet")
    df = df[df.split == args.split].reset_index(drop=True)
    docs = json.loads((args.data / "label_docs.json").read_text())
    leaves = sorted(docs)
    variants = list(next(iter(docs.values())))

    device = pick_device(args.device)
    model = load_model(device)

    item_emb = embed_items(
        model, df, args.data / f"emb_items_{args.split}.npy", args.batch_size
    )

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

    pivot = result.pivot(index="stratum", columns="variant", values="r@20").reindex(
        [s for s in STRATA + ["ALL"] if s in set(result.stratum)]
    )
    print("\nrecall@20 — the table to show\n")
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    best = variants[-1]
    overall = result[(result.stratum == "ALL") & (result.variant == best)]["r@20"]
    tails = result[(result.stratum.isin(["few_shot", "zero_shot"])) & (result.variant == best)]["r@20"]
    if not len(overall):
        return
    worst_tail = float(tails.min()) if len(tails) else float("nan")
    ok = overall.iloc[0] >= 0.95 and (worst_tail >= 0.85 if len(tails) else True)
    print(f"\nGATE ({best}): {'PASS' if ok else 'FAIL'}")
    print(f"  overall r@20      {overall.iloc[0]:.3f}   (need >= 0.95)")
    if len(tails):
        print(f"  worst tail r@20   {worst_tail:.3f}   (need >= 0.85)")
    print(
        "  -> retriever stays frozen; skip fine-tuning entirely."
        if ok
        else "  -> richer label documents first. Fine-tuning is the last resort, not the first."
    )


if __name__ == "__main__":
    main()
