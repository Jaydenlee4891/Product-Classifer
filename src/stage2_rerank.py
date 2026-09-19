"""Build step 5b: Stage 2's cross-encoder.

The bi-encoder embeds item and label independently, so it can only measure similarity
in a space that never saw them together. The cross-encoder reads the pair jointly, which
is what lets it separate siblings — the job the bi-encoder structurally cannot do.

Order of operations mirrors step 2: measure the free thing first.

    --epochs 0    score with an off-the-shelf reranker, no training. If that already
                  lifts top-1 enough, you have your answer for the price of an hour.
    --epochs 2    fine-tune listwise on hard negatives mined from the frozen retriever.

Two decisions worth understanding before reading the code:

  NEGATIVES come from the retriever's own top-K, never at random. A cross-encoder trained
  on random negatives learns to tell shoes from lawnmowers, which is not its job at
  inference — by then the retriever has already removed the lawnmowers.

  TRAINING ITEMS are capped per leaf. One ABO leaf is 53% of the corpus; without a cap
  the reranker spends its capacity re-learning phone cases and never sees the siblings
  it exists to separate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from recall import (assert_finite, embed_items, load_model, pick_device, save_array)

SEED = 17


def mine(item_emb: np.ndarray, label_emb: np.ndarray, k: int) -> np.ndarray:
    """Top-k label indices per item, best first."""
    assert_finite(item_emb, "item embeddings")
    assert_finite(label_emb, "label embeddings")
    out = np.empty((len(item_emb), k), dtype=np.int32)
    for i in range(0, len(item_emb), 4096):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = item_emb[i : i + 4096] @ label_emb.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out[i : i + 4096] = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), 1
        )
    return out


def build_groups(cands: np.ndarray, gold: np.ndarray, group_size: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """One training group per item: the gold label plus (group_size-1) hard negatives
    drawn from that item's retrieved candidates.

    Gold is force-inserted when retrieval missed it (~2.3% of items). Skipping those
    would train only on cases retrieval already solved, which is the wrong distribution
    to learn from. At inference a missing gold shows up as an empty conformal set and
    escalates — that path is handled downstream, not here.
    """
    groups, pos = [], []
    for row, g in zip(cands, gold):
        negs = [c for c in row if c != g][: group_size - 1]
        while len(negs) < group_size - 1:            # tiny taxonomies only
            negs.append(int(rng.integers(0, cands.max() + 1)))
        members = [g] + negs
        order = rng.permutation(group_size)          # gold must not sit at a fixed index
        groups.append([members[o] for o in order])
        pos.append(int(np.where(order == 0)[0][0]))
    return np.array(groups, dtype=np.int32), np.array(pos, dtype=np.int64)


def encode_pairs(tok, items: list[str], labels: list[str], max_len: int):
    enc = tok(items, labels, truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def train(model, ids, mask, pos, group_size, device, epochs, lr, bs_groups, use_fp16):
    """Listwise: softmax over each group's scores, cross-entropy against the gold slot.
    This optimises 'pick the right one out of these K', which is exactly the inference
    task — a per-pair binary loss optimises something adjacent but different."""
    n_groups = len(pos)
    ds = TensorDataset(ids.view(n_groups, group_size, -1), mask.view(n_groups, group_size, -1),
                       torch.tensor(pos))
    loader = DataLoader(ds, batch_size=bs_groups, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if not any(x in n for x in ("bias", "LayerNorm.weight"))], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if any(x in n for x in ("bias", "LayerNorm.weight"))], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    step = 0
    for ep in range(epochs):
        run = 0.0
        for gi, gm, gp in loader:
            b = gi.shape[0]
            flat_i = gi.reshape(b * group_size, -1).to(device)
            flat_m = gm.reshape(b * group_size, -1).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(input_ids=flat_i, attention_mask=flat_m).logits.view(b, group_size)
                loss = lossf(logits.float(), gp.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); step += 1
            if step % 100 == 0:
                print(f"  epoch {ep+1} step {step}/{total} loss {run/100:.4f}", flush=True)
                run = 0.0
    return model


@torch.no_grad()
def score(model, tok, texts: list[str], docs: list[str], cands: np.ndarray,
          max_len: int, bs: int, device, use_fp16) -> np.ndarray:
    """Score every (item, candidate) pair. Returns [n_items, k] raw logits."""
    n, k = cands.shape
    flat_items = [texts[i] for i in range(n) for _ in range(k)]
    flat_docs = [docs[c] for row in cands for c in row]
    out = np.empty(n * k, dtype=np.float32)
    model.eval()
    for i in range(0, len(flat_items), bs):
        ids, mask = encode_pairs(tok, flat_items[i:i+bs], flat_docs[i:i+bs], max_len)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            lg = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out[i:i+bs] = lg.float().squeeze(-1).cpu().numpy()
        if (i // bs) % 200 == 0:
            print(f"  scored {i:,}/{len(flat_items):,}", flush=True)
    return out.reshape(n, k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="off-the-shelf reranker; use a Korean encoder for arm B")
    ap.add_argument("--k", type=int, default=50, help="candidates per item; the step-2 gate settled on 50")
    ap.add_argument("--doc-variant", default="v3_proto", choices=["v1_name", "v2_path", "v3_proto", "v4_desc"],
                    help="RETRIEVAL and RERANKING want different label docs. Retrieval rewards breadth — "
                         "prototype titles gave +0.06 macro@20. Reranking rewards discrimination, and the "
                         "same prototypes create spurious lexical overlap. Ablate v2_path against v3_proto "
                         "here; do not assume the step-2 winner wins twice.")
    ap.add_argument("--group-size", type=int, default=8, help="1 gold + 7 hard negatives per training group")
    ap.add_argument("--cap-per-leaf", type=int, default=200, help="training items per leaf; blunts the 53% head")
    ap.add_argument("--epochs", type=int, default=2, help="0 = score off-the-shelf, no training")
    ap.add_argument("--batch-groups", type=int, default=4, help="groups per step; x group_size = sequences")
    ap.add_argument("--score-batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=192,
                    help="item ~80 tok + label doc ~70 tok; 192 covers the pair with room")
    ap.add_argument("--limit-eval", type=int, default=0,
                    help="subsample val/test before scoring. Scoring is the expensive step: "
                         "24k items x 50 candidates = 1.2M pairs. Use 4000 for a first pass.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(args.data / "items.parquet")
    docs_all = json.loads((args.data / "label_docs.json").read_text())
    leaves = sorted(docs_all)
    docs = [docs_all[l][args.doc_variant] for l in leaves]
    lidx = {l: i for i, l in enumerate(leaves)}
    longest = max(len(d) for d in docs)
    if longest / 3.5 + 80 > args.max_len:      # ~3.5 chars/token, item text ~80 tokens
        print(f"NOTE: longest label doc is {longest} chars; with --max-len {args.max_len} the pair "
              "truncates and the tail of the doc (the prototype titles) is silently dropped. "
              "Raise --max-len or use a shorter --doc-variant.")

    device = pick_device(args.device)
    use_fp16 = device == "cuda"
    print(f"Device: {device}{' (fp16)' if use_fp16 else ''}")

    bi = load_model(device)
    label_emb = bi.encode(docs, batch_size=16, normalize_embeddings=True).astype(np.float32)

    # --- training pool: capped per leaf so the reranker sees siblings, not phone cases
    train_df = df[df.split == "train"]
    capped = (train_df.groupby("product_type", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), args.cap_per_leaf), random_state=SEED))
                      .reset_index(drop=True))
    print(f"Training pool: {len(capped):,} items (from {len(train_df):,}) "
          f"across {capped.product_type.nunique()} leaves, cap {args.cap_per_leaf}/leaf")

    cal = df[df.split == "val"].reset_index(drop=True)     # conformal calibration
    test = df[df.split == "test"].reset_index(drop=True)   # evaluation only
    if args.limit_eval:
        cal = cal.sample(min(args.limit_eval, len(cal)), random_state=SEED).reset_index(drop=True)
        test = test.sample(min(args.limit_eval, len(test)), random_state=SEED).reset_index(drop=True)
        print(f"SUBSAMPLED eval to {len(cal):,} cal / {len(test):,} test — rare strata will be "
              "thin, and the conformal quantile needs >= 1/alpha - 1 points per group.")

    from recall import item_text
    emb = {}
    for name, part in (("trainpool", capped), ("val", cal), ("test", test)):
        emb[name] = embed_items(bi, part, args.data / f"emb_items_{name}.npy", 16)

    cands = {n: mine(e, label_emb, args.k) for n, e in emb.items()}
    del bi
    if device == "cuda":
        torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)

    tag = "offshelf" if args.epochs == 0 else f"ft{args.epochs}"
    if args.epochs > 0:
        gold = capped.product_type.map(lidx).to_numpy()
        groups, pos = build_groups(cands["trainpool"], gold, args.group_size, rng)
        texts = [item_text(r) for r in capped.itertuples()]
        flat_items = [texts[i] for i in range(len(texts)) for _ in range(args.group_size)]
        flat_docs = [docs[c] for row in groups for c in row]
        print(f"Tokenising {len(flat_items):,} pairs ({len(groups):,} groups x {args.group_size})")
        ids, mask = encode_pairs(tok, flat_items, flat_docs, args.max_len)
        print(f"Training listwise, {args.epochs} epochs")
        train(model, ids, mask, pos, args.group_size, device, args.epochs,
              args.lr, args.batch_groups, use_fp16)

    out = args.data / "stage2"
    out.mkdir(parents=True, exist_ok=True)

    # Persist the fine-tuned reranker. Nothing else in this script reproduces it, and
    # scores alone cannot rank a candidate list you have not already scored.
    if args.epochs > 0:
        ckpt = out / f"model_{tag}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        (ckpt / "run_config.json").write_text(
            json.dumps({**{k: str(v) for k, v in vars(args).items()},
                        # args.device is what was REQUESTED ("auto"); these two are what
                        # it resolved to, which is the part that reproduces a number.
                        "resolved_device": device, "fp16_autocast": use_fp16,
                        "n_train_groups": len(capped)}, indent=2))
        print(f"Saved checkpoint -> {ckpt}")

    for name, part in (("val", cal), ("test", test)):
        texts = [item_text(r) for r in part.itertuples()]
        print(f"Scoring {name}: {len(part):,} items x {args.k} candidates")
        s = score(model, tok, texts, docs, cands[name], args.max_len,
                  args.score_batch, device, use_fp16)
        save_array(out / f"scores_{name}_{tag}.npy", s, model=model,
                   device=device, fp16_autocast=use_fp16)
        save_array(out / f"cands_{name}.npy", cands[name], device=device,
                   note="candidate ids from the BI-ENCODER, not this model")
    (out / "leaves.json").write_text(json.dumps(leaves))

    # --- THE GATE ---------------------------------------------------------------
    # Not top-1 accuracy. The cross-encoder exists to hand Stage 3's agent a SHORT
    # list, so the question is how many candidates the agent must read to have the
    # right answer in front of it. That is recall@m, and the reranker earns its
    # place only by reaching a target recall at smaller m than the retriever alone.
    tgold = test.product_type.map(lidx).to_numpy()
    s = np.load(out / f"scores_test_{tag}.npy")
    c_bi = cands["test"]
    c_ce = np.take_along_axis(c_bi, np.argsort(-s, axis=1), 1)

    def gold_rank(cand):
        hit = cand == tgold[:, None]
        return np.where(hit.any(1), hit.argmax(1), 10**6)

    r_bi, r_ce = gold_rank(c_bi), gold_rank(c_ce)
    leaf = test.product_type

    print(f"\n=== recall@m, macro over leaves — m is what the agent reads ===")
    print(f"{'m':>4s} {'bi-encoder':>11s} {'cross-enc':>11s} {'lift':>8s}")
    for m in (1, 3, 5, 10, 20, args.k):
        b = pd.Series(r_bi < m).groupby(leaf).mean().mean()
        e = pd.Series(r_ce < m).groupby(leaf).mean().mean()
        print(f"{m:>4d} {b:>11.3f} {e:>11.3f} {e - b:>+8.3f}")
    # invariant: reordering the same pool cannot change recall at full depth
    full_b = pd.Series(r_bi < args.k).groupby(leaf).mean().mean()
    full_e = pd.Series(r_ce < args.k).groupby(leaf).mean().mean()
    assert abs(full_b - full_e) < 1e-9, "recall@k differs — the reordering is wrong"

    print(f"\nsmallest m reaching a target recall:")
    for target in (0.90, 0.95):
        line = []
        for name, r in (("bi", r_bi), ("ce", r_ce)):
            hit = next((m for m in range(1, args.k + 1)
                        if pd.Series(r < m).groupby(leaf).mean().mean() >= target), None)
            line.append(f"{name} m={hit if hit else '>' + str(args.k)}")
        print(f"  {target:.0%}: " + "   ".join(line))
    # Per stratum. An aggregate here is a report on the head: it is 70% of the sample,
    # and a reranker that wins on head while burying rare leaves looks like a triumph
    # micro-averaged and a regression macro-averaged. Both happened on the first run.
    print(f"\n=== recall@1 and mean gold rank, by stratum (macro over leaves) ===")
    print(f"{'stratum':>10s} {'n':>6s} {'leaves':>7s} | {'bi r@1':>7s} {'ce r@1':>7s} {'lift':>7s} "
          f"| {'bi rank':>8s} {'ce rank':>8s}")
    got = r_bi < 10**6
    for st in ["head", "torso", "tail", "few_shot", "zero_shot", "ALL"]:
        m = np.ones(len(test), bool) if st == "ALL" else (test.stratum == st).to_numpy()
        if m.sum() == 0:
            continue
        lf = leaf[m]
        b1 = pd.Series(r_bi[m] < 1).groupby(lf.values).mean().mean()
        c1 = pd.Series(r_ce[m] < 1).groupby(lf.values).mean().mean()
        mg = m & got
        lg = leaf[mg].values
        br = pd.Series(r_bi[mg]).groupby(lg).mean().mean()
        cr = pd.Series(r_ce[mg]).groupby(lg).mean().mean()
        print(f"{st:>10s} {m.sum():>6d} {lf.nunique():>7d} | {b1:>7.3f} {c1:>7.3f} {c1-b1:>+7.3f} "
              f"| {br:>8.2f} {cr:>8.2f}")
    print(f"\nCE moved gold up for {(r_ce[got] < r_bi[got]).mean():.1%} of items, "
          f"down for {(r_ce[got] > r_bi[got]).mean():.1%} (item-weighted).")
    print("Read the zero_shot row first. Leaves with no training rows appear in the training")
    print("groups ONLY as hard negatives, so the model is taught they are always wrong and")
    print("learns to bury them. Capping per leaf limits the head; nothing floors the tail.")
    print(f"\nWrote {out}/  (tag={tag})")
    print("If the CE cannot reach a target recall at smaller m than the bi-encoder, it is")
    print("lengthening the agent's prompt for 50x the compute. Drop the CE tier in that")
    print("case — the retriever's own ordering becomes the shortlist. Stage 3 is unaffected.")


if __name__ == "__main__":
    main()
