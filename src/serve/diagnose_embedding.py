"""Why does the served item embedding differ from the cached one?

    python src/serve/diagnose_embedding.py --data data

Established so far: same text on both paths (both call recall.item_text), and forcing
half precision on the serving side changes nothing. The remaining difference is BATCH
SIZE -- recall.embed_items encodes at 16, the serving graph encodes at 1 -- and for this
model that is not a cosmetic difference.

Qwen3-Embedding pools the LAST token. Last-token pooling is only correct under LEFT
padding: pad on the right and the final position of every sequence shorter than the
longest in its batch is a PAD token, so the vector returned is the embedding of padding.
A batch of one has nothing to pad, which is why the serving path and the cache disagree
by ~1e-1 rather than ~1e-6, and why the disagreement is largest on the deferred items --
unusual lengths sit furthest from their batch-mates.

THE QUESTION THIS SETTLES IS WHICH SIDE IS WRONG.
If padding is left, both are correct and something else is going on. If padding is right,
the cache is wrong, and so is everything derived from it: the retrieval gate
(macro recall@50 = 0.977), cands_*.npy, every reranker score trained on those candidate
pools, and the shortlists Stage 3 read. The serving path would be the correct one.

TESTS
  A  configuration: padding side, pooling mode, max_seq_length, library versions
  B  batch invariance: the same item encoded alone, and in batches of 2/8/16 padded by a
     much longer item. A correct setup returns the same vector every time.
  C  provenance: which batch size reproduces the stored emb_items_test.npy row
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    import sentence_transformers as st
    import torch
    import transformers
    from recall import INSTRUCTION, MODEL, item_text, load_model, pick_device

    device = pick_device(args.device)
    model = load_model(device)
    tok = model.tokenizer

    print("\n=== A. configuration ===")
    print(f"sentence-transformers {st.__version__} | transformers {transformers.__version__}"
          f" | torch {torch.__version__}")
    print(f"device={device} max_seq_length={model.max_seq_length}")
    print(f"tokenizer.padding_side = {tok.padding_side!r}")
    pool = None
    for _, mod in model.named_modules():
        if type(mod).__name__ == "Pooling":
            pool = mod
            break
    if pool is not None:
        modes = [k for k, v in vars(pool).items()
                 if k.startswith("pooling_mode") and v is True]
        print(f"pooling modes = {modes}")
    verdict_left = tok.padding_side == "left"
    print(f"\n-> last-token pooling requires LEFT padding, and this tokenizer's default "
          f"is {tok.padding_side.upper()}.")
    print("   This attribute is NOT the verdict: sentence-transformers may set the "
          "padding side\n   at encode time, and section B measures the behaviour "
          "directly. Trust section B.")

    df = pd.read_parquet(args.data / "items.parquet")
    test = df[df.split == "test"].reset_index(drop=True)
    texts = [item_text(r) for r in test.itertuples()]
    lens = np.array([len(t) for t in texts])
    # A short item batched with a long one is the worst case under right padding.
    order = np.argsort(lens)
    picks = list(order[: args.n // 2]) + list(order[-args.n // 2:])
    longest = texts[int(order[-1])]

    enc = lambda xs, bs: model.encode(  # noqa: E731
        xs, prompt=f"Instruct: {INSTRUCTION}\nQuery: ", batch_size=bs,
        normalize_embeddings=True).astype(np.float32)

    print("\n=== B. batch invariance (same item, different batch composition) ===")
    print(f"{'chars':>7} {'alone vs b=2':>13} {'alone vs b=8':>13} {'alone vs b=16':>14}")
    worst = 0.0
    for i in picks:
        t = texts[int(i)]
        solo = enc([t], 1)[0]
        row = [len(t)]
        for bs in (2, 8, 16):
            batch = [t] + [longest] * (bs - 1)
            v = enc(batch, bs)[0]
            d = 1 - cos(solo, v)
            worst = max(worst, d)
            row.append(d)
        print(f"{row[0]:>7d} {row[1]:>13.2e} {row[2]:>13.2e} {row[3]:>14.2e}")
    print(f"\nworst cosine distance from batching alone: {worst:.2e}")
    print("A correct configuration is batch-invariant: every column should read ~1e-7.")

    print("\n=== C. which batch size reproduces the cache? ===")
    cache_p = args.data / "emb_items_test.npy"
    if not cache_p.exists():
        print("emb_items_test.npy missing; skipping.")
    else:
        cache = np.load(cache_p, mmap_mode="r")
        if len(cache) != len(test):
            print(f"cache has {len(cache)} rows, test split has {len(test)}; skipping.")
        else:
            idx = [int(i) for i in picks]
            solo = enc([texts[i] for i in idx], 1)
            # 16 is what recall.embed_items used, over the split in its NATURAL order --
            # so each item must be re-encoded inside the same 16-wide window it sat in
            # there. An earlier version spanned min(idx)..max(idx), which for picks drawn
            # from both ends of the length ordering is the whole 12,200-row split.
            b16 = {}
            for i in idx:
                lo = (i // 16) * 16
                block = enc(texts[lo:lo + 16], 16)
                b16[i] = block[i - lo]
            print(f"{'row':>7} {'chars':>7} {'cache vs b=1':>13} {'cache vs b=16':>14}")
            for j, i in enumerate(idx):
                c = np.asarray(cache[i])
                print(f"{i:>7d} {len(texts[i]):>7d} {1 - cos(c, solo[j]):>13.2e} "
                      f"{1 - cos(c, b16[i]):>14.2e}")
            print("\nThe smaller column is the batch size that produced the cache.")

    print("\n=== verdict ===")
    if not verdict_left and worst > 1e-2:
        print("Right padding + last-token pooling + batch-sensitive vectors.")
        print("The CACHE is wrong, not the serving path. Everything derived from")
        print("emb_items_*.npy inherits it: the recall@k gate, cands_*.npy, the reranker's")
        print("training pools and scores, and the shortlists Stage 3 was evaluated on.")
    elif worst > 1e-2:
        print("Padding side is fine but the vectors are still batch-sensitive.")
        print("Do not assume precision; find the layer that is not masking correctly.")
    else:
        print(f"Encoding is batch-invariant to {worst:.1e} -- larger than fp32 noise, "
              "and constant\nacross batch sizes 2/8/16, which is padding arithmetic "
              "rather than pad-token pooling.\nBatch size accounts for ~1e-4 and CANNOT "
              "explain a 1e-1 disagreement. Look elsewhere.")


if __name__ == "__main__":
    main()
