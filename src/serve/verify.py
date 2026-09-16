"""Does the served path reproduce the offline arrays?

    python src/serve/verify.py --data data --n 400 --deferred-only

This is the check that the serving layer did not quietly become a different model.

ALIGNMENT IS BY ROW POSITION, NOT BY item_id.
ABO ships 716 duplicate item_ids, 56 of them filed under two different leaves. An earlier
version of this script built {item_id: row} dictionaries, which silently keeps the LAST
occurrence -- so roughly 6% of comparisons scored one item's served output against a
different item's cached vector, and reported the result as a divergence. Both slices are
therefore indexed off positions in the ORIGINAL items.parquet row order, which is what
the .npy files were written in and is unique by construction.

stage3_agent.compose and sweep.compose align the same two slices with item_id dicts and
inherit the same collapse. It does not corrupt their numbers the way it corrupted this
script's -- they intersect the slices rather than comparing vectors elementwise -- but it
is the same latent flaw and worth fixing there too.

WHAT IS COMPARED
  argmax     S1's predicted class index, against probs_<s1_tag>.npy
  p_max      S1's confidence, as max absolute deviation
  defer      the routing decision -- the only difference that changes which tier answers
  embedding  the served item vector against emb_items_test.npy, as cosine distance
  shortlist  the ten leaves Stage 3 would read, against the stored cands/scores

SHORTLIST IS REPORTED THREE WAYS, BECAUSE EXACT EQUALITY IS THE WRONG TEST.
'fused' interleaves the retriever's ordering with the cross-encoder's, so one swapped
pair marks a whole row unequal. What matters, in descending order:

  membership  which ten leaves Stage 3 can choose from, and above all whether the GOLD
              leaf is among them. A leaf that is not reachable cannot be selected.
  top-1       S2's own answer, and the fallback on an abstention or a parse failure.
  order       matters only through the agent, and stage3_agent shuffles before sending,
              so a reordering here is washed out before the model sees it.

Stage 3 is not called. Provider is forced to 'none'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from shortlist import build_shortlist


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, cen - half), min(1.0, cen + half)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--retriever-fp16", action="store_true")
    ap.add_argument("--deferred-only", action="store_true",
                    help="sample only items the OFFLINE run deferred and that are in the "
                         "test split -- every such item yields a comparable shortlist")
    args = ap.parse_args()

    from serve.graph import CascadeRuntime
    rt = CascadeRuntime.load(args.data, device=args.device, provider="none",
                             retriever_fp16=args.retriever_fp16)
    c = rt.cascade
    print(f"device={c.device} fp16={c.fp16} retriever_fp16={args.retriever_fp16} "
          f"tau={c.cfg.tau} k={c.cfg.k}\n")

    df = pd.read_parquet(args.data / "items.parquet")
    probs = np.load(args.data / "stage1" / f"probs_{c.cfg.s1_tag}.npy")
    cands = np.load(args.data / "stage2" / "cands_test.npy")
    scores = np.load(args.data / "stage2" / f"scores_test_{c.cfg.s2_tag}.npy")
    leaves = json.loads((args.data / "stage2" / "leaves.json").read_text())

    # Position of each original row within each saved slice, or -1. Built from the row
    # ORDER the slices were written in, so duplicate item_ids cannot collapse.
    s1_rows = np.where(df.split.isin(["val", "test"]).to_numpy())[0]
    s2_rows = np.where((df.split == "test").to_numpy())[0]
    assert len(s1_rows) == len(probs), f"S1 slice {len(s1_rows)} != probs {len(probs)}"
    assert len(s2_rows) == len(scores) == len(cands), "S2 slice does not match its arrays"
    s1_at = np.full(len(df), -1);  s1_at[s1_rows] = np.arange(len(s1_rows))
    s2_at = np.full(len(df), -1);  s2_at[s2_rows] = np.arange(len(s2_rows))

    off_short = build_shortlist(cands, scores, c.cfg.shortlist, c.cfg.shortlist_mode)
    emb_store = None
    ep = args.data / "emb_items_test.npy"
    if ep.exists() and len(np.load(ep, mmap_mode="r")) == len(s2_rows):
        emb_store = np.load(ep, mmap_mode="r")

    arg_all, pmax_all = probs.argmax(1), probs.max(1)
    off_defer_all = (arg_all == c.other_i) | (pmax_all < c.cfg.tau)

    pool = s1_rows
    if args.deferred_only:
        d = off_defer_all[s1_at[s1_rows]]
        pool = s1_rows[d & (s2_at[s1_rows] >= 0)]
        print(f"sampling from the {len(pool):,} items the offline run deferred within "
              f"the test split\n")
    rng = np.random.default_rng(args.seed)
    take = rng.choice(pool, size=min(args.n, len(pool)), replace=False)

    from recall import INSTRUCTION

    n_arg = n_defer = n_cmp = n_exact = n_top1 = 0
    gold_lost = gold_gained = gold_in_off = 0
    overlaps: list[int] = []
    dp = dcos = 0.0

    for r in take:
        row = df.iloc[int(r)]
        out = rt.classify({"title": row.title, "brand": row.brand, "bullets": row.bullets})

        i1 = s1_at[r]
        off_arg, off_p = int(arg_all[i1]), float(pmax_all[i1])
        off_defer = bool(off_defer_all[i1])
        srv_defer = out["stage"] != "S1"

        if out["stage"] == "S1" and c.s1_classes.index(out["leaf"]) != off_arg:
            n_arg += 1
        n_defer += srv_defer != off_defer
        if out["s1_confidence"] is not None:
            dp = max(dp, abs(out["s1_confidence"] - off_p))

        j = s2_at[r]
        if not (srv_defer and off_defer and j >= 0 and out["shortlist"]):
            continue
        n_cmp += 1
        off = [leaves[x] for x in off_short[j]]
        srv = out["shortlist"]
        n_exact += srv == off
        n_top1 += srv[0] == off[0]
        overlaps.append(len(set(srv) & set(off)))
        g = row.product_type
        in_off, in_srv = g in off, g in srv
        gold_in_off += in_off
        gold_lost += in_off and not in_srv
        gold_gained += in_srv and not in_off
        if emb_store is not None:
            e = c._bi().encode([out["text"]], prompt=f"Instruct: {INSTRUCTION}\nQuery: ",
                               batch_size=1, normalize_embeddings=True)
            dcos = max(dcos, float(1.0 - np.dot(e[0], np.asarray(emb_store[j]))))

    n = len(take)
    size = int(c.cfg.shortlist)
    print(f"{'items compared':<34}{n:>8d}")
    print(f"{'S1 argmax mismatches':<34}{n_arg:>8d}")
    print(f"{'max |p_max| deviation':<34}{dp:>8.2e}")
    print(f"{'defer decision flips':<34}{n_defer:>8d}   <- changes which tier answers")
    if emb_store is not None:
        print(f"{'max item-embedding cos distance':<34}{dcos:>8.2e}")
    print(f"{'shortlists compared':<34}{n_cmp:>8d}")
    if n_cmp:
        lo, hi = wilson(gold_lost, max(gold_in_off, 1))
        print(f"{'  same members (any order)':<34}{sum(o == size for o in overlaps):>8d}"
              f" / {n_cmp}")
        print(f"{'  same top-1 (S2 fallback)':<34}{n_top1:>8d} / {n_cmp}")
        print(f"{'  identical ordering':<34}{n_exact:>8d} / {n_cmp}"
              "   <- washed out by the shuffle")
        print(f"{'  mean members shared':<34}{np.mean(overlaps):>8.2f} / {size}")
        print(f"{'  gold reachable offline':<34}{gold_in_off:>8d} / {n_cmp}")
        print(f"{'  gold LOST by served path':<34}{gold_lost:>8d}"
              "          <- the only membership change that costs accuracy")
        print(f"{'  gold GAINED by served path':<34}{gold_gained:>8d}")
        print(f"{'  gold-loss rate (95% Wilson)':<34}"
              f"{gold_lost}/{gold_in_off} = {gold_lost / max(gold_in_off, 1):.3%}"
              f"  [{lo:.2%}, {hi:.2%}]")
        print(f"{'  => end-to-end micro impact <=':<34}{0.165 * hi:.3%} of all items")

    ok = n_defer == 0 and n_arg == 0 and (n_cmp == 0 or gold_lost <= gold_gained)
    print("\nServed path reproduces the offline arrays: same tier decisions, no net loss "
          "of reachable gold leaves." if ok else
          "\nRouting or reachable-gold diverges. Check the embedding line: large cosine "
          "distance points at the retriever, small points at the fusion.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
