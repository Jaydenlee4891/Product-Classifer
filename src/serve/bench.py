"""Per-tier latency for the served cascade.

    python src/serve/bench.py --data data --n 300

The README reports accuracy per tier. This is the other half: what each tier costs in
wall time, and what the mix of tiers does to the distribution.

WHY A MEAN IS THE WRONG SUMMARY HERE.
83.5% of items are answered by one DistilBERT forward pass. The rest pay for a 0.6B
bi-encoder, fifty cross-encoder pairs, and -- with a live provider -- a synchronous LLM
call. Those differ by orders of magnitude, and tau sets the mix. A single average over
them describes no actual request. Everything below is reported per terminating tier, and
the tail is reported as p95 because that is what a caller experiences.

COLD START IS EXCLUDED, AND SEPARATELY REPORTED.
The retriever is lazy: 83.5% of items never need it, so it is not built until the first
deferred item arrives, and that item pays several seconds for a model load. Folding that
into p95 would make the tail look like a latency problem rather than a warm-up cost. A
warm-up pass runs first and its cost is printed on its own line.

Default provider is 'cached', which replays stage3_fused.csv from disk -- so the S3 row
below measures the graph's own overhead, NOT an API round trip. Use --provider anthropic
for true end-to-end numbers, at roughly a third of a cent per escalated item.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--provider", default="cached")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    from serve.graph import CascadeRuntime
    t0 = time.perf_counter()
    rt = CascadeRuntime.load(args.data, device=args.device, provider=args.provider)
    load_s = time.perf_counter() - t0
    c = rt.cascade
    print(f"device={c.device} fp16={c.fp16} provider={args.provider} "
          f"tau={c.cfg.tau}\nstartup (S1 + cross-encoder): {load_s:.2f}s")

    df = pd.read_parquet(args.data / "items.parquet")
    test = df[df.split == "test"].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(test), size=min(args.n, len(test)), replace=False)

    item = lambda r: {"title": r.title, "brand": r.brand, "bullets": r.bullets}  # noqa: E731

    # Warm up until one item has actually deferred, so the retriever's load is paid
    # before measurement rather than inside the p95 of the first sample.
    t0, warmed = time.perf_counter(), False
    for i in range(len(test)):
        w = test.iloc[i]
        if rt.classify(item(w), item_id=str(w.item_id))["stage"] != "S1":
            warmed = True
            break
    print(f"retriever cold start: {time.perf_counter() - t0:.2f}s"
          if warmed else "no deferred item found while warming up")

    per_node = defaultdict(list)
    per_stage = defaultdict(list)
    rows = []
    for i in take:
        r = test.iloc[int(i)]
        out = rt.classify(item(r), item_id=str(r.item_id))
        for t in out["trace"]:
            per_node[t["node"]].append(t["ms"])
        per_stage[out["stage"]].append(out["total_ms"])
        rows.append({"item_id": r.item_id, "stage": out["stage"],
                     "total_ms": out["total_ms"],
                     **{f"ms_{t['node']}": t["ms"] for t in out["trace"]}})

    n = len(take)
    print(f"\n=== per node, over the items that reached it ({n} requests) ===")
    print(f"{'node':<10}{'n':>6}{'p50':>9}{'p95':>9}{'mean':>9}   (ms)")
    for node in ("s1", "retrieve", "rerank", "stage3"):
        xs = per_node.get(node, [])
        if xs:
            print(f"{node:<10}{len(xs):>6}{pct(xs, 50):>9.1f}{pct(xs, 95):>9.1f}"
                  f"{np.mean(xs):>9.1f}")

    print(f"\n=== end to end, by the tier that answered ===")
    print(f"{'stage':<12}{'share':>8}{'p50':>9}{'p95':>9}{'mean':>9}   (ms)")
    for stage in ("S1", "S2", "S3", "S3-abstain"):
        xs = per_stage.get(stage, [])
        if xs:
            print(f"{stage:<12}{len(xs) / n:>8.1%}{pct(xs, 50):>9.1f}"
                  f"{pct(xs, 95):>9.1f}{np.mean(xs):>9.1f}")
    allms = [v for xs in per_stage.values() for v in xs]
    print(f"{'ALL':<12}{1.0:>8.1%}{pct(allms, 50):>9.1f}{pct(allms, 95):>9.1f}"
          f"{np.mean(allms):>9.1f}")

    esc = 1 - len(per_stage.get("S1", [])) / n
    print(f"\nescalated {esc:.1%} of requests (offline deferral at this tau: 16.5%)")
    print("The gap between the S1 row and the rest is what tau is buying and spending.")
    if args.provider == "cached":
        print("provider='cached': the stage3 row is graph overhead, not an API call.")

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
