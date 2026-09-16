"""The accuracy-vs-cost frontier of the S1 gate, computed from artefacts already on disk.

    python src/sweep.py --data data
    python src/sweep.py --data data --csv data/sweep.csv

WHY THIS COSTS NOTHING. The gate is

    defer  <=>  argmax == __OTHER__  OR  p_max < tau

Lowering tau below the operating point can only make FEWER items defer, so the escalated
set at any tau <= 0.900 is a strict subset of the 2,015 items already answered in
stage3_fused.csv. Every point on the curve is a replay. Raising tau above 0.900 is a
superset and would need new API calls; this script refuses to go there rather than
silently scoring those items as if Stage 3 had answered them.

WHAT THE CURVE IS FOR. The cascade reports one operating point. That point was fitted on
val for a 98% precision target on S1 -- a decision about S1 in isolation, made before
Stage 3 existed. Whether it is the right point for the END-TO-END system, once the cost of
Stage 3 is on the table, is a different question and this is the plot that answers it.

Read the macro column. Micro is close to a report on CELLULAR_PHONE_CASE.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shortlist import build_shortlist

OTHER = "__OTHER__"
ABSTAIN = "NONE_OF_THESE"
STRATA = ["head", "torso", "tail", "few_shot", "zero_shot"]
SEED = 17

# $ per million tokens, standard (non-batch) rates. Batch halves both.
PRICES = {"sonnet45": (3.0, 15.0), "sonnet5": (2.0, 10.0), "haiku45": (1.0, 5.0)}


def compose(data: Path, s1_tag: str, s2_tag: str):
    """Rebuild the evaluation slice exactly as stage3_agent.compose does.

    Kept byte-identical in behaviour to that function on purpose: if the two ever
    disagree about which row is which item, this script silently scores Stage 3's
    answers against the wrong gold labels and the curve looks fine.
    """
    df = pd.read_parquet(data / "items.parquet")
    s1_ev = df[df.split.isin(["val", "test"])].reset_index(drop=True)
    probs = np.load(data / "stage1" / f"probs_{s1_tag}.npy")
    classes = json.loads((data / "stage1" / f"classes_{s1_tag}.json").read_text())
    assert len(s1_ev) == len(probs), f"S1 slice {len(s1_ev)} != probs {len(probs)}"

    scores = np.load(data / "stage2" / f"scores_test_{s2_tag}.npy")
    cands = np.load(data / "stage2" / "cands_test.npy")
    s2_ev = df[df.split == "test"].reset_index(drop=True)
    if len(s2_ev) != len(scores):
        s2_ev = s2_ev.sample(len(scores), random_state=SEED).reset_index(drop=True)
    assert len(s2_ev) == len(scores) == len(cands), "S2 slice does not match its arrays"

    s1_pos = {iid: i for i, iid in enumerate(s1_ev.item_id)}
    keep = [(j, s1_pos[iid]) for j, iid in enumerate(s2_ev.item_id) if iid in s1_pos]
    assert keep, "no overlap between the S1 and S2 slices"
    j2, i1 = np.array([k[0] for k in keep]), np.array([k[1] for k in keep])
    ev = s2_ev.iloc[j2].reset_index(drop=True)
    return ev, probs[i1], classes, cands[j2], scores[j2]


def macro(hit: np.ndarray, gold: np.ndarray) -> float:
    return pd.Series(hit).groupby(gold).mean().mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--s1-tag", default="head69")
    ap.add_argument("--s2-tag", default="ft2")
    ap.add_argument("--s3", type=Path, default=None,
                    help="Stage 3 results csv (default data/stage3_<mode>.csv)")
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--shortlist-mode", default="fused")
    ap.add_argument("--tau-max", type=float, default=0.900,
                    help="the tau Stage 3 was actually run at. Points above it are not "
                         "replayable and are refused.")
    ap.add_argument("--steps", type=int, default=19)
    ap.add_argument("--in-tokens", type=float, default=1682.0,
                    help="mean input tokens per escalated item. The default was once "
                         "930, derived from characters at an English prose ratio; "
                         "count_tokens measured 1,682. The candidate block is "
                         "ALL_CAPS_UNDERSCORE leaf ids and taxonomy paths, which "
                         "tokenise about twice as badly as prose. Measure exactly with "
                         "`python src/serve/llm.py --count-tokens` and pass the result.")
    ap.add_argument("--out-tokens", type=float, default=30.0)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    s3_path = args.s3 or args.data / f"stage3_{args.shortlist_mode}.csv"
    s3 = pd.read_csv(s3_path)
    ran_at = float(s3.tau.iloc[0]) if "tau" in s3 else args.tau_max
    if args.tau_max > ran_at + 1e-9:
        raise SystemExit(
            f"--tau-max {args.tau_max} exceeds the tau Stage 3 ran at ({ran_at}).\n"
            "Items escalated above that point have no cached answer; scoring them would "
            "invent one. Re-run stage3_agent.py at the higher tau first.")

    # ABO ships 716 duplicate item_ids. dict() keeps the last, which is what
    # pipeline.evaluate does -- mirrored here so the two agree rather than diverging.
    s3_pred = dict(zip(s3.item_id.astype(str), s3.pred))

    ev, probs, classes, cands, scores = compose(args.data, args.s1_tag, args.s2_tag)
    gold = ev.product_type.to_numpy()
    stratum = ev.stratum.to_numpy()
    ids = ev.item_id.astype(str).to_numpy()
    other_i = classes.index(OTHER) if OTHER in classes else -1

    arg, pmax = probs.argmax(1), probs.max(1)
    s1_leaf = np.array([classes[i] for i in arg], dtype=object)

    leaves = json.loads((args.data / "stage2" / "leaves.json").read_text())
    short = build_shortlist(cands, scores, args.shortlist, args.shortlist_mode)
    s2_leaf = np.array([leaves[row[0]] for row in short], dtype=object)

    # Stage 3's answer where it has one; None is an abstention, which is a refusal to
    # label and therefore a miss, not a fallback to S2. Matches pipeline.evaluate.
    s3_leaf = np.array(
        [(None if s3_pred.get(i) == ABSTAIN else s3_pred.get(i)) for i in ids], dtype=object)
    has_s3 = np.array([i in s3_pred for i in ids])

    print(f"eval set {len(ev):,} items, {len(set(gold))} leaves | "
          f"Stage 3 answers cached for {has_s3.sum():,}")
    print(f"replayable range: tau <= {ran_at:g}\n")

    rows = []
    for tau in np.linspace(args.tau_max - (args.steps - 1) * 0.05, args.tau_max, args.steps):
        tau = round(float(tau), 4)
        if tau <= 0:
            continue
        deferred = (arg == other_i) | (pmax < tau)
        # An item that defers but has no cached Stage 3 answer would be scored on S2's
        # top-1, quietly understating the cascade. At tau <= ran_at this must be empty.
        gap = int((deferred & ~has_s3).sum())
        assert gap == 0, f"tau={tau}: {gap} deferred items have no cached Stage 3 answer"

        leaf = np.where(deferred, np.where(has_s3, s3_leaf, s2_leaf), s1_leaf)
        hit = leaf == gold
        n_esc = int(deferred.sum())
        row = {"tau": tau, "escalated": n_esc, "escalated_pct": n_esc / len(ev),
               "micro": hit.mean(), "macro": macro(hit, gold),
               "abstained": int((deferred & (s3_leaf == None)).sum())}  # noqa: E711
        for st in STRATA:
            k = stratum == st
            row[f"macro_{st}"] = macro(hit[k], gold[k]) if k.any() else np.nan
        for name, (pi, po) in PRICES.items():
            row[f"usd_{name}"] = (n_esc * args.in_tokens * pi
                                  + n_esc * args.out_tokens * po) / 1e6
        rows.append(row)

    out = pd.DataFrame(rows)
    base = out[np.isclose(out.tau, ran_at)]

    print(f"{'tau':>6s} {'esc':>6s} {'esc%':>6s} {'micro':>7s} {'macro':>7s} "
          f"{'zero_s':>7s} {'few_s':>7s} {'abst':>5s} {'$s4.5':>7s} {'$batch':>7s}")
    for r in out.to_dict("records"):
        mark = "  <- operating point" if np.isclose(r["tau"], ran_at) else ""
        print(f"{r['tau']:>6.2f} {r['escalated']:>6d} {r['escalated_pct']:>6.1%} "
              f"{r['micro']:>7.3f} {r['macro']:>7.3f} {r['macro_zero_shot']:>7.3f} "
              f"{r['macro_few_shot']:>7.3f} {r['abstained']:>5d} "
              f"{r['usd_sonnet45']:>7.2f} {r['usd_sonnet45'] / 2:>7.2f}{mark}")

    if len(base):
        b = base.iloc[0]
        print(f"\nAt the operating point: micro {b.micro:.3f}, macro {b.macro:.3f}, "
              f"{b.escalated:,} items to Stage 3.")
        # The question the curve exists to answer.
        cheap = out[out.macro >= b.macro - 0.005]
        if len(cheap):
            c = cheap.loc[cheap.escalated.idxmin()]
            if c.escalated < b.escalated:
                print(f"Cheapest tau within 0.005 macro of it: tau={c.tau:g} -- "
                      f"{c.escalated:,} escalated ({c.escalated / b.escalated:.0%} of the "
                      f"LLM spend) for macro {c.macro:.3f}.")
            else:
                print("No cheaper tau holds macro within 0.005: the operating point is "
                      "already on the knee.")
    print("\nAll points replayed from cached Stage 3 answers. Nothing spent.")

    if args.csv:
        out.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
