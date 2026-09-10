"""Build step 6: turn reranker scores into prediction SETS, and route on set size.

Split conformal prediction. Calibrate on val, evaluate on test, never the reverse.

    nonconformity   s(x,y) = 1 - p(y | x)          p = softmax over the K candidates
    threshold       q = the ceil((n+1)(1-a))/n empirical quantile of calibration scores
    prediction set  C(x) = { y in candidates : p(y|x) >= 1 - q }

    routing         |C| == 1  accept
                    |C| >  1  escalate to S3 with exactly those candidates
                    |C| == 0  escalate with the full top-K; the item is out of distribution

Two things this file is careful about, both easy to get wrong:

  THE GUARANTEE IS CONDITIONAL ON RETRIEVAL. The reranker only scores K candidates, so an
  item whose gold leaf was never retrieved cannot be covered at any alpha. Calibration
  items in that position have no score for the true class and are excluded from the
  quantile — which is precisely what makes the guarantee conditional. Overall coverage is
  therefore P(gold retrieved) x P(covered | retrieved), and this script reports all three
  numbers rather than the nominal one.

  MONDRIAN GROUPS MUST BE OBSERVABLE AT INFERENCE. Conditioning the threshold on the true
  label's frequency stratum would be circular: at inference the true label is what you are
  trying to find. --mondrian predicted groups by the stratum of the top-1 PREDICTED leaf,
  which is observable and therefore valid. Coverage is still reported by true stratum, but
  only as a diagnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = ["head", "torso", "tail", "few_shot", "zero_shot"]


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected quantile. The (n+1) is not decoration: without it the
    guarantee fails for small n, which is exactly the rare-stratum regime here."""
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def gold_position(cands: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Column of the gold leaf within each item's candidate list, or -1 if never retrieved."""
    hit = cands == gold[:, None]
    pos = np.where(hit.any(1), hit.argmax(1), -1)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--tag", default="ft2", help="which reranker scored: offshelf | ft2 | ...")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.10, 0.05, 0.01])
    ap.add_argument("--mondrian", choices=["none", "predicted"], default="none")
    args = ap.parse_args()

    d = args.data / "stage2"
    leaves = json.loads((d / "leaves.json").read_text())
    lidx = {l: i for i, l in enumerate(leaves)}
    df = pd.read_parquet(args.data / "items.parquet")
    strat_of = df.drop_duplicates("product_type").set_index("product_type").stratum.to_dict()

    parts = {}
    for name in ("val", "test"):
        p = df[df.split == name].reset_index(drop=True)
        s = np.load(d / f"scores_{name}_{args.tag}.npy")
        c = np.load(d / f"cands_{name}.npy")
        if len(p) != len(s):
            raise SystemExit(f"{name}: {len(p)} rows but {len(s)} score rows — regenerate stage2 outputs.")
        parts[name] = (p, softmax(s), c, p.product_type.map(lidx).to_numpy())

    cal, cal_p, cal_c, cal_g = parts["val"]
    tst, tst_p, tst_c, tst_g = parts["test"]

    cal_pos = gold_position(cal_c, cal_g)
    tst_pos = gold_position(tst_c, tst_g)
    print(f"Calibration: {len(cal):,} items, gold retrieved for {(cal_pos >= 0).mean():.1%}")
    print(f"Test:        {len(tst):,} items, gold retrieved for {(tst_pos >= 0).mean():.1%}")
    print("Items whose gold was never retrieved are excluded from the quantile and can")
    print("never be covered — that is the retrieval ceiling, not a calibration failure.\n")

    # group assignment must be computable at inference time
    if args.mondrian == "predicted":
        cal_grp = np.array([strat_of.get(leaves[c[p.argmax()]], "head") for c, p in zip(cal_c, cal_p)])
        tst_grp = np.array([strat_of.get(leaves[c[p.argmax()]], "head") for c, p in zip(tst_c, tst_p)])
    else:
        cal_grp = np.full(len(cal), "ALL", dtype=object)
        tst_grp = np.full(len(tst), "ALL", dtype=object)

    rows, size_rows = [], []
    for alpha in args.alpha:
        qs = {}
        for g in np.unique(cal_grp):
            m = (cal_grp == g) & (cal_pos >= 0)
            s = 1.0 - cal_p[m][np.arange(m.sum()), cal_pos[m]]
            qs[g] = conformal_q(s, alpha)
            if m.sum() < np.ceil(1 / alpha) - 1:
                print(f"  WARNING alpha={alpha} group {g}: only {m.sum()} calibration points; "
                      f"need >= {int(np.ceil(1/alpha) - 1)} for the guarantee to bind.")

        q_row = np.array([qs[g] for g in tst_grp])
        keep = tst_p >= (1.0 - q_row)[:, None]
        size = keep.sum(1)
        covered = np.zeros(len(tst), bool)
        got = tst_pos >= 0
        covered[got] = keep[got, tst_pos[got]]

        top1 = tst_c[np.arange(len(tst)), tst_p.argmax(1)]
        accept = size == 1
        acc_correct = accept & (top1 == tst_g)

        for st in STRATA + ["ALL"]:
            m = np.ones(len(tst), bool) if st == "ALL" else (tst.stratum == st).to_numpy()
            if m.sum() == 0:
                continue
            rows.append({
                "alpha": alpha, "stratum": st, "n": int(m.sum()),
                "retrieved": float(got[m].mean()),
                "cov_given_retr": float(covered[m & got].mean()) if (m & got).sum() else np.nan,
                "coverage": float(covered[m].mean()),
                "median_set": float(np.median(size[m])),
                "accept_rate": float(accept[m].mean()),
                "acc_on_accept": float(acc_correct[m].sum() / accept[m].sum()) if accept[m].sum() else np.nan,
            })
        size_rows.append({
            "alpha": alpha,
            "|C|=0 -> S3 (OOD)": float((size == 0).mean()),
            "|C|=1 -> accept": float((size == 1).mean()),
            "|C|2-5 -> S3": float(((size >= 2) & (size <= 5)).mean()),
            "|C|>5 -> S3": float((size > 5).mean()),
            "escalation": float((size != 1).mean()),
        })

    res = pd.DataFrame(rows)
    res.to_csv(d / f"conformal_{args.tag}_{args.mondrian}.csv", index=False)

    for alpha in args.alpha:
        sub = res[res.alpha == alpha].set_index("stratum").reindex(
            [s for s in STRATA + ["ALL"] if s in set(res.stratum)])
        print(f"=== alpha = {alpha}  (nominal coverage {1-alpha:.2f}) ===")
        print(sub[["n", "retrieved", "cov_given_retr", "coverage",
                   "median_set", "accept_rate", "acc_on_accept"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
        row = sub.loc["ALL"]
        gap = row.cov_given_retr - (1 - alpha)
        flag = "OK" if abs(gap) <= 0.02 else "OFF"
        print(f"  calibration check: cov|retrieved {row.cov_given_retr:.3f} vs "
              f"nominal {1-alpha:.3f}  ({gap:+.3f})  {flag}")
        print(f"  end-to-end coverage {row.coverage:.3f} = retrieval {row.retrieved:.3f} "
              f"x conditional {row.cov_given_retr:.3f}\n")

    print("=== routing / set-size distribution (test, all strata) ===")
    print(pd.DataFrame(size_rows).set_index("alpha").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nThe escalation column is the Stage 3 cost driver. Read it against section 07")
    print("of the design doc: 5% is the planning assumption, 20% is where it stops being cheap.")
    print(f"\nWrote {d}/conformal_{args.tag}_{args.mondrian}.csv")


if __name__ == "__main__":
    main()
