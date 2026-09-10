"""Build step 4: the baselines the cascade has to beat.

Run this before believing any result above it. Two floors:

  majority   predict the single most frequent leaf for everything. On ABO that scores
             ~53% micro, which is why accuracy is not a usable metric here and every
             number in this repo is macro-averaged.

  tfidf+svm  character and word n-grams into a linear one-vs-rest SVM. No GPU, no
             pretrained anything, ~20 minutes. If this matches the cascade on head, the
             expensive machinery is only earning its place on the tail — which is a true
             and useful statement, and much better said by you than by an interviewer.

Same splits, same strata, same macro averaging as everything else, so the numbers are
directly comparable to stage1.py and recall.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC

from recall import item_text

STRATA = ["head", "torso", "tail", "few_shot", "zero_shot"]


def report(name: str, pred: np.ndarray, ev: pd.DataFrame) -> pd.DataFrame:
    gold = ev.product_type.to_numpy()
    hit = pred == gold
    rows = []
    for st in STRATA + ["ALL"]:
        m = np.ones(len(ev), bool) if st == "ALL" else (ev.stratum == st).to_numpy()
        if m.sum() == 0:
            continue
        rows.append({
            "stratum": st,
            "n": int(m.sum()),
            "micro": float(hit[m].mean()),
            "macro": float(pd.Series(hit[m]).groupby(gold[m]).mean().mean()),
        })
    t = pd.DataFrame(rows).set_index("stratum")
    print(f"\n=== {name} ===")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    return t.assign(model=name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--fast", action="store_true",
                    help="SGDClassifier instead of LinearSVC — minutes instead of tens of minutes, "
                         "slightly worse. Use it for a first look.")
    ap.add_argument("--split", default="val+test", choices=["val", "test", "val+test"],
                    help="which split to score. Use 'test' to compare against pipeline.py — "
                         "a baseline scored on val+test cannot sit in the same table as a "
                         "cascade scored on test.")
    args = ap.parse_args()

    df = pd.read_parquet(args.data / "items.parquet")
    train = df[df.split == "train"]
    ev = df[df.split.isin(args.split.split("+"))].reset_index(drop=True)
    print(f"train {len(train):,} / eval {len(ev):,} ({args.split}) · "
          f"{train.product_type.nunique()} trainable leaves of {df.product_type.nunique()}")

    Xtr_txt = [item_text(r) for r in train.itertuples()]
    Xev_txt = [item_text(r) for r in ev.itertuples()]
    ytr = train.product_type.to_numpy()

    out = []
    majority = train.product_type.value_counts().idxmax()
    out.append(report(f"majority ({majority})", np.full(len(ev), majority), ev))

    # Word n-grams catch product nouns; char n-grams survive the typos, spacing and
    # code-mixing that a Korean corpus will be full of. Cheap insurance, and the union
    # is what makes a linear model competitive at all on short titles.
    vec = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        max_features=300_000, sublinear_tf=True),
    )
    print("\nvectorising…", flush=True)
    Xtr = vec.fit_transform(Xtr_txt)
    Xev = vec.transform(Xev_txt)
    print(f"  {Xtr.shape[1]:,} features, {Xtr.nnz / Xtr.shape[0]:.0f} nonzero/doc")

    clf = (SGDClassifier(loss="hinge", alpha=1e-5, max_iter=15, tol=1e-4, random_state=17)
           if args.fast else LinearSVC(C=0.5, max_iter=3000, random_state=17))
    print(f"fitting {type(clf).__name__} over {len(set(ytr))} classes…", flush=True)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xev)
    out.append(report(f"tfidf + {type(clf).__name__}", yp, ev))

    # Per-item predictions, so this baseline can be compared to the cascade with a paired
    # test rather than by eyeballing two aggregate tables. An aggregate row cannot tell
    # you whether a 0.034 gap is real.
    tag = args.split.replace("+", "_")
    preds = args.data / f"baseline_preds_{tag}.csv"
    pd.DataFrame({"item_id": ev.item_id.to_numpy(), "stratum": ev.stratum.to_numpy(),
                  "gold": ev.product_type.to_numpy(), "pred": yp}).to_csv(preds, index=False)
    print(f"Wrote {preds}")

    res = pd.concat(out).reset_index()
    res.insert(0, "split", args.split)
    path = args.data / f"baselines_{args.split.replace('+', '_')}.csv"
    res.to_csv(path, index=False)
    print(f"\nWrote {path}")
    print("\nCompare against stage1.py's accepted accuracy and recall.py's macro@1.")
    print("A linear model that matches the cascade on head is not a failure — it is the")
    print("finding that the cascade's value is concentrated in the tail. Say it first.")


if __name__ == "__main__":
    main()
