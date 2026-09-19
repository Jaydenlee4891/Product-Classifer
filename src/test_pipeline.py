"""Unit tests for the only genuinely new logic in pipeline.py.

Everything else in that file is glue over stages that already have their own measured
gates. Two pieces are new and therefore untested by any of those runs:

  1. shortlist.build_shortlist's 'fused' mode — an interleave-and-dedup over two
     orderings. It lives in shortlist.py; pipeline.Cascade._shortlist is a thin caller.
  2. evaluate's Stage 3 fold-in — in particular that an abstention becomes its own tier
     instead of being scored as a wrong answer inside S3.

Neither needs a model, so this runs in a second on any machine:

    python src/test_pipeline.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import evaluate                        # noqa: E402
from shortlist import build_shortlist                # noqa: E402

FAILURES = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


class shortlister:
    """Adapter so these tests read the way the callers do."""
    def __init__(self, mode, m):
        self.mode, self.m = mode, m

    def _shortlist(self, cands, scores):
        return build_shortlist(cands, scores, self.m, self.mode)


def test_shortlist():
    K, M = 8, 4
    # Row 0: the cross-encoder exactly reverses the bi-encoder — the worst case for a
    # fusion, and the only case where an interleave is visibly different from either.
    # Row 1: the two agree, so fusion must reduce to a no-op.
    cands = np.array([[10, 11, 12, 13, 14, 15, 16, 17],
                      [20, 21, 22, 23, 24, 25, 26, 27]])
    scores = np.array([[1., 2., 3., 4., 5., 6., 7., 8.],
                       [8., 7., 6., 5., 4., 3., 2., 1.]])

    print("_shortlist, mode=cross")
    out = shortlister("cross", M)._shortlist(cands, scores)
    check("row0 follows the CE ordering", list(out[0]) == [17, 16, 15, 14])
    check("row1 CE ordering equals bi ordering", list(out[1]) == [20, 21, 22, 23])
    check("shape is (n, m)", out.shape == (2, M))

    print("_shortlist, mode=bi")
    out = shortlister("bi", M)._shortlist(cands, scores)
    check("ignores the scores entirely", list(out[0]) == [10, 11, 12, 13])

    print("_shortlist, mode=fused")
    out = shortlister("fused", M)._shortlist(cands, scores)
    check("row0 alternates bi, CE", list(out[0]) == [10, 17, 11, 16])
    check("row1 dedups back to plain order", list(out[1]) == [20, 21, 22, 23])
    check("no duplicates in any row", all(len(set(r)) == len(r) for r in out))
    check("candidate dtype preserved", out.dtype == cands.dtype)
    out = shortlister("fused", K)._shortlist(cands, scores)
    check("m == k fills every slot without repeats", all(len(set(r)) == K for r in out))
    check("row0 is a permutation of its candidates", sorted(out[0]) == sorted(cands[0]))
    out = shortlister("fused", 1)._shortlist(cands[:1], scores[:1])
    check("m == 1 keeps the bi-encoder's top-1", list(out[0]) == [10])

    print("_shortlist, guards")
    # A shortlist wider than the retrieved candidates would silently pad or wrap;
    # both callers pass m and k independently, so this has to raise.
    for mode in ("bi", "cross", "fused"):
        try:
            shortlister(mode, K + 1)._shortlist(cands, scores)
            check(f"m > k raises in mode={mode}", False)
        except ValueError:
            check(f"m > k raises in mode={mode}", True)
    try:
        shortlister("crossed", M)._shortlist(cands, scores)
        check("an unknown mode raises rather than silently defaulting", False)
    except ValueError:
        check("an unknown mode raises rather than silently defaulting", True)
    # 'bi' must not hand back a view into the caller's array.
    out = shortlister("bi", M)._shortlist(cands, scores)
    out[0, 0] = -1
    check("returned array does not alias the input", cands[0, 0] == 10)


class FakeCascade:
    """Returns a fixed predict() so evaluate's fold-in can be tested on its own."""
    def __init__(self, out):
        self.out = out

    def predict(self, df):
        return self.out


def test_evaluate_fold_in():
    df = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "product_type": ["HOME", "SHOES", "LAMP", "RUG"],
        "stratum": ["head", "torso", "tail", "few_shot"],
    })
    # evaluate mutates the frame predict() returned, so one call shows both the printed
    # table and the post-fold state.
    out = pd.DataFrame({
        "leaf": ["HOME", "WRONG", "WRONG", "RUG"],
        "stage": ["S1", "S2", "S2", "S2"],
        "confidence": [0.99, np.nan, np.nan, np.nan],
        "shortlist": [None, ["SHOES"], ["LAMP"], ["RUG"]],
    })
    # b: the agent fixes it. c: the agent abstains. d: outside the S3 run entirely.
    s3 = pd.DataFrame({"item_id": ["b", "c"], "pred": ["SHOES", "NONE_OF_THESE"]})

    print("evaluate, Stage 3 fold-in")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate(FakeCascade(out), df, s3)
    text = buf.getvalue()

    check("b became a correct S3 decision", out.leaf[1] == "SHOES" and out.stage[1] == "S3")
    # pandas 2 keeps the assigned None on an object column; pandas 3 infers a str column
    # and stores nan. Both compare False against gold, which is the behaviour that
    # matters, so the test asserts the invariant rather than the representation.
    check("c became an abstention carrying no leaf",
          pd.isna(out.leaf[2]) and out.stage[2] == "S3-abstain")
    check("d untouched by a run that never saw it",
          out.stage[3] == "S2" and out.leaf[3] == "RUG")
    check("abstention appears as its own tier in the table", "S3-abstain" in text)
    check("abstention is not counted as an S3 success",
          "S3 " in text and "1.000" in text)


def main():
    test_shortlist()
    test_evaluate_fold_in()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
