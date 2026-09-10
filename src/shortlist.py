"""How the agent's candidate list is built from the two Stage 2 orderings.

One implementation, imported by pipeline.py and stage3_agent.py. It lives in its own
module for a specific reason: if these two disagree about what the shortlist is, the
agent is evaluated on a different list than the one the cascade would hand it, and
nothing in either script would look wrong. That failure already happened once —
stage3_results.csv was produced against a cross-only shortlist and then folded into a
fused-mode evaluation.

Pure numpy, no torch, so importing it costs nothing.

WHY FUSED IS THE DEFAULT, measured on the full test split (12,200 items, 530 leaves),
macro recall@10 — is the gold leaf in the ten candidates the agent reads:

    stratum      bi   cross   fused
       head   0.740   0.887   0.877
      torso   0.884   0.962   0.958
       tail   0.920   0.931   0.937
   few_shot   0.958   0.825   0.952
  zero_shot   0.876   0.124   0.860
        ALL   0.919   0.760   0.931

The fine-tuned cross-encoder is worse than the frozen retriever it was meant to improve
(0.760 vs 0.919) because the 90 leaves with no training rows appear in its training
groups only as hard negatives — it is taught they are always wrong. But it is sharply
better where it has data. Interleaving keeps both: it costs 0.006 on few_shot and 0.016
on zero_shot, and buys 0.137 on head and 0.074 on torso.
"""
import numpy as np

MODES = ("fused", "cross", "bi")


def build_shortlist(cands: np.ndarray, scores: np.ndarray, m: int,
                    mode: str = "fused") -> np.ndarray:
    """Return the (n, m) candidate ids the agent should read.

    cands  (n, k) candidate ids in BI-ENCODER order — highest similarity first.
    scores (n, k) cross-encoder scores, aligned elementwise to cands.

    'bi'    keeps the retriever's ordering and ignores scores entirely.
    'cross' sorts by score — the reranker's ordering.
    'fused' walks both in lockstep taking bi, then cross, skipping anything already
            taken. So bi's top-1 is always first and cross's top-1 is second unless
            they agree, and m=10 holds roughly each model's top 5.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if m > cands.shape[1]:
        raise ValueError(f"shortlist {m} exceeds the {cands.shape[1]} candidates retrieved")

    if mode == "bi":
        return cands[:, :m].copy()
    ce = np.take_along_axis(cands, np.argsort(-scores, 1), 1)
    if mode == "cross":
        return ce[:, :m]

    out = np.empty((len(cands), m), dtype=cands.dtype)
    for r in range(len(cands)):
        seen, merged = set(), []
        for a, b in zip(cands[r], ce[r]):
            for c in (a, b):
                if c not in seen:
                    seen.add(c)
                    merged.append(c)
                    if len(merged) == m:
                        break
            if len(merged) == m:
                break
        out[r] = merged
    return out
