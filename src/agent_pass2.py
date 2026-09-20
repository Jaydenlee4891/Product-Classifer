"""Evaluate the second-pass agent on the items Stage 3 abstained on.

    python src/agent_pass2.py --dry-run     # free: prompts + a probe of the search tool
    python src/agent_pass2.py --limit 5     # live: a handful of items, to see the cost
    python src/agent_pass2.py               # live: every abstention (153 at the shipped config)

WHAT IS MEASURED. The residual is every escalated item whose stored Stage 3 answer is
NONE_OF_THESE. All of those were counted as misses, so any correct label the agent
produces is a pure gain in accuracy. What it can NOT do is lose accuracy -- but it can
turn a flagged refusal into a confident wrong label, which is worse in kind, so that is
reported on its own line rather than folded into a headline.

The residual is split by whether the gold leaf was in Stage 3's shortlist:

  answerable    gold WAS in the shortlist. The agent could have picked it; this measures
                whether a second look and the ability to compare siblings changes the
                answer. (105 of the 153.)
  unanswerable  gold was NOT in the shortlist. Only search can help. This is the retrieval
                ceiling the agent exists to break. (48 of the 153.)

WHAT --dry-run PROBES. It runs the real search tool over the real residual with a NAIVE
query -- the item's own title -- and reports how often the gold leaf appears in the top 8.
That is the floor for what an agent that writes better queries can find. It is not an
agent result, and it is labelled as such in the output.

COST. Live runs make up to (max_steps) model calls per item, each carrying the growing
history. Run --limit 5 first and read the token counts it prints before running all of
them; nothing here estimates the total for you.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage3_agent as S3                                        # noqa: E402
from shortlist import build_shortlist                            # noqa: E402

SEARCH_HITS = 8


def residual(data: Path, args):
    """Residual items with everything needed to run and score them."""
    ev, deferred, cands, scores = S3.compose(data, args.s1_tag, args.s2_tag, args.tau)
    idx = np.where(deferred)[0]
    shortlists = build_shortlist(cands, scores, args.shortlist, "fused")

    base = pd.read_csv(data / "stage3_fused.csv")
    # compose() aligns by reconstruction, so prove the alignment instead of assuming it:
    # a silent shift here would hand every item the wrong shortlist and nothing downstream
    # would look wrong.
    assert len(base) == len(idx), f"stage3_fused.csv has {len(base)} rows, compose gave {len(idx)}"
    assert (base.item_id.astype(str).to_numpy()
            == ev.item_id.iloc[idx].astype(str).to_numpy()).all(), "item order mismatch"

    pos = np.where(base.pred == S3.NONE)[0]
    if args.limit:
        pos = pos[: args.limit]
    rows = []
    for p in pos:
        i = idx[p]
        rows.append({
            "item_id": str(base.item_id.iloc[p]), "stratum": base.stratum.iloc[p],
            "gold": base.gold.iloc[p], "gold_in_shortlist": bool(base.gold_in_shortlist.iloc[p]),
            "item": {"title": ev.title.iloc[i], "brand": ev.brand.iloc[i],
                     "bullets": ev.bullets.iloc[i]},
            "shortlist": shortlists[i],
        })
    return rows, base


def build_index(data: Path, args):
    from pipeline import Cascade
    from serve.agent import TaxonomyIndex, embed_from_cascade
    cascade = Cascade.load(data, device=args.device)
    docs_all = json.loads((data / "label_docs.json").read_text())
    return TaxonomyIndex(cascade.leaves, docs_all, cascade.label_emb,
                         embed_from_cascade(cascade)), cascade


def dry_run(rows, index, args):
    from serve.agent import AGENT_SYSTEM
    r = rows[0]
    cands = S3.render_candidates(r["shortlist"], index.docs, index.leaves,
                                 random.Random(S3.SEED), shuffle=True)
    row = SimpleNamespace(**r["item"])
    print("\n--- system prompt ---\n" + AGENT_SYSTEM)
    print("\n--- first user message (item 0) ---\n" + S3.render_item(row) + "\n\n" + cands)
    print(f"\ngold for that item: {r['gold']}  (in shortlist: {r['gold_in_shortlist']})")

    print(f"\n--- search-tool probe: naive query = the item's own title, top {SEARCH_HITS} ---")
    hit = {True: [0, 0], False: [0, 0]}
    for r in rows:
        _, ids = index.search(r["item"]["title"], SEARCH_HITS)
        hit[r["gold_in_shortlist"]][0] += r["gold"] in ids
        hit[r["gold_in_shortlist"]][1] += 1
    for name, k in (("answerable   (gold was in the shortlist)", True),
                    ("unanswerable (gold was NOT in the shortlist)", False)):
        h, n = hit[k]
        print(f"  {name}: gold in search results for {h}/{n}"
              + (f" = {h / n:.1%}" if n else ""))
    print("  This is a floor for what better queries can find. It is not an agent result.")
    print(f"\n{len(rows)} residual items. Nothing sent to any model. Nothing spent.")


def live_run(rows, index, cascade, args):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")
    from serve.agent import AnthropicMessages, Stage3Agent
    from serve.llm import Stage3, Stage3Config
    s3 = Stage3(args.data, index.leaves, Stage3Config(provider="anthropic", model=args.model))
    agent = Stage3Agent(index, AnthropicMessages(model=args.model, headers=s3._headers()),
                        max_steps=args.max_steps)
    print(f"Running {len(rows)} items, up to {args.max_steps} model calls each ...", flush=True)

    out = []
    for n, r in enumerate(rows, 1):
        res = agent.run(r["item"], np.array(r["shortlist"]))
        out.append({
            "item_id": r["item_id"], "stratum": r["stratum"], "gold": r["gold"],
            "gold_in_shortlist": r["gold_in_shortlist"], "pred": res.leaf,
            "confidence": res.confidence, "stopped": res.stopped,
            "steps": len(res.steps),
            "searches": sum(s["tool"] == "search_taxonomy" for s in res.steps),
            "input_tokens": res.usage["input"], "output_tokens": res.usage["output"],
        })
        print(f"  {n:>3d}/{len(rows)}  {res.stopped:<10s} {str(res.leaf):<28s} gold={r['gold']}",
              flush=True)
    return pd.DataFrame(out)


def report(df, base, args):
    df = df.copy()
    df["correct"] = df.pred == df.gold
    df["committed_wrong"] = df.pred.notna() & ~df.correct
    df["still_abstained"] = df.stopped == "abstained"
    df["unresolved"] = df.stopped.isin(["invalid", "no_tool"])

    print(f"\n=== second pass on {len(df)} abstained items ===")
    print(f"{'group':>13s} {'n':>4s} {'correct':>8s} {'wrong':>6s} {'abstain':>8s} {'unresolved':>11s}")
    for name, s in (("answerable", df[df.gold_in_shortlist]),
                    ("unanswerable", df[~df.gold_in_shortlist]), ("ALL", df)):
        if len(s):
            print(f"{name:>13s} {len(s):>4d} {int(s.correct.sum()):>8d} "
                  f"{int(s.committed_wrong.sum()):>6d} {int(s.still_abstained.sum()):>8d} "
                  f"{int(s.unresolved.sum()):>11d}")
    print(f"\nmean steps {df.steps.mean():.2f}, mean searches {df.searches.mean():.2f}, "
          f"tokens in/out per item {df.input_tokens.mean():.0f} / {df.output_tokens.mean():.0f}")
    print(f"{int(df.committed_wrong.sum())} items went from a flagged refusal to a confident "
          "wrong label. They were misses before too, so accuracy cannot fall, but each is "
          "worse in kind.")

    base_correct = int((base.pred == base.gold).sum())
    gain = int(df.correct.sum())
    complete = len(df) == int((base.pred == S3.NONE).sum())
    print(f"\nEscalated items answered correctly: {base_correct}/{len(base)} before"
          + (f", {base_correct + gain}/{len(base)} after ({gain:+d})." if complete else
             f". With --limit, only {len(df)} of the abstentions were run; "
             f"{gain} recovered so far. Run without --limit for the overall figure."))
    out = args.data / "stage3_agent.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--s1-tag", default="head69")
    ap.add_argument("--s2-tag", default="ft2")
    ap.add_argument("--tau", type=float, default=0.900)
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default=S3.MODEL)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N abstentions")
    ap.add_argument("--dry-run", action="store_true", help="prompts + search probe; spends nothing")
    args = ap.parse_args()

    rows, base = residual(args.data, args)
    print(f"{len(rows)} abstained items to review "
          f"({sum(r['gold_in_shortlist'] for r in rows)} answerable from the shortlist)")
    index, cascade = build_index(args.data, args)
    if args.dry_run:
        return dry_run(rows, index, args)
    report(live_run(rows, index, cascade, args), base, args)


if __name__ == "__main__":
    main()
