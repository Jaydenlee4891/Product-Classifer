"""Tests for the second-pass agent loop. A scripted model stands in for the API, so this
needs no weights, no torch and no key, and spends nothing:

    python src/serve/test_agent.py

What is under test is the LOOP, not the model: that it stops, that it enforces what the
prompt only asks for, that it fails closed, and that it changes nothing for items that
never reach it. Whether it recovers real items is a live measurement (agent_pass2.py),
and no test here can say.

  1. the taxonomy tools: nearest-first search, a hard cap on k, siblings, unknown ids
  2. the loop: look -> answer, tool results fed back, tool errors returned to the model
  3. the stopping criterion: the last turn is FORCED to record_category, and a rejected
     forced answer ends the loop rather than spinning
  4. guards the code enforces: no abstaining before a search, no invented ids
  5. the graph: only a Stage 3 ABSTENTION reaches the agent; a failure inside it leaves
     the item abstained; nothing changes when it is not configured
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Importing test_graph installs the stub for `recall` (torch at module scope) and gives us
# the same stub cascade and Stage 3 the routing tests use.
from serve import test_graph as TG                            # noqa: E402
from serve.agent import (AGENT_SYSTEM, MAX_STEPS, SEARCH_K_MAX, Reply,        # noqa: E402
                         Stage3Agent, TaxonomyIndex)
from serve.graph import CascadeRuntime, build_graph          # noqa: E402
from serve.llm import Decision                               # noqa: E402

NONE = "NONE_OF_THESE"
FAILURES = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- fixtures

NAMES = ["MOUSE", "KEYBOARD", "MOUSE_PAD", "TOILET_PAPER_HOLDER", "TOWEL_RING", "SHOES",
         "SANDALS", "LOOSE_LEAF"]
PATHS = {
    "MOUSE": "Computers > Peripherals > Mice",
    "KEYBOARD": "Computers > Peripherals > Keyboards",
    "MOUSE_PAD": "Computers > Peripherals > Mouse Pads",
    "TOILET_PAPER_HOLDER": "Hardware > Bathroom > Toilet Paper Holders",
    "TOWEL_RING": "Hardware > Bathroom > Towel Rings",
    "SHOES": "Apparel > Footwear > Shoes",
    "SANDALS": "Apparel > Footwear > Sandals",
    "LOOSE_LEAF": None,             # 39 of the real 530 leaves have no path at all
}


def docs_all():
    d = {}
    for n in NAMES:
        base = n.lower().replace("_", " ")
        p = f"{base}. Category: {PATHS[n]}." if PATHS[n] else base
        d[n] = {"v2_path": p, "v3_proto": p + f" Examples: Acme {base} 1; Acme {base} 2"}
    return d


def make_index():
    eye = np.eye(len(NAMES), dtype=np.float32)

    def embed(q):
        # The leaf whose (first) word appears in the query wins; otherwise the query is
        # equally far from everything.
        for i, n in enumerate(NAMES):
            if n.split("_")[0].lower() in q.lower():
                return eye[i]
        return np.full(len(NAMES), 0.01, dtype=np.float32)
    return TaxonomyIndex(NAMES, docs_all(), eye, embed)


def call(name, **args):
    return {"type": "tool_use", "id": f"id-{name}-{len(args)}", "name": name, "input": args}


def record(leaf, conf="probable"):
    return call("record_category", leaf_id=leaf, confidence=conf)


class Scripted:
    """One scripted reply per model turn. Remembers what it was asked."""

    def __init__(self, *turns, usage=(100, 20)):
        self.turns, self.calls, self.usage = list(turns), [], usage

    def create(self, system, messages, tools, tool_choice):
        self.calls.append({"choice": tool_choice, "n_msgs": len(messages),
                           "last_user": messages[-1]["content"], "system": system})
        blocks = self.turns.pop(0) if self.turns else []
        return Reply(blocks, {"input": self.usage[0], "output": self.usage[1]})


def agent(*turns, **kw):
    client = Scripted(*turns)
    return Stage3Agent(make_index(), client, **kw), client


ITEM = {"title": "Acme Wireless Mouse", "brand": "Acme", "bullets": ""}
SHORT = np.array([3, 4, 5])          # a shortlist that does not contain MOUSE


# ---------------------------------------------------------------- tests

def test_index():
    print("\ntaxonomy tools")
    ix = make_index()
    text, ids = ix.search("a wireless mouse", 3)
    check("search returns the nearest category first", ids[0] == "MOUSE")
    check("search lines carry the id and the path", text.splitlines()[0].startswith("[MOUSE]")
          and "Category:" in text.splitlines()[0])
    check("search caps k at the limit", len(ix.search("mouse", 99)[1]) == SEARCH_K_MAX)
    check("search rejects an empty query", ix.search("   ")[0].startswith("error"))
    text, ids = ix.get_category("TOILET_PAPER_HOLDER")
    check("get_category shows examples", "Examples:" in text)
    check("get_category lists siblings under the same parent, not itself",
          "TOWEL_RING" in text and ids.count("TOILET_PAPER_HOLDER") == 1
          and "Siblings" in text and "MOUSE" not in text.split("Siblings")[1])
    check("get_category on an unknown id is an error",
          ix.get_category("NOPE")[0].startswith("error"))
    check("a leaf with no path has no siblings", "Siblings" not in ix.get_category("LOOSE_LEAF")[0])
    bad = TaxonomyIndex(NAMES, docs_all(), np.eye(len(NAMES), dtype=np.float32),
                        lambda q: np.full(len(NAMES), np.nan, dtype=np.float32))
    check("a non-finite similarity is an error, not a silent ranking",
          bad.search("mouse")[0].startswith("error"))


def test_loop():
    print("\nthe loop")
    a, c = agent([call("search_taxonomy", query="wireless mouse")], [record("MOUSE")])
    r = a.run(ITEM, SHORT)
    check("looks, then answers", r.leaf == "MOUSE" and r.stopped == "recorded" and not r.abstained)
    check("the steps are recorded in order",
          [s["tool"] for s in r.steps] == ["search_taxonomy", "record_category"])
    check("search results are fed back to the model", "[MOUSE]" in str(c.calls[1]["last_user"]))
    check("the history grows by one exchange per tool call",
          [k["n_msgs"] for k in c.calls] == [1, 3])
    check("token usage is summed across turns", r.usage == {"input": 200, "output": 40})
    check("the first prompt names the product and every candidate",
          "Acme Wireless Mouse" in c.calls[0]["last_user"]
          and all(NAMES[i] in c.calls[0]["last_user"] for i in SHORT))

    a, _ = agent([call("get_category", leaf_id="SHOES")], [record("SANDALS")])
    r = a.run(ITEM, SHORT)
    check("opening a category is a step of its own",
          [s["tool"] for s in r.steps] == ["get_category", "record_category"]
          and r.steps[0]["returned"][0] == "SHOES")

    a, _ = agent([call("search_taxonomy", query="mouse")], [record(NONE)])
    r = a.run(ITEM, SHORT)
    check("abstaining after a search is accepted",
          r.abstained and r.leaf is None and r.stopped == "abstained")

    a, _ = agent([])
    r = a.run(ITEM, SHORT)
    check("a reply with no tool call ends the loop with no label",
          r.leaf is None and not r.abstained and r.stopped == "no_tool")

    a, _ = agent([call("browse_shelves")], [record("MOUSE")])
    r = a.run(ITEM, SHORT)
    check("an unknown tool is an error the model can recover from",
          r.steps[0]["error"] and r.leaf == "MOUSE")


def test_stopping():
    print("\nstopping criterion")
    a, c = agent(*([[call("search_taxonomy", query="mouse")]] * 3), [record("MOUSE")],
                 max_steps=4)
    r = a.run(ITEM, SHORT)
    check("earlier turns let the model pick any tool",
          [k["choice"]["type"] for k in c.calls[:3]] == ["any"] * 3)
    check("the LAST turn is forced to record_category",
          c.calls[3]["choice"]["type"] == "tool" and c.calls[3]["choice"]["name"] == "record_category")
    check("parallel tool calls are disabled", all(
        k["choice"]["disable_parallel_tool_use"] for k in c.calls))
    check("a search-heavy run still ends with an answer at the budget",
          r.leaf == "MOUSE" and len(c.calls) == 4)
    check("the budget is stated to the model", "You have 4 turns" in c.calls[0]["system"])

    a, c = agent([call("search_taxonomy", query="mouse")], [record("INVENTED")], max_steps=2)
    r = a.run(ITEM, SHORT)
    check("an invalid FORCED answer ends the loop instead of spinning",
          r.leaf is None and r.stopped == "invalid" and not r.abstained and len(c.calls) == 2)

    a, _ = agent([call("get_category", leaf_id="SHOES")], [record(NONE)], max_steps=2)
    r = a.run(ITEM, SHORT)
    check("with the budget spent, abstaining without a search is allowed", r.abstained)
    try:
        agent(max_steps=1)
        check("a one-turn budget is refused", False)
    except ValueError:
        check("a one-turn budget is refused", True)
    check("the default budget matches the prompt", f"{MAX_STEPS} turns" in AGENT_SYSTEM)


def test_guards():
    print("\nguards the code enforces")
    a, c = agent([record(NONE)], [call("search_taxonomy", query="mouse")], [record("MOUSE")])
    r = a.run(ITEM, SHORT)
    check("abstaining before any search is rejected", r.steps[0]["error"] is True)
    check("the rejection reaches the model as a tool error",
          "not searched" in str(c.calls[1]["last_user"]))
    check("the model gets the turn back and can still answer", r.leaf == "MOUSE")

    a, _ = agent([record("INVENTED_ID")], [record("SHOES")])
    r = a.run(ITEM, SHORT)
    check("an invented id is rejected, not accepted", r.steps[0]["error"] and r.leaf == "SHOES")

    a, _ = agent([record(NONE)], [record(NONE)], require_search_before_abstain=False)
    check("the search-first rule can be switched off for an ablation",
          a.run(ITEM, SHORT).abstained)

    a, _ = agent([call("search_taxonomy", query="mouse")], [record("MOUSE")])
    r = a.run(ITEM, SHORT)
    check("a valid answer never carries an error step", not any(
        s["error"] for s in r.steps))


# ---------------------------------------------------------------- the graph

class StubAgent:
    def __init__(self, result=None, boom=False):
        self.calls, self._r, self._boom = 0, result, boom

    def run(self, item, shortlist_idx):
        self.calls += 1
        if self._boom:
            raise ConnectionError("api down")
        return self._r


def steps_result(leaf, stopped="recorded"):
    from serve.agent import AgentResult
    return AgentResult(leaf, "probable" if leaf else None, False, stopped,
                       [{"step": 1, "tool": "record_category", "error": False}],
                       {"input": 1, "output": 1})


def run_graph(s3_decision, stub_agent):
    c = TG.StubCascade(pred=0, pmax=0.5)
    s3 = TG.StubStage3(s3_decision)
    g = build_graph(c, s3, k=c.cfg.k, agent=stub_agent)
    return g.invoke({"item": TG.ITEM, "item_id": "X1", "trace": []})


def test_graph():
    print("\ngraph integration")
    abst = Decision(None, "certain", True, "stub")

    ag = StubAgent(steps_result("L5"))
    out = run_graph(abst, ag)
    check("an abstention reaches the agent", ag.calls == 1)
    check("a recovered item is its own tier", out["stage"] == "S3-agent" and out["leaf"] == "L5")
    check("the agent appears in the trace, after stage3",
          [t["node"] for t in out["trace"]] == ["s1", "retrieve", "rerank", "stage3", "agent"])
    check("the agent's steps are returned", out["agent_steps"][0]["tool"] == "record_category")

    out = run_graph(abst, StubAgent(steps_result(None, "abstained")))
    check("an agent abstention leaves the item abstained",
          out["stage"] == "S3-abstain" and out["leaf"] is None)

    ag = StubAgent(boom=True)
    out = run_graph(abst, ag)
    check("an error inside the agent fails closed, without raising",
          out["stage"] == "S3-abstain" and out["leaf"] is None)
    check("the failure is visible in the response, not swallowed",
          out["agent_steps"][0]["exception"] == "ConnectionError")

    ag = StubAgent(steps_result("L5"))
    out = run_graph(Decision("L3", "certain", False, "stub"), ag)
    check("an answered item never reaches the agent", ag.calls == 0 and out["stage"] == "S3")
    check("and carries no agent_steps", "agent_steps" not in out)

    ag = StubAgent(steps_result("L5"))
    out = run_graph(Decision(None, None, False, "stub"), ag)
    check("a Stage 3 PARSE FAILURE is not an abstention, so it skips the agent",
          ag.calls == 0 and out["stage"] == "S2")

    c = TG.StubCascade(pred=0, pmax=0.99)
    ag = StubAgent(steps_result("L5"))
    out = build_graph(c, TG.StubStage3(abst), agent=ag).invoke(
        {"item": TG.ITEM, "item_id": "X1", "trace": []})
    check("a confident S1 item never reaches the agent", ag.calls == 0 and out["stage"] == "S1")


def test_runtime():
    print("\nresponse contract")
    c = TG.StubCascade(pred=0, pmax=0.5)
    abst = Decision(None, "certain", True, "stub")
    s3 = TG.StubStage3(abst)
    rt = CascadeRuntime(c, s3, build_graph(c, s3, agent=StubAgent(steps_result("L5"))))
    out = rt.classify(TG.ITEM, item_id="X1")
    check("classify() exposes agent_steps when the agent ran", "agent_steps" in out)
    check("total_ms includes the agent's node",
          out["total_ms"] == round(sum(t["ms"] for t in out["trace"]), 2))

    s3 = TG.StubStage3(Decision("L3", "certain", False, "stub"))
    rt = CascadeRuntime(c, s3, build_graph(c, s3))
    out = rt.classify(TG.ITEM, item_id="X1")
    check("classify() is unchanged when there is no agent", "agent_steps" not in out)

    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        CascadeRuntime.load(".", agent=True)
        check("agent=True without an API key refuses to start", False)
    except SystemExit as e:
        check("agent=True without an API key refuses to start", "ANTHROPIC_API_KEY" in str(e))
    except Exception:                                    # noqa: BLE001
        check("agent=True without an API key refuses to start", False)


def main():
    test_index()
    test_loop()
    test_stopping()
    test_guards()
    test_graph()
    test_runtime()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
