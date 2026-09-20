"""Unit tests for the routing, which is the only new logic in the serving layer.

The model path is not tested here -- it is the same four Cascade methods the offline
evaluation already exercises. What IS new, and what these tests cover, is the set of
decisions the graph makes about where an item goes and what happens to it when a tier
declines to answer:

  1. a confident item stops at S1 and never touches the retriever
  2. __OTHER__ defers even at p_max = 1.0 (the gate is OR, not AND)
  3. an abstention ends with NO label -- it is not quietly given S2's top-1
  4. a parse failure or an invented leaf id falls back to S2 and is reported as S2,
     not as an S3 decision that never happened
  5. provider 'none' short-circuits after rerank
  6. every visited node appears in the trace, in order

The cascade and the retriever are stubbed, so this needs no weights and no torch:

    python src/serve/test_graph.py
"""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# recall.py imports torch at module scope. The graph only needs three names from it, and
# stubbing them keeps these tests runnable on a machine with no GPU stack installed.
_recall = ModuleType("recall")
_recall.INSTRUCTION = "stub instruction"
_recall.assert_finite = lambda a, what: None
_recall.item_text = lambda row: " ".join(
    [row.title] + ([f"Brand: {row.brand}"] if row.brand else [])
    + ([row.bullets] if row.bullets else []))[:1000]
sys.modules.setdefault("recall", _recall)

from serve.graph import build_graph                          # noqa: E402
from serve.llm import Decision                               # noqa: E402
from shortlist import build_shortlist                        # noqa: E402

FAILURES = []
LEAVES = [f"L{i}" for i in range(20)]


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


class StubCascade:
    """Same surface as pipeline.Cascade, with the four model calls replaced by fixtures.
    _shortlist uses the REAL build_shortlist so the ordering under test is the real one."""

    def __init__(self, pred, pmax, tau=0.9, k=6, m=4):
        self.cfg = SimpleNamespace(tau=tau, k=k, shortlist=m, shortlist_mode="fused",
                                   batch_size=32)
        self.s1_classes = ["A", "B", "__OTHER__"]
        self.other_i = 2
        self.leaves = LEAVES
        self.label_emb = np.eye(len(LEAVES), 8, dtype=np.float32)
        self._pred, self._pmax = pred, pmax
        self.bi_calls = 0

    def _s1(self, texts):
        return np.array([self._pred]), np.array([self._pmax], dtype=np.float32)

    def _bi(self):
        self.bi_calls += 1
        enc = lambda texts, **kw: np.tile(  # noqa: E731
            np.arange(8, dtype=np.float32), (len(texts), 1))
        return SimpleNamespace(encode=enc)

    def _rerank(self, texts, cands):
        # Descending so the cross-encoder ordering differs from the retriever's.
        return np.tile(np.linspace(1.0, 0.0, cands.shape[1], dtype=np.float32),
                       (len(texts), 1))

    def _shortlist(self, cands, scores):
        return build_shortlist(cands, scores, self.cfg.shortlist, self.cfg.shortlist_mode)


class StubStage3:
    def __init__(self, decision, provider="anthropic"):
        self.cfg = SimpleNamespace(provider=provider)
        self._d = decision
        self.calls = 0

    def decide(self, item, shortlist_idx, item_id=None):
        self.calls += 1
        return self._d


ITEM = {"title": "Widget", "brand": "Acme", "bullets": "blue"}


def run(cascade, stage3):
    g = build_graph(cascade, stage3, k=cascade.cfg.k)
    return g.invoke({"item": ITEM, "item_id": "X1", "trace": []})


def test_routing():
    print("\nrouting")
    c = StubCascade(pred=0, pmax=0.99)
    s3 = StubStage3(Decision("L3", "certain", False, "stub"))
    out = run(c, s3)
    check("confident item is answered by S1", out["leaf"] == "A" and out["stage"] == "S1")
    check("confident item never loads the retriever", c.bi_calls == 0)
    check("confident item never calls Stage 3", s3.calls == 0)
    check("trace holds only s1", [t["node"] for t in out["trace"]] == ["s1"])

    c = StubCascade(pred=0, pmax=0.5)
    out = run(c, StubStage3(Decision("L3", "certain", False, "stub")))
    check("low confidence escalates to S3", out["stage"] == "S3" and out["leaf"] == "L3")
    check("trace records every visited node in order",
          [t["node"] for t in out["trace"]] == ["s1", "retrieve", "rerank", "stage3"])
    check("every node reports a duration", all(t["ms"] >= 0 for t in out["trace"]))

    # The gate is (argmax == OTHER) OR (p_max < tau). A confident OTHER must still defer:
    # __OTHER__ is not a label, it is S1 saying the answer is outside its 69 classes.
    c = StubCascade(pred=2, pmax=1.0)
    out = run(c, StubStage3(Decision("L7", "probable", False, "stub")))
    check("__OTHER__ defers even at p_max = 1.0", out["stage"] == "S3")


def test_abstention_and_failures():
    print("\nabstention and failure handling")
    c = StubCascade(pred=0, pmax=0.5)
    out = run(c, StubStage3(Decision(None, "certain", True, "stub")))
    check("abstention yields no label", out["leaf"] is None)
    check("abstention is its own tier", out["stage"] == "S3-abstain")
    check("abstention is NOT downgraded to S2's top-1",
          out["leaf"] != out["shortlist"][0])

    c = StubCascade(pred=0, pmax=0.5)
    out = run(c, StubStage3(Decision(None, None, False, "stub")))
    check("a parse failure keeps S2's answer", out["leaf"] == out["shortlist"][0])
    check("a parse failure is reported as S2, not S3", out["stage"] == "S2")


def test_provider_none():
    print("\nprovider 'none'")
    c = StubCascade(pred=0, pmax=0.5)
    s3 = StubStage3(Decision("L3", "certain", False, "stub"), provider="none")
    out = run(c, s3)
    check("provider 'none' stops after rerank", out["stage"] == "S2")
    check("provider 'none' never calls the model", s3.calls == 0)
    check("provider 'none' still returns S2's top-1",
          out["leaf"] == out["shortlist"][0] and out["leaf"] in LEAVES)
    check("trace stops at rerank",
          [t["node"] for t in out["trace"]] == ["s1", "retrieve", "rerank"])


def test_shortlist_shape():
    print("\nshortlist")
    c = StubCascade(pred=0, pmax=0.5)
    out = run(c, StubStage3(Decision("L3", "certain", False, "stub")))
    check("shortlist has the configured length", len(out["shortlist"]) == c.cfg.shortlist)
    check("shortlist entries are leaf names, not indices",
          all(s in LEAVES for s in out["shortlist"]))
    check("shortlist has no duplicates", len(set(out["shortlist"])) == len(out["shortlist"]))


def test_cached_provider_without_item_id():
    """The cached provider is the default the server starts with, so the first thing
    anyone tries -- POST /classify with just a title -- goes through this path."""
    print("\ncached provider")
    from serve.llm import Stage3, Stage3Config
    s3 = Stage3.__new__(Stage3)          # bypass __init__: no label_docs.json needed
    s3.cfg = Stage3Config(provider="cached")
    s3._replay = {"KNOWN": "L4", "ABSTAINED": "NONE_OF_THESE"}
    s3.leaves = LEAVES

    check("a known item_id replays its cached label",
          s3.decide({}, [], item_id="KNOWN").leaf == "L4")
    check("a cached abstention stays an abstention",
          s3.decide({}, [], item_id="ABSTAINED").abstained is True)
    d = s3.decide({}, [], item_id=None)
    check("no item_id does not raise", d.leaf is None and d.abstained is False)
    d = s3.decide({}, [], item_id="NOT_IN_FILE")
    check("an unknown item_id does not raise", d.leaf is None and d.abstained is False)

    # End to end: the graph must report this as S2, not as an S3 decision.
    c = StubCascade(pred=0, pmax=0.5)
    out = run(c, s3)
    check("an uncached deferred item falls back to S2", out["stage"] == "S2")
    check("and keeps S2's top-1", out["leaf"] == out["shortlist"][0])


def test_live_provider_grounding():
    """The prompt says "exactly one category id from the list", but the old check only
    tested the whole taxonomy, so a real id that was NOT offered would have passed. Zero
    such answers were measured in the recorded run; that is a measurement, and this makes
    it a rule for every live provider."""
    print("\nlive provider grounding")
    from serve.llm import Stage3, Stage3Config
    s3 = Stage3.__new__(Stage3)
    s3.cfg, s3.leaves, s3.last_usage = Stage3Config(provider="anthropic"), LEAVES, None
    s3._render = lambda item, shortlist: ("item", "candidates")
    said = {}
    s3._call_anthropic = lambda item_text, cand_text: said["v"]
    shown = np.array([1, 2, 3])                       # L1, L2, L3

    said["v"] = {"leaf_id": "L2", "confidence": "certain"}
    d = s3.decide({}, shown)
    check("an answer from the shown shortlist is accepted", d.leaf == "L2" and d.reason is None)

    said["v"] = {"leaf_id": "L9", "confidence": "certain"}
    d = s3.decide({}, shown)
    check("a real id that was not shown is rejected as ungrounded",
          d.leaf is None and d.abstained is False and d.reason == "ungrounded")

    said["v"] = {"leaf_id": "NOT_A_LEAF", "confidence": "certain"}
    check("an id outside the taxonomy is invalid_id, a different failure",
          s3.decide({}, shown).reason == "invalid_id")

    said["v"] = {"leaf_id": "NONE_OF_THESE", "confidence": "certain"}
    d = s3.decide({}, shown)
    check("an abstention needs no grounding", d.abstained is True and d.reason is None)

    said["v"] = None
    check("no parseable answer is unparseable", s3.decide({}, shown).reason == "unparseable")

    cached = Stage3.__new__(Stage3)
    cached.cfg, cached.leaves = Stage3Config(provider="cached"), LEAVES
    cached._replay = {"K": "L9"}
    check("the cached provider is exempt: its answers were given against another shortlist",
          cached.decide({}, shown, item_id="K").leaf == "L9")

    said["v"] = {"leaf_id": "L19", "confidence": "certain"}      # real, and not on the page
    out = run(StubCascade(pred=0, pmax=0.5), s3)
    check("end to end, an ungrounded answer is reported as S2, not as a Stage 3 decision",
          out["stage"] == "S2")
    check("and the item keeps S2's top-1", out["leaf"] == out["shortlist"][0])


def main():
    test_routing()
    test_abstention_and_failures()
    test_provider_none()
    test_shortlist_shape()
    test_cached_provider_without_item_id()
    test_live_provider_grounding()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
