"""The cascade as a LangGraph state graph.

    from serve.graph import build_graph, CascadeRuntime
    rt = CascadeRuntime.load("data", provider="cached")
    rt.graph.invoke(rt.initial({"title": "..."}, item_id="B01..."))

                 ┌──────────────────────────── p_max >= tau and argmax != OTHER
                 │                                                            │
    item ──▶ s1 ─┤                                                            ▼
                 └─▶ retrieve ──▶ rerank ──┬── provider == none ─────────────▶ END
                                           └─▶ stage3 ──────────────────────▶ END

WHAT IS AND IS NOT REIMPLEMENTED HERE.
Nothing that produces a number is reimplemented. Every node calls the same
pipeline.Cascade methods the offline evaluation calls -- _s1, _bi, _rerank, _shortlist --
and the retrieval node is a line-for-line copy of the block inside Cascade.predict.
This file contributes routing, timing and an abstention rule, and nothing else. That is
deliberate: a serving layer that reimplements the model path is a serving layer whose
numbers you have to re-earn.

THE ONE REAL DIFFERENCE, AND IT IS NOT FREE.
Offline runs batch 32 items; serving runs one. With attention masks a batch of one is
numerically equivalent in principle, but fp16 autocast on CUDA reduces in a different
order at a different shape, so agreement is checked rather than assumed:

    python src/serve/verify.py --data data --n 500

reports where the served path and the offline arrays disagree. Treat a nonzero count as
a finding to explain, not noise to round away.

ABSTENTION IS NOT A FALLBACK.
When Stage 3 abstains the item ends with leaf=None and stage='S3-abstain'. It is NOT
quietly given S2's top-1. pipeline.evaluate counts abstentions as misses because the
cascade produced no label, and this path agrees with it. An abstention that silently
became an S2 guess would inflate every accuracy number here relative to the offline
report while looking like an improvement.
"""
from __future__ import annotations

import json
import operator
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from serve.llm import Stage3, Stage3Config


class CascadeState(TypedDict, total=False):
    item: dict[str, Any]
    item_id: str | None
    text: str
    s1_leaf: str | None
    s1_conf: float | None
    deferred: bool
    cands: list[int]
    shortlist: list[str]
    shortlist_idx: list[int]
    leaf: str | None
    stage: str
    llm_confidence: str | None
    agent_steps: list[dict]
    trace: Annotated[list[dict], operator.add]


def _timed(name: str, fn):
    """Every node reports its own wall time into the trace. Per-tier latency is the
    reason this layer exists; measuring it at the edge only tells you the total."""
    def node(state: CascadeState) -> dict:
        t0 = time.perf_counter()
        out = fn(state)
        out["trace"] = [{"node": name, "ms": round((time.perf_counter() - t0) * 1000, 2)}]
        return out
    return node


def build_graph(cascade, stage3: Stage3, k: int | None = None, agent=None):
    """cascade: a pipeline.Cascade (or any object with the same four methods).
    Injected rather than constructed so the routing can be tested without loading 1 GB
    of weights -- see serve/test_graph.py.

    agent: an optional serve.agent.Stage3Agent. When given, an item Stage 3 ABSTAINED on
    gets a second look from a bounded loop that can search the taxonomy. It is the only
    consumer of an abstention; every other Stage 3 outcome ends the graph as before, and
    with agent=None the graph is exactly what it was."""
    from langgraph.graph import END, START, StateGraph
    from recall import INSTRUCTION, assert_finite, item_text

    k = k or cascade.cfg.k

    def s1(state):
        from types import SimpleNamespace
        it = state["item"]
        row = SimpleNamespace(title=it.get("title", ""), brand=it.get("brand", ""),
                              bullets=it.get("bullets", ""))
        text = item_text(row)
        pred, pmax = cascade._s1([text])
        deferred = bool(pred[0] == cascade.other_i or pmax[0] < cascade.cfg.tau)
        leaf = None if deferred else cascade.s1_classes[pred[0]]
        return {"text": text, "deferred": deferred, "s1_conf": float(pmax[0]),
                "s1_leaf": leaf, "leaf": leaf, "stage": "S2" if deferred else "S1"}

    def retrieve(state):
        # Copied from Cascade.predict. Documents were embedded bare; the instruction
        # goes on the query side only.
        emb = cascade._bi().encode(
            [state["text"]], prompt=f"Instruct: {INSTRUCTION}\nQuery: ",
            batch_size=1, normalize_embeddings=True).astype(np.float32)
        assert_finite(emb, "item embeddings")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = emb @ cascade.label_emb.T
        part = np.argpartition(-sims, k - 1, 1)[:, :k]
        cands = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, 1), 1), 1)
        return {"cands": cands[0].tolist()}

    def rerank(state):
        cands = np.array([state["cands"]])
        scores = cascade._rerank([state["text"]], cands)
        short = cascade._shortlist(cands, scores)[0]
        names = [cascade.leaves[c] for c in short]
        # S2's own answer, and the no-agent baseline Stage 3 has to beat.
        return {"shortlist_idx": short.tolist(), "shortlist": names,
                "leaf": names[0], "stage": "S2"}

    def stage3_node(state):
        d = stage3.decide(state["item"], np.array(state["shortlist_idx"]),
                          item_id=state.get("item_id"))
        if d.abstained:
            return {"leaf": None, "stage": "S3-abstain", "llm_confidence": d.confidence}
        if d.leaf is None:
            # No cached answer, a parse failure, or an invented id. Keep S2's answer and
            # say so, rather than reporting an S3 decision that never happened.
            return {"stage": "S2", "llm_confidence": None}
        return {"leaf": d.leaf, "stage": "S3", "llm_confidence": d.confidence}

    def agent_node(state):
        # Fail closed: an error anywhere in the loop (network, a tool bug) leaves the item
        # abstained. It must never take the request down, and it must never turn into a
        # guess that is reported as an agent decision.
        try:
            r = agent.run(state["item"], np.array(state["shortlist_idx"]))
        except Exception as e:                                   # noqa: BLE001
            return {"stage": "S3-abstain", "leaf": None,
                    "agent_steps": [{"error": True, "exception": type(e).__name__}]}
        if r.leaf is None:
            return {"stage": "S3-abstain", "leaf": None, "llm_confidence": r.confidence,
                    "agent_steps": r.steps}
        return {"leaf": r.leaf, "stage": "S3-agent", "llm_confidence": r.confidence,
                "agent_steps": r.steps}

    g = StateGraph(CascadeState)
    g.add_node("s1", _timed("s1", s1))
    g.add_node("retrieve", _timed("retrieve", retrieve))
    g.add_node("rerank", _timed("rerank", rerank))
    g.add_node("stage3", _timed("stage3", stage3_node))
    if agent is not None:
        g.add_node("agent", _timed("agent", agent_node))

    g.add_edge(START, "s1")
    g.add_conditional_edges("s1", lambda s: "retrieve" if s["deferred"] else END,
                            {"retrieve": "retrieve", END: END})
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges(
        "rerank", lambda s: END if stage3.cfg.provider == "none" else "stage3",
        {"stage3": "stage3", END: END})
    if agent is None:
        g.add_edge("stage3", END)
    else:
        g.add_conditional_edges(
            "stage3", lambda s: "agent" if s.get("stage") == "S3-abstain" else END,
            {"agent": "agent", END: END})
        g.add_edge("agent", END)
    return g.compile()


@dataclass
class CascadeRuntime:
    cascade: Any
    stage3: Stage3
    graph: Any
    agent: Any = None

    @classmethod
    def load(cls, data: str | Path = "data", device: str = "auto",
             provider: str = "cached", retriever_fp16: bool = False,
             agent: bool = False, **s3kw) -> "CascadeRuntime":
        """retriever_fp16 forces half precision for the bi-encoder on any backend.

        recall.load_model requests fp16 only on CUDA, so the cached embeddings in data/
        (which carry an fp16 fingerprint in every value) were produced in half precision
        while a CPU or MPS serving run produces fp32 query vectors. That mismatch is a
        query-side precision difference against fp16 label vectors -- a configuration
        neither offline run used. Setting this makes the serving path round the same way
        the cache did. It is not bit-identical to CUDA fp16, but it removes the precision
        gap rather than leaving it uncontrolled."""
        # First, before any weights load: the second pass makes LIVE model calls even when
        # Stage 3 replays cached answers (the cache has no second-pass entries), so a
        # missing key should cost nothing rather than a 3GB load followed by a failure.
        if agent and not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("agent=True needs ANTHROPIC_API_KEY: the second pass makes "
                             "live model calls.")
        from pipeline import Cascade
        cascade = Cascade.load(data, device=device)
        if retriever_fp16:
            import torch
            from sentence_transformers import SentenceTransformer
            from recall import MODEL as BI_MODEL
            cascade.retriever = SentenceTransformer(
                BI_MODEL, device=cascade.device,
                model_kwargs={"torch_dtype": torch.float16})
        s3 = Stage3(Path(data), cascade.leaves, Stage3Config(provider=provider, **s3kw))
        ag = None
        if agent:
            from serve.agent import AnthropicMessages, Stage3Agent, TaxonomyIndex, embed_from_cascade
            docs_all = json.loads((Path(data) / "label_docs.json").read_text())
            index = TaxonomyIndex(cascade.leaves, docs_all, cascade.label_emb,
                                  embed_from_cascade(cascade))
            ag = Stage3Agent(index, AnthropicMessages(model=s3.cfg.model,
                                                      headers=s3._headers()))
        return cls(cascade, s3, build_graph(cascade, s3, agent=ag), ag)

    @staticmethod
    def initial(item: dict, item_id: str | None = None) -> CascadeState:
        return {"item": item, "item_id": item_id, "trace": []}

    def classify(self, item: dict, item_id: str | None = None) -> dict:
        s = self.graph.invoke(self.initial(item, item_id))
        out = {
            "leaf": s.get("leaf"), "stage": s.get("stage"), "text": s.get("text"),
            "s1_confidence": s.get("s1_conf"), "llm_confidence": s.get("llm_confidence"),
            "shortlist": s.get("shortlist"), "trace": s.get("trace", []),
            "total_ms": round(sum(t["ms"] for t in s.get("trace", [])), 2),
        }
        # Present only when the second pass ran, so the response contract that both front
        # ends share is unchanged for every item that never reaches it.
        if s.get("agent_steps") is not None:
            out["agent_steps"] = s["agent_steps"]
        return out
