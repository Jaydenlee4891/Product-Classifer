"""Stage 3b: the second pass. A bounded agent loop that can search the taxonomy.

WHY IT EXISTS. Every earlier tier is trapped inside the retriever's top-K. If the correct
leaf was never retrieved -- 9.4% of escalated items at K=50, shortlist 10 -- no reranker,
threshold or prompt can recover it, and Stage 3's single call can only pick from the list
it is handed. This is the one tier that can go and look: it may search all 530 categories
and open one to see its siblings.

WHAT IS AN AGENT HERE, PRECISELY. Stage 3 (stage3_agent.py) is one forced tool call and is
NOT an agent. This is:

  tools       search_taxonomy, get_category (read-only), record_category (terminal)
  loop        model turn -> tool result -> model turn, until it records an answer
  state       the message history carries what it has already looked at
  stopping    it records a category, OR it abstains, OR the step budget runs out. On the
              last permitted turn the tool choice is forced to record_category, so the loop
              cannot end by running off the edge.
  recovery    an invalid id or an abstention with no search is REJECTED as a tool error and
              the model gets the turn back -- the code enforces what the prompt asks for.

It runs only on items Stage 3 abstained on (7.6% of escalated items, 153 of 2,015), so a
multi-turn loop is affordable there and nowhere else.

WHAT IT CANNOT FIX. It is only invoked on abstentions. Of the 190 escalated items whose
gold leaf was not in the shortlist, Stage 3 abstained on 48; the other 142 got a confident
wrong label and never reach this tier. Widening the trigger is a separate, measurable
change, not something to slip in here.

FAILURE POLICY. Any failure leaves the item abstained -- never a guess, and never S2's
top-1 reported as an agent decision. The stage is 'S3-agent' only when the agent produced
a valid label.

STATUS. Built and unit-tested against a scripted model (serve/test_agent.py). NOT yet
evaluated live: no result here says how often it recovers an item. That number comes from
`python src/agent_pass2.py`, which spends API credit.
"""
from __future__ import annotations

import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import stage3_agent as S3

MAX_STEPS = 5           # model turns, including the final forced record_category
SEARCH_K_DEFAULT = 5
SEARCH_K_MAX = 8
DOC_CHARS = 420         # matches Stage3Config.doc_chars, so candidates and hits look alike
ENTRY_CHARS = 700       # a full get_category entry, examples included
RECORD = S3.RECORD_TOOL["name"]

SEARCH_TOOL = {
    "name": "search_taxonomy",
    "description": "Semantic search over ALL catalogue categories, not just the candidates. "
                   "Describe the kind of product it is (e.g. 'wireless computer mouse'), "
                   "not the listing's marketing text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What kind of product to look for."},
            "k": {"type": "integer",
                  "description": f"How many categories to return, 1 to {SEARCH_K_MAX}."},
        },
        "required": ["query"],
    },
}

GET_TOOL = {
    "name": "get_category",
    "description": "Open one category: its path, example products already filed under it, "
                   "and its sibling categories. Use it to compare close matches.",
    "input_schema": {
        "type": "object",
        "properties": {"leaf_id": {"type": "string",
                                   "description": "A category id from the candidates or a search."}},
        "required": ["leaf_id"],
    },
}

TOOLS = [SEARCH_TOOL, GET_TOOL, S3.RECORD_TOOL]

AGENT_SYSTEM = f"""You assign e-commerce products to exactly one catalogue category.

You will see a product listing and some candidate categories from an earlier retrieval
step. That step sometimes misses the right category, so you can look beyond it:

- search_taxonomy(query): semantic search over ALL categories.
- get_category(leaf_id): one category in full, with example products and sibling categories.

Work like this: decide what kind of thing the product IS; check the candidates; if none
clearly fits, search for the product type; if a result looks close, open it and compare it
with its siblings before choosing. The candidates are in no meaningful order.

Rules:
- Choose the category describing what the product IS, not what it mentions. A phone case
  with a card slot is a phone case, not a wallet.
- Only answer with an id you have actually seen, in the candidates or in a tool result.
- Answer '{S3.NONE}' only after you have searched at least once and still found nothing
  that fits. An unsupported guess is worse than a flagged abstention.
- Use 'unsure' freely. A flagged uncertainty is cheap; a confident error is not.
- You have {MAX_STEPS} turns. The last one must be record_category.

Always finish by calling record_category."""


# ------------------------------------------------------------------ the taxonomy

def _parent_path(v2_path: str) -> tuple:
    """('A', 'B') for 'x. Category: A > B > C.' -- () when the leaf has no path (39 of the
    530 do not), so it has no siblings rather than sharing a giant empty-parent group."""
    m = re.search(r"Category:\s*(.*?)\.?\s*$", v2_path)
    parts = [p.strip() for p in m.group(1).split(">")] if m else []
    return tuple(parts[:-1]) if len(parts) >= 2 else ()


class TaxonomyIndex:
    """Read-only access to the 530 categories, shared by both tools.

    `embed` maps a string to a unit vector in the SAME space as `label_emb`; it is
    injected so the tools can be tested without loading Qwen3.
    """

    def __init__(self, leaves: list[str], docs_all: dict, label_emb: np.ndarray,
                 embed: Callable[[str], np.ndarray]):
        self.leaves = list(leaves)
        self._pos = {l: i for i, l in enumerate(self.leaves)}
        self._docs, self._emb, self._embed = docs_all, label_emb, embed
        # Same rendering Stage 3 shows, so a candidate and a search hit look alike.
        self.docs = [S3.compose_doc(docs_all[l], "v2_path", DOC_CHARS) for l in self.leaves]
        self._children: dict[tuple, list[int]] = {}
        self._parent: list[tuple] = []
        for i, l in enumerate(self.leaves):
            p = _parent_path(docs_all[l]["v2_path"])
            self._parent.append(p)
            if p:
                self._children.setdefault(p, []).append(i)

    def search(self, query: Any, k: Any = SEARCH_K_DEFAULT) -> tuple[str, list[str]]:
        q = query.strip() if isinstance(query, str) else ""
        if not q:
            return "error: query must be a non-empty string.", []
        try:
            k = max(1, min(int(k), SEARCH_K_MAX))
        except (TypeError, ValueError):
            k = SEARCH_K_DEFAULT
        # Same suppression as graph.retrieve: the Accelerate matmul on Apple silicon raises
        # spurious divide/overflow/invalid warnings on finite input. What is NOT spurious
        # is a non-finite result, so that is checked instead of being warned about.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self._emb @ np.asarray(self._embed(q), dtype=np.float32)
        if not np.isfinite(sims).all():
            return "error: search failed (non-finite similarity).", []
        top = np.argsort(-sims)[:k]
        return ("\n".join(f"[{self.leaves[i]}] {self.docs[i]}" for i in top),
                [self.leaves[i] for i in top])

    def get_category(self, leaf_id: Any) -> tuple[str, list[str]]:
        i = self._pos.get(leaf_id) if isinstance(leaf_id, str) else None
        if i is None:
            return (f"error: {leaf_id!r} is not a category id. "
                    "Use search_taxonomy to find valid ids.", [])
        text = f"[{leaf_id}] {S3.compose_doc(self._docs[leaf_id], 'v3_proto', ENTRY_CHARS)}"
        sib = [self.leaves[j] for j in self._children.get(self._parent[i], []) if j != i][:8]
        if sib:
            text += "\nSiblings (same parent): " + ", ".join(sib)
        return text, [leaf_id, *sib]

    def __contains__(self, leaf_id) -> bool:
        return leaf_id in self._pos


def embed_from_cascade(cascade) -> Callable[[str], np.ndarray]:
    """The retriever the graph already uses, wrapped as a one-string embed function.

    Same instruction prefix as graph.retrieve. That prefix was tuned for product listings,
    not for the short phrases an agent writes, so search quality on terse queries is
    something agent_pass2.py measures rather than assumes."""
    from recall import INSTRUCTION
    prompt = f"Instruct: {INSTRUCTION}\nQuery: "

    def embed(q: str) -> np.ndarray:
        return cascade._bi().encode([q], prompt=prompt, batch_size=1,
                                    normalize_embeddings=True)[0].astype(np.float32)
    return embed


# ------------------------------------------------------------------ the model

@dataclass
class Reply:
    blocks: list[dict]              # {"type":"tool_use","id","name","input"} | {"type":"text","text"}
    usage: dict = field(default_factory=dict)


class LLM(Protocol):
    def create(self, system: str, messages: list, tools: list, tool_choice: dict) -> Reply: ...


class AnthropicMessages:
    """The raw SDK behind the loop's one-method interface."""

    def __init__(self, model: str = S3.MODEL, max_tokens: int = 400,
                 headers: dict | None = None, client=None):
        self.model, self.max_tokens = model, max_tokens
        if client is None:
            from anthropic import Anthropic
            client = Anthropic(default_headers=headers or None)
        self._client = client

    def create(self, system, messages, tools, tool_choice) -> Reply:
        r = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0, system=system,
            messages=messages, tools=tools, tool_choice=tool_choice)
        # Rebuilt as minimal dicts: SDK block objects carry extra fields, and the history
        # is sent back to the API on every turn.
        blocks = []
        for b in r.content:
            if b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            elif b.type == "text":
                blocks.append({"type": "text", "text": b.text})
        u = getattr(r, "usage", None)
        return Reply(blocks, {"input": getattr(u, "input_tokens", 0) or 0,
                              "output": getattr(u, "output_tokens", 0) or 0})


# ------------------------------------------------------------------ the loop

@dataclass
class AgentResult:
    leaf: str | None
    confidence: str | None
    abstained: bool
    stopped: str            # recorded | abstained | invalid | no_tool
    steps: list[dict]
    usage: dict


class Stage3Agent:
    def __init__(self, index: TaxonomyIndex, client: LLM, max_steps: int = MAX_STEPS,
                 seed: int = S3.SEED, require_search_before_abstain: bool = True):
        if max_steps < 2:
            raise ValueError("max_steps must be at least 2: one turn to look, one to answer")
        self.index, self.client, self.max_steps = index, client, max_steps
        self.require_search = require_search_before_abstain
        self._rng = random.Random(seed)
        self.system = AGENT_SYSTEM.replace(f"You have {MAX_STEPS} turns",
                                           f"You have {max_steps} turns")

    def _reject(self, leaf, searched: bool, last: bool) -> str | None:
        """Why a record_category call cannot be accepted, or None. These are returned to
        the model as tool errors, so it can fix them within its remaining turns."""
        if not isinstance(leaf, str):
            return "error: leaf_id must be a string."
        if leaf == S3.NONE:
            if self.require_search and not searched and not last:
                return ("error: you have not searched the taxonomy yet. Call "
                        "search_taxonomy at least once before abstaining.")
            return None
        if leaf not in self.index:
            return (f"error: {leaf!r} is not a category id. Use search_taxonomy or "
                    f"get_category to find a valid id, or answer {S3.NONE}.")
        return None

    def run(self, item: dict, shortlist_idx) -> AgentResult:
        row = SimpleNamespace(title=item.get("title", ""), brand=item.get("brand", ""),
                              bullets=item.get("bullets", ""))
        cands = S3.render_candidates(shortlist_idx, self.index.docs, self.index.leaves,
                                     self._rng, shuffle=True)
        messages: list = [{"role": "user", "content": f"{S3.render_item(row)}\n\n{cands}"}]
        steps: list[dict] = []
        usage = {"input": 0, "output": 0}
        searched = False

        for n in range(1, self.max_steps + 1):
            last = n == self.max_steps
            choice = ({"type": "tool", "name": RECORD, "disable_parallel_tool_use": True}
                      if last else {"type": "any", "disable_parallel_tool_use": True})
            reply = self.client.create(self.system, messages, TOOLS, choice)
            for k in usage:
                usage[k] += int(reply.usage.get(k) or 0)

            uses = [b for b in reply.blocks if b.get("type") == "tool_use"]
            if not uses:
                steps.append({"step": n, "tool": None, "error": True})
                return AgentResult(None, None, False, "no_tool", steps, usage)

            results = []
            for tu in uses:
                name, args, t0 = tu.get("name"), tu.get("input") or {}, time.perf_counter()
                ids: list[str] = []
                if name == RECORD:
                    err = self._reject(args.get("leaf_id"), searched, last)
                    if err is None:
                        steps.append({"step": n, "tool": name, "input": args, "error": False,
                                      "ms": round((time.perf_counter() - t0) * 1000, 2)})
                        leaf = args["leaf_id"]
                        abstained = leaf == S3.NONE
                        return AgentResult(None if abstained else leaf, args.get("confidence"),
                                           abstained, "abstained" if abstained else "recorded",
                                           steps, usage)
                    text = err
                elif name == "search_taxonomy":
                    text, ids = self.index.search(args.get("query"), args.get("k"))
                    searched = searched or not text.startswith("error")
                elif name == "get_category":
                    text, ids = self.index.get_category(args.get("leaf_id"))
                else:
                    text = f"error: unknown tool {name!r}."
                is_err = text.startswith("error")
                steps.append({"step": n, "tool": name, "input": args, "returned": ids,
                              "error": is_err,
                              "ms": round((time.perf_counter() - t0) * 1000, 2)})
                res = {"type": "tool_result", "tool_use_id": tu["id"], "content": text}
                if is_err:
                    res["is_error"] = True
                results.append(res)

            if last:        # a forced record_category that was rejected: nothing left to try
                return AgentResult(None, None, False, "invalid", steps, usage)
            messages.append({"role": "assistant", "content": reply.blocks})
            messages.append({"role": "user", "content": results})

        return AgentResult(None, None, False, "no_tool", steps, usage)   # unreachable
