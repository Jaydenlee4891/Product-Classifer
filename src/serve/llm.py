"""Stage 3 as a swappable node.

Five providers behind one call signature. The point is that the tier the cascade
escalates to is a configuration choice, not a rewrite:

  anthropic   LangChain chat model, forced tool call. The reference arm.
  raw         the same prompt through the Anthropic SDK directly, bypassing LangChain.
              Exists as a NEGATIVE CONTROL, not as a fallback -- see parity below.
  cached      replays data/stage3_<mode>.csv by item_id. No network, no spend.
              This is what the evaluation and the demo use. An item with no item_id, or
              one absent from the file, simply has no Stage 3 answer and keeps S2's --
              reported as stage 'S2', never as an S3 decision that did not happen.
  ollama      a local model via LangChain. Same prompt, same forced tool.
  none        skip the tier; the cascade answers with S2's top-1.

KEYS. An organisation-level API key is not scoped to a workspace and the API rejects it
with a 400 unless the request names one. Set ANTHROPIC_WORKSPACE_ID and it is sent as the
anthropic-workspace-id header on both the LangChain and raw paths; a workspace-scoped key
needs nothing. stage3_agent.py's batch path has no such option, so a key that works there
works here and not the reverse.

PARITY, AND WHY 'raw' EXISTS.
Everything that decides what the model sees -- the system prompt, the worked examples,
the candidate rendering, the shuffle, the tool schema -- is imported from stage3_agent
rather than restated here. A second copy of a prompt is a second thing to drift.

But LangChain and the raw SDK encode the same conversation differently: the batch script
puts tool results in a user turn, LangChain wraps them in ToolMessage. The rendered TEXT
is identical; the message structure is not. That is a difference the model can in
principle see, so it is checked rather than assumed:

    python src/serve/llm.py --parity --n 40

sends the same 40 items both ways and reports how often the two disagree. If that number
is not ~0, the serving layer changed the model's input and the offline numbers no longer
describe it.

COST, AND WHERE THE CACHE MARKER HAS TO GO. Measured, not estimated -- via the free
count_tokens endpoint (`--count-tokens`), on claude-sonnet-4-5:

    tools + system                        882 tokens   under the 1,024 minimum: INERT
    tools + system + worked examples    1,282 tokens   caches
    full request (mean of 20)           1,682 tokens   p95 1,763

The marker used to sit on the system block, which is 882 tokens, which is below the
minimum, which means the API silently did not cache -- no error, and both
cache_creation_input_tokens and cache_read_input_tokens return 0. An optimisation that
can be believed indefinitely without ever having run. It now sits on the last worked
example instead, which clears the threshold and covers 76% of a mean request.

Estimating this from characters is how it went wrong the first time: at an English prose
ratio the full request looks like ~930 tokens. It is 1,682. The candidate block is
ALL_CAPS_UNDERSCORE leaf ids and taxonomy paths, which tokenise about twice as badly as
prose, and every cost figure derived from the character count was low by ~1.8x.

Caching is verified rather than assumed: decide() records the cache counts from every
response on `last_usage` and --parity prints them. Expect cache_write on the first call
and cache_read after. The entry is 5-minute ephemeral, so a sequential run stays warm;
the Batch API makes no ordering guarantee, so hits there are not guaranteed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import stage3_agent as S3
from shortlist import build_shortlist

ABSTAIN = S3.NONE
PROVIDERS = ("anthropic", "raw", "cached", "ollama", "none")


@dataclass
class Decision:
    """What Stage 3 returned. `leaf is None` means abstention, which is a refusal to
    label -- never silently downgraded to S2's answer."""
    leaf: str | None
    confidence: str | None          # certain | probable | unsure
    abstained: bool
    provider: str
    raw: dict | None = field(default=None, repr=False)
    usage: dict | None = field(default=None, repr=False)


@dataclass
class Stage3Config:
    provider: str = "cached"
    model: str = S3.MODEL
    doc_variant: str = "v2_path"
    doc_chars: int = 420
    shuffle: bool = True            # False reproduces the --ablate rank-order arm
    fewshot: bool = True            # False reproduces the --ablate no-fewshot arm
    max_tokens: int = 200
    # Where the cache_control marker goes. Measured with --count-tokens on Sonnet 4.5:
    #   "system"  tools + system                     882 tokens -> under the 1,024
    #             minimum, so the API silently does not cache. This was the default and
    #             it never did anything.
    #   "fewshot" tools + system + worked examples  1282 tokens -> caches, covering 76%
    #             of a 1,682-token mean request.
    #   "none"    no marker.
    # Verify with --parity: cache_write > 0 on the first call, cache_read > 0 after.
    cache_at: str = "fewshot"
    seed: int = S3.SEED
    # An organisation-level key is not tied to a workspace, so every request must name
    # one. A workspace-scoped key carries it implicitly and needs nothing here.
    workspace_id: str | None = None
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    cached_csv: str | None = None


class Stage3:
    # Class-level default so the attribute exists even when __init__ is bypassed --
    # decide() is reachable on a bare instance (serve/test_graph.py builds one that way
    # to avoid needing label_docs.json) and must not depend on construction order.
    last_usage: dict | None = None

    def __init__(self, data: Path, leaves: list[str], cfg: Stage3Config):
        self.data, self.leaves, self.cfg = Path(data), leaves, cfg
        if cfg.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {cfg.provider!r}")
        docs_all = json.loads((self.data / "label_docs.json").read_text())
        self.docs = [S3.compose_doc(docs_all[l], cfg.doc_variant, cfg.doc_chars)
                     for l in leaves]
        self.system = S3.system_prompt(cfg.doc_variant)
        self._rng = random.Random(cfg.seed)
        self._client = None
        self._replay: dict[str, str] | None = None
        self.last_usage: dict | None = None
        if cfg.workspace_id is None:
            cfg.workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID") or None

    def _headers(self) -> dict:
        """Workspace scoping, plus a response-encoding override.

        WORKSPACE. An organisation-level key is not tied to a workspace, so every request
        must name one. A workspace-scoped key carries it implicitly.

        ENCODING. Some installed combinations of httpx2 and a zstd binding raise
        `TypeError: process() takes no keyword arguments` while DECOMPRESSING a response,
        which the SDK re-raises as APIConnectionError -- a message that sends you looking
        at the network when the request already succeeded and was already billed. The
        tell is that it only happens on success: 4xx bodies are small and come back
        uncompressed, so they decode fine and the fault looks intermittent.

        Asking for gzip sidesteps the zstd path entirely. It costs a little bandwidth and
        changes nothing else. Set ANTHROPIC_ALLOW_ZSTD=1 to stop overriding once the
        underlying package conflict is fixed.
        """
        h = {}
        if self.cfg.workspace_id:
            h["anthropic-workspace-id"] = self.cfg.workspace_id
        if not os.environ.get("ANTHROPIC_ALLOW_ZSTD"):
            h["accept-encoding"] = "gzip"
        return h

    # ---------- prompt ----------

    def _render(self, item: dict, shortlist_idx) -> tuple[str, str]:
        row = SimpleNamespace(title=item.get("title", ""), brand=item.get("brand", ""),
                              bullets=item.get("bullets", ""))
        item_text = S3.render_item(row)
        cand_text = S3.render_candidates(shortlist_idx, self.docs, self.leaves,
                                         self._rng, shuffle=self.cfg.shuffle)
        return item_text, cand_text

    # ---------- providers ----------

    def _anthropic_chain(self):
        if self._client is None:
            from langchain_anthropic import ChatAnthropic
            self._client = ChatAnthropic(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens, temperature=0,
                default_headers=self._headers() or None,
            ).bind_tools([S3.RECORD_TOOL],
                         tool_choice={"type": "tool", "name": S3.RECORD_TOOL["name"]})
        return self._client

    def _lc_messages(self, item_text: str, cand_text: str):
        """LangChain mirror of stage3_agent.build_messages.

        The system block is a list with cache_control so the fixed prefix is billed once
        per cache window rather than once per item.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        anthropic = self.cfg.provider == "anthropic"
        mark = {"type": "ephemeral"}
        # The marker caches everything BEFORE and including it, in the order
        # tools -> system -> messages. Putting it on the system block therefore caches
        # only tools + system; putting it on the last worked example also catches those.
        # Nothing after it is cached, which is correct: the item is different every time.
        at_system = anthropic and self.cfg.cache_at == "system"
        at_fewshot = anthropic and self.cfg.cache_at == "fewshot" and self.cfg.fewshot

        sys_content = [{"type": "text", "text": self.system}]
        if at_system:
            sys_content[0]["cache_control"] = mark
        msgs = [SystemMessage(content=sys_content)]

        if self.cfg.fewshot:
            shots = ((S3.FEWSHOT_USER, S3.FEWSHOT_ASSISTANT_ARGS, "t1"),
                     (S3.FEWSHOT_USER_ABSTAIN, S3.FEWSHOT_ASSISTANT_ABSTAIN_ARGS, "t2"))
            for n, (user, args, tid) in enumerate(shots):
                last = n == len(shots) - 1
                tool_content = "recorded"
                if at_fewshot and last:
                    tool_content = [{"type": "text", "text": "recorded",
                                     "cache_control": mark}]
                msgs += [
                    HumanMessage(content=user),
                    AIMessage(content="", tool_calls=[
                        {"name": S3.RECORD_TOOL["name"], "args": args, "id": tid}]),
                    ToolMessage(content=tool_content, tool_call_id=tid),
                ]
        msgs.append(HumanMessage(content=f"{item_text}\n\n{cand_text}"))
        return msgs

    def _call_anthropic(self, item_text, cand_text) -> dict | None:
        out = self._anthropic_chain().invoke(self._lc_messages(item_text, cand_text))
        self.last_usage = self._usage(out)
        return out.tool_calls[0]["args"] if out.tool_calls else None

    @staticmethod
    def _usage(msg) -> dict:
        """Cache tokens as the API reported them. Both zero means nothing was cached --
        almost always because the prefix is under the model's minimum, which the API
        does not report as an error."""
        u = getattr(msg, "usage_metadata", None) or {}
        d = u.get("input_token_details", {}) if isinstance(u, dict) else {}
        # cache_creation appears in usage_metadata on some langchain-anthropic versions
        # and only in the raw response usage on others. Read both rather than reporting
        # a zero that means "I did not look there".
        raw = (getattr(msg, "response_metadata", None) or {}).get("usage", {}) or {}
        return {"input": u.get("input_tokens"),
                "output": u.get("output_tokens"),
                "cache_read": d.get("cache_read")
                              or raw.get("cache_read_input_tokens") or 0,
                "cache_creation": d.get("cache_creation")
                                  or raw.get("cache_creation_input_tokens") or 0}

    def _call_raw(self, item_text, cand_text) -> dict | None:
        """The batch script's exact message list, through the SDK. Control arm only."""
        from anthropic import Anthropic
        client = Anthropic(default_headers=self._headers() or None)
        r = client.messages.create(
            model=self.cfg.model, max_tokens=self.cfg.max_tokens, system=self.system,
            messages=S3.build_messages(item_text, cand_text, fewshot=self.cfg.fewshot),
            tools=[S3.RECORD_TOOL],
            tool_choice={"type": "tool", "name": S3.RECORD_TOOL["name"]})
        blocks = [c for c in r.content if c.type == "tool_use"]
        return blocks[0].input if blocks else None

    def _call_ollama(self, item_text, cand_text) -> dict | None:
        if self._client is None:
            from langchain_ollama import ChatOllama
            self._client = ChatOllama(
                model=self.cfg.ollama_model, base_url=self.cfg.ollama_base_url,
                temperature=0,
            ).bind_tools([S3.RECORD_TOOL])
        out = self._client.invoke(self._lc_messages(item_text, cand_text))
        return out.tool_calls[0]["args"] if out.tool_calls else None

    def _call_cached(self, item_id: str | None) -> dict | None:
        if self._replay is None:
            path = (Path(self.cfg.cached_csv) if self.cfg.cached_csv
                    else self.data / "stage3_fused.csv")
            if not path.exists():
                raise SystemExit(f"provider 'cached' needs {path}; run stage3_agent.py first.")
            df = pd.read_csv(path)
            # ABO ships 716 duplicate item_ids; dict() keeps the last, matching
            # pipeline.evaluate. Mirrored so the two cannot disagree.
            self._replay = dict(zip(df.item_id.astype(str), df.pred))
        # No item_id means no cached answer, which is the same condition as an id that
        # is not in the file: this provider has nothing to say about the item. It is NOT
        # an error. Raising here made an ad-hoc request to /classify -- the obvious thing
        # to try first -- fail with a 500 instead of returning S2's answer, and made the
        # cascade's behaviour depend on whether the caller happened to know an ABO ASIN.
        # The graph reports stage 'S2' for this, which is exactly what happened.
        if item_id is None:
            return None
        pred = self._replay.get(str(item_id))
        return None if pred is None or pd.isna(pred) else {"leaf_id": pred,
                                                           "confidence": "cached"}

    # ---------- entry point ----------

    def decide(self, item: dict, shortlist_idx, item_id: str | None = None) -> Decision:
        p = self.cfg.provider
        if p == "none":
            return Decision(None, None, False, "none")
        if p == "cached":
            args = self._call_cached(item_id)
        else:
            item_text, cand_text = self._render(item, shortlist_idx)
            args = ({"anthropic": self._call_anthropic, "raw": self._call_raw,
                     "ollama": self._call_ollama}[p])(item_text, cand_text)

        u = self.last_usage
        if not args or not isinstance(args.get("leaf_id"), str):
            # A parse failure is not an abstention. It gets no label and is marked so it
            # cannot be counted as a principled refusal in the attribution table.
            return Decision(None, None, False, p, raw=args, usage=u)
        leaf = args["leaf_id"]
        if leaf == ABSTAIN:
            return Decision(None, args.get("confidence"), True, p, raw=args, usage=u)
        if leaf not in self.leaves:
            # The model invented an id. Same treatment as a parse failure.
            return Decision(None, args.get("confidence"), False, p, raw=args,
                            usage=self.last_usage)
        return Decision(leaf, args.get("confidence"), False, p, raw=args,
                        usage=self.last_usage)


def _dry_run(args) -> None:
    """Build everything the anthropic provider would send, and send nothing.

    This exercises the two things most likely to be wrong the first time: whether
    langchain-anthropic accepts RECORD_TOOL in its native Anthropic shape (it carries
    `input_schema`, not the OpenAI `parameters`), and whether the forced tool_choice and
    the cache_control marker survive binding. Both fail loudly here for free rather than
    inside a paid call.
    """
    import os
    data = Path(args.data)
    leaves = json.loads((data / "stage2" / "leaves.json").read_text())
    st = Stage3(data, leaves, Stage3Config(provider="anthropic"))

    ev, deferred, cands, scores = S3.compose(data, args.s1_tag, args.s2_tag, args.tau)
    i = int(np.where(deferred)[0][0])
    short = build_shortlist(cands, scores, args.shortlist, args.shortlist_mode)[i]
    item = {"title": ev.title.iloc[i], "brand": ev.brand.iloc[i],
            "bullets": ev.bullets.iloc[i]}
    msgs = st._lc_messages(*st._render(item, short))

    print("=== messages LangChain would send ===")
    for m in msgs:
        kind = type(m).__name__
        body = m.content if isinstance(m.content, str) else json.dumps(m.content)[:300]
        extra = f"  tool_calls={m.tool_calls}" if getattr(m, "tool_calls", None) else ""
        print(f"\n--- {kind} ---\n{body}{extra}")

    cached = [c for m in msgs if isinstance(m.content, list)
              for c in m.content if isinstance(c, dict) and "cache_control" in c]
    print(f"\n=== cache_control markers: {len(cached)} ===")

    print("\n=== tool binding ===")
    os.environ.setdefault("ANTHROPIC_API_KEY", "dry-run-not-a-real-key")
    try:
        chain = st._anthropic_chain()
        kw = getattr(chain, "kwargs", {})
        print(f"model        : {st.cfg.model}")
        print(f"headers      : {st._headers() or 'none (key must be workspace-scoped)'}")
        print(f"tool_choice  : {kw.get('tool_choice')}")
        tools = kw.get("tools") or []
        print(f"tools bound  : {len(tools)}")
        if tools:
            t = tools[0]
            t = t if isinstance(t, dict) else t.__dict__
            print(f"tool name    : {t.get('name')}")
            schema = t.get("input_schema") or t.get("parameters")
            print(f"schema keys  : {sorted((schema or {}).get('properties', {}))}")
            print(f"required     : {(schema or {}).get('required')}")
        print("\nBinding succeeded. Nothing was sent and nothing was spent.")
    except Exception as e:
        print(f"BINDING FAILED: {type(e).__name__}: {e}")
        print("\nFix this before spending anything. The usual cause is that "
              "RECORD_TOOL is in\nAnthropic's shape (input_schema) and the installed "
              "langchain-anthropic wants\nthe OpenAI shape (parameters).")
        raise SystemExit(1)


def _count_tokens(args) -> None:
    """Exact token counts per prefix segment, using the free count_tokens endpoint.

    Character-based estimates are badly wrong for this prompt. The candidate block is
    ALL_CAPS_UNDERSCORE leaf ids and taxonomy paths, which tokenise far worse than prose,
    so a ratio calibrated on English understates it by roughly 2x. Every cost figure and
    the cache-threshold question depend on this number, so it is measured.

    Prints the three prefixes that matter:
      tools + system                     what cache_control currently covers
      tools + system + worked examples   what it would cover if the marker moved down
      full request                       what each item actually costs
    """
    import os
    from anthropic import Anthropic
    data = Path(args.data)
    leaves = json.loads((data / "stage2" / "leaves.json").read_text())
    st = Stage3(data, leaves, Stage3Config(provider="anthropic"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (count_tokens is free, but authed).")
    client = Anthropic(default_headers=st._headers() or None)

    ev, deferred, cands, scores = S3.compose(data, args.s1_tag, args.s2_tag, args.tau)
    idx = np.where(deferred)[0][: args.n]
    short = build_shortlist(cands, scores, args.shortlist, args.shortlist_mode)

    def count(msgs):
        return client.messages.count_tokens(
            model=st.cfg.model, system=st.system, tools=[S3.RECORD_TOOL],
            messages=msgs).input_tokens

    # A single minimal user turn isolates the fixed part: the API requires >=1 message.
    base = count([{"role": "user", "content": "x"}])
    fs = count(S3.build_messages("x", "y", fewshot=True))
    full = []
    for i in idx:
        it = {"title": ev.title.iloc[i], "brand": ev.brand.iloc[i],
              "bullets": ev.bullets.iloc[i]}
        full.append(count(S3.build_messages(*st._render(it, short[i]), fewshot=True)))
    full = np.array(full)

    MIN = {"claude-sonnet-4-5": 1024, "claude-sonnet-4-6": 1024, "claude-sonnet-5": 1024,
           "claude-opus-5": 512, "claude-haiku-4-5": 4096}
    lim = MIN.get(st.cfg.model, 1024)

    print(f"\nmodel {st.cfg.model} — minimum cacheable prefix {lim} tokens\n")
    print(f"{'tools + system (marker is here)':<38}{base:>7} tokens  "
          f"{'CACHES' if base >= lim else 'below minimum — inert'}")
    print(f"{'  + worked examples':<38}{fs:>7} tokens  "
          f"{'CACHES' if fs >= lim else 'below minimum'}   <- if the marker moved here")
    print(f"{'full request (mean of %d)' % len(full):<38}{full.mean():>7.0f} tokens  "
          f"p95 {np.percentile(full, 95):.0f}")

    n_esc = int(deferred.sum())
    print(f"\n=== cost at tau={args.tau}, {n_esc:,} escalated items ===")
    out_tok = 30
    for name, pi, po in [("Sonnet 4.5", 3, 15), ("Sonnet 5", 2, 10), ("Haiku 4.5", 1, 5)]:
        sync = (n_esc * full.mean() * pi + n_esc * out_tok * po) / 1e6
        print(f"  {name:<11} sync ${sync:6.2f}   batch ${sync / 2:6.2f}")
    print(f"\nPass --in-tokens {full.mean():.0f} to sweep.py so its dollar columns match.")
    print("Nothing was generated; count_tokens is not billed.")


def _parity(args) -> None:
    """Send the same items through LangChain and the raw SDK; report disagreement."""
    data = Path(args.data)
    leaves = json.loads((data / "stage2" / "leaves.json").read_text())
    ev, deferred, cands, scores = S3.compose(data, args.s1_tag, args.s2_tag, args.tau)
    idx = np.where(deferred)[0][: args.n]
    short = build_shortlist(cands, scores, args.shortlist, args.shortlist_mode)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    rows = []
    for i in idx:
        item = {"title": ev.title.iloc[i], "brand": ev.brand.iloc[i],
                "bullets": ev.bullets.iloc[i]}
        a = Stage3(data, leaves, Stage3Config(provider="anthropic", seed=S3.SEED)
                   ).decide(item, short[i])
        b = Stage3(data, leaves, Stage3Config(provider="raw", seed=S3.SEED)
                   ).decide(item, short[i])
        lab = lambda d: ("ABSTAIN" if d.abstained else  # noqa: E731
                         d.leaf if d.leaf else "FAIL")
        rows.append({"item_id": ev.item_id.iloc[i], "lc": lab(a), "raw": lab(b),
                     "agree": lab(a) == lab(b),
                     "in_tok": (a.usage or {}).get("input"),
                     "cache_read": (a.usage or {}).get("cache_read"),
                     "cache_write": (a.usage or {}).get("cache_creation")})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    real = out[(out.lc != "FAIL") & (out.raw != "FAIL")]
    print(f"\nagreement {out.agree.mean():.1%} over {len(out)} items "
          f"({len(real)} answered, {len(out) - len(real)} failed on at least one path)")
    if len(out) != len(real):
        print("A row where both paths returned nothing agrees trivially. Read the "
              "answered count.")
    cr, cw = out.cache_read.fillna(0).sum(), out.cache_write.fillna(0).sum()
    print(f"cache tokens: {int(cw)} written, {int(cr)} read")
    if cr == 0 and cw == 0:
        print("Both zero: nothing was cached. Check the prefix against the model minimum "
              "with\n--count-tokens; below it the API declines silently.")
    else:
        rd, wr = out.cache_read.fillna(0), out.cache_write.fillna(0)
        tok = out.in_tok.fillna(0)
        hits = out[rd > 0]
        if tot := tok.sum():
            # Billing multipliers: a cache READ costs 0.1x, a 5-minute cache WRITE costs
            # 1.25x. Both matter, and the write is why a short sample understates the
            # saving -- one cold call is 20% of a 5-item run and ~0.05% of a 2,015-item one.
            eff = (tok - rd - wr + 0.1 * rd + 1.25 * wr).sum()
            print(f"cache hits on {len(hits)}/{len(out)} calls; effective input "
                  f"{eff:.0f} vs {tot:.0f} tokens = {1 - eff / tot:.0%} off input "
                  f"cost on THIS sample.")
            if len(hits):
                w = out[rd > 0]
                wt, wr_ = w.in_tok.fillna(0), w.cache_read.fillna(0)
                warm = ((wt - wr_ + 0.1 * wr_).sum()) / wt.sum()
                print(f"warm steady state (the number to quote at scale): "
                      f"{1 - warm:.0%} off input.")
                print(f"  A run amortises one cache write over every item that follows it, "
                      f"so the\n  sample figure above is a floor, not the rate.")
        print("The 5-minute entry stays warm for a sequential run. The Batch API makes no "
              "ordering\nguarantee, so do not assume the same hit rate there.")
    print("Anything below ~95% means the LangChain encoding changed the model's input "
          "and the offline numbers no longer describe the served path.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", action="store_true",
                    help="send the same items through LangChain and the raw SDK and "
                         "report disagreement. COSTS MONEY: 2 calls per item.")
    ap.add_argument("--count-tokens", action="store_true",
                    help="measure exact prompt tokens per prefix segment. Free.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the LangChain request and print it. Spends nothing.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--s1-tag", default="head69")
    ap.add_argument("--s2-tag", default="ft2")
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--shortlist-mode", default="fused")
    a = ap.parse_args()
    if a.count_tokens:
        _count_tokens(a)
    elif a.dry_run:
        _dry_run(a)
    elif a.parity:
        _parity(a)
    else:
        ap.print_help()
