"""The cascade as one loadable object, plus the end-to-end evaluation.

    from pipeline import Cascade
    cascade = Cascade.load("data")
    out = cascade.predict(df)      # leaf, stage, confidence, shortlist — per item

    python src/pipeline.py --data data --evaluate            # S1+S2, free
    python src/pipeline.py --data data --evaluate --with-llm # folds in S3 results

The point of the report is STAGE ATTRIBUTION. An end-to-end accuracy number alone says
nothing about whether the architecture earned its complexity; the same number split by
which tier produced it says everything. A tier that decides a large share at low accuracy
is mis-tuned, not useful.

THREE DOC RENDERINGS, NOT ONE. Each tier reads label documents in the exact form the run
that measured it used, and nothing here is free to choose:
  - the bi-encoder reads the full document of `retriever_variant` (recall.py embeds it
    bare and untruncated; the gate number is only valid for that text);
  - the cross-encoder reads the full document of the variant recorded in its own
    checkpoint's run_config.json, truncated by the tokenizer at that run's max_len;
  - Stage 3 builds its own prompt from label_docs.json via stage3_agent.compose_doc, so
    this module hands it leaf names and stays out of it.
Collapsing these into one variant silently swaps a measured tier for an unmeasured one.

WHY THE SHORTLIST IS FUSED. On the full test split, the fine-tuned cross-encoder loses
to the frozen retriever it was meant to improve — macro recall@10 of 0.760 against 0.919
— because the 90 leaves with no training rows appear in its training groups ONLY as hard
negatives, so it learns to bury them (zero_shot recall@10: 0.124). Its own gate says drop
the tier. But it is strongly RIGHT where it has data, and interleaving the two orderings
keeps both, macro recall@10 over 530 leaves:

    stratum      bi   cross   fused
       head   0.740   0.887   0.877
      torso   0.884   0.962   0.958
       tail   0.920   0.931   0.937
   few_shot   0.958   0.825   0.952
  zero_shot   0.876   0.124   0.860
        ALL   0.919   0.760   0.931

Fusing costs 0.006 on few_shot and 0.016 on zero_shot and buys 0.137 on head and 0.074 on
torso. The cross-encoder is not a shortlist producer; it is a complement to a retriever
that never learned which labels are rare. 'cross' and 'bi' remain selectable to reproduce
the comparison.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from recall import (INSTRUCTION, assert_finite, item_text, load_model,
                    pick_device, save_array)
from shortlist import MODES, build_shortlist

STRATA = ["head", "torso", "tail", "few_shot", "zero_shot"]
OTHER = "__OTHER__"
ABSTAIN = "NONE_OF_THESE"


@dataclass
class Config:
    """Measured operating points. Change these only with a run behind the change."""
    # Fitted on VAL at a 98% precision target, reported on TEST: 83.5% answered at
    # 0.977 accuracy, deferral recall 0.849. The earlier 0.925 came from a table that
    # pooled val and test; the honest refit moves it to 0.900 and costs 0.003 accuracy.
    tau: float = 0.900
    k: int = 50                     # step-2 gate: macro recall@50 = 0.977 vs 0.946 at 20
    shortlist: int = 10             # what S3 reads; +0.155 macro over S2 top-1 at this size
    s1_tag: str = "head69"          # head cut = leaves with >=100 train items (F1 knee)
    s2_tag: str = "ft2"
    retriever_variant: str = "v3_proto"   # the variant the step-2 gate actually scored
    shortlist_mode: str = "fused"   # fused | cross | bi — macro recall@10 .931/.760/.919
    batch_size: int = 32


class Cascade:
    def __init__(self, data: Path, cfg: Config, device: str = "auto"):
        self.data, self.cfg = Path(data), cfg
        self.device = pick_device(device)
        self.fp16 = self.device == "cuda"

        s1 = self.data / "stage1" / f"model_{cfg.s1_tag}"
        s2 = self.data / "stage2" / f"model_{cfg.s2_tag}"
        for p in (s1, s2):
            if not p.exists():
                raise SystemExit(
                    f"Missing checkpoint {p}.\n"
                    "Re-run the stage that produces it — earlier versions of these scripts "
                    "saved predictions and discarded the weights, so the runs behind the\n"
                    "numbers in data/ left nothing loadable behind."
                )
        s1_cfg = json.loads((s1 / "run_config.json").read_text())
        s2_cfg = json.loads((s2 / "run_config.json").read_text())
        self.s1_max_len = int(s1_cfg["max_len"])
        self.s2_max_len = int(s2_cfg["max_len"])
        self.s2_variant = s2_cfg["doc_variant"]
        if int(s2_cfg["k"]) != cfg.k:
            print(f"NOTE: reranker was trained on top-{s2_cfg['k']} candidates but retrieval "
                  f"here uses K={cfg.k}. Its hard negatives came from a different pool.")

        self.s1_tok = AutoTokenizer.from_pretrained(s1)
        self.s1 = AutoModelForSequenceClassification.from_pretrained(s1).to(self.device).eval()
        self.s1_classes = json.loads(
            (self.data / "stage1" / f"classes_{cfg.s1_tag}.json").read_text())
        self.other_i = self.s1_classes.index(OTHER) if OTHER in self.s1_classes else -1

        self.ce_tok = AutoTokenizer.from_pretrained(s2)
        self.ce = AutoModelForSequenceClassification.from_pretrained(s2).to(self.device).eval()

        docs_all = json.loads((self.data / "label_docs.json").read_text())
        self.leaves = json.loads((self.data / "stage2" / "leaves.json").read_text())
        self.ce_docs = [docs_all[l][self.s2_variant] for l in self.leaves]
        retr_docs = [docs_all[l][cfg.retriever_variant] for l in self.leaves]

        # Label embeddings are keyed on the variant: two variants must never share a cache.
        cache = self.data / f"emb_labels_{cfg.retriever_variant}.npy"
        self.retriever = None            # lazy: only deferred items need it
        if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(self.leaves):
            self.label_emb = np.load(cache)
        else:
            self.label_emb = self._encode_labels(retr_docs)
            save_array(cache, self.label_emb, model=self.retriever,
                       device=self.device, variant=cfg.retriever_variant)
        assert_finite(self.label_emb, "label embeddings")

    def _encode_labels(self, docs: list[str]) -> np.ndarray:
        # Documents are embedded bare — the instruction goes on the query side only.
        return self._bi().encode(docs, batch_size=16,
                                 normalize_embeddings=True).astype(np.float32)

    def _bi(self):
        if self.retriever is None:
            self.retriever = load_model(self.device)
        return self.retriever

    @classmethod
    def load(cls, data="data", **kw):
        p = Path(data) / "cascade_config.json"
        cfg = Config(**json.loads(p.read_text())) if p.exists() else Config()
        return cls(Path(data), cfg, **kw)

    def save_config(self):
        (self.data / "cascade_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @torch.no_grad()
    def _s1(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for i in range(0, len(texts), self.cfg.batch_size):
            enc = self.s1_tok(texts[i:i + self.cfg.batch_size], truncation=True,
                              max_length=self.s1_max_len, padding=True,
                              return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                out.append(torch.softmax(self.s1(**enc).logits.float(), -1).cpu().numpy())
        p = np.concatenate(out)
        return p.argmax(1), p.max(1)

    @torch.no_grad()
    def _rerank(self, texts: list[str], cands: np.ndarray) -> np.ndarray:
        n, k = cands.shape
        pairs_a = [texts[i] for i in range(n) for _ in range(k)]
        pairs_b = [self.ce_docs[c] for row in cands for c in row]
        out = np.empty(n * k, dtype=np.float32)
        bs = self.cfg.batch_size * 2
        for i in range(0, len(pairs_a), bs):
            enc = self.ce_tok(pairs_a[i:i + bs], pairs_b[i:i + bs], truncation=True,
                              max_length=self.s2_max_len, padding=True,
                              return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                out[i:i + bs] = self.ce(**enc).logits.float().squeeze(-1).cpu().numpy()
        return out.reshape(n, k)

    def _shortlist(self, cands: np.ndarray, scores: np.ndarray) -> np.ndarray:
        return build_shortlist(cands, scores, self.cfg.shortlist, self.cfg.shortlist_mode)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """leaf, stage (which tier decided), confidence, and the shortlist S3 would read.

        For a deferred item the reported leaf is shortlist[0] — the top of the ordering
        the shortlist mode selects, i.e. S2's own answer if the agent never runs.
        """
        texts = [item_text(r) for r in df.itertuples()]
        pred, pmax = self._s1(texts)
        deferred = (pred == self.other_i) | (pmax < self.cfg.tau)

        leaf = np.array([None] * len(df), dtype=object)
        stage = np.where(deferred, "S2", "S1").astype(object)
        acc = ~deferred
        leaf[acc] = [self.s1_classes[i] for i in pred[acc]]

        shortlists = [None] * len(df)
        idx = np.where(deferred)[0]
        if len(idx):
            emb = self._bi().encode([texts[i] for i in idx],
                                    prompt=f"Instruct: {INSTRUCTION}\nQuery: ",
                                    batch_size=16, normalize_embeddings=True).astype(np.float32)
            assert_finite(emb, "item embeddings")
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = emb @ self.label_emb.T
            part = np.argpartition(-sims, self.cfg.k - 1, 1)[:, :self.cfg.k]
            cands = np.take_along_axis(
                part, np.argsort(-np.take_along_axis(sims, part, 1), 1), 1)
            short = self._shortlist(cands, self._rerank([texts[i] for i in idx], cands))
            for j, i in enumerate(idx):
                shortlists[i] = [self.leaves[c] for c in short[j]]
                leaf[i] = shortlists[i][0]

        return pd.DataFrame({
            "leaf": leaf, "stage": stage,
            "confidence": np.where(deferred, np.nan, pmax),
            "shortlist": shortlists,
        }, index=df.index)


def evaluate(cascade: Cascade, df: pd.DataFrame, s3: pd.DataFrame | None) -> None:
    out = cascade.predict(df)
    gold = df.product_type.to_numpy()

    if s3 is not None:
        # Stage 3 answered some of the deferred items. An abstention is not an answer:
        # it becomes its own tier so the table cannot hide it inside S3's accuracy.
        m = dict(zip(s3.item_id.astype(str), s3.pred))
        col_l, col_s = out.columns.get_loc("leaf"), out.columns.get_loc("stage")
        for i, iid in enumerate(df.item_id.astype(str)):
            if out.stage.iloc[i] != "S2" or iid not in m:
                continue
            p = m[iid]
            if not isinstance(p, str) or not p:
                continue
            out.iloc[i, col_l] = None if p == ABSTAIN else p
            out.iloc[i, col_s] = "S3-abstain" if p == ABSTAIN else "S3"

    hit = out.leaf.to_numpy() == gold
    print(f"\n=== end to end, {len(df):,} items — micro and macro over the SAME set ===")
    print(f"{'stratum':>10s} {'n':>6s} {'leaves':>7s} {'micro':>7s} {'macro':>7s}")
    for st in STRATA + ["ALL"]:
        k = np.ones(len(df), bool) if st == "ALL" else (df.stratum == st).to_numpy()
        if k.sum() == 0:
            continue
        print(f"{st:>10s} {k.sum():>6d} {df.product_type[k].nunique():>7d} "
              f"{hit[k].mean():>7.3f} {pd.Series(hit[k]).groupby(gold[k]).mean().mean():>7.3f}")

    print("\n=== stage attribution — who decided, and were they right ===")
    print(f"{'stage':>11s} {'share':>7s} {'micro':>7s} {'macro':>7s}")
    for st in ("S1", "S2", "S3", "S3-abstain"):
        k = (out.stage == st).to_numpy()
        if k.sum() == 0:
            continue
        print(f"{st:>11s} {k.mean():>7.1%} {hit[k].mean():>7.3f} "
              f"{pd.Series(hit[k]).groupby(gold[k]).mean().mean():>7.3f}")
    print("\nRead the share and the accuracy together or the table will flatter you.")
    print("S3-abstain is refusal, not error: those items are routed to a human, and they")
    print("count as misses above because the cascade produced no label for them.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with-llm", nargs="?", const="stage3_fused.csv", default=None,
                    metavar="CSV",
                    help="fold in a Stage 3 results file for the items it covered "
                         "(default stage3_fused.csv). Name another to score an ablation.")
    ap.add_argument("--shortlist-mode", choices=list(MODES), default=None)
    ap.add_argument("--retriever-variant", default=None,
                    choices=["v1_name", "v2_path", "v3_proto", "v4_desc"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-config", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.shortlist_mode:
        cfg.shortlist_mode = args.shortlist_mode
    if args.retriever_variant:
        cfg.retriever_variant = args.retriever_variant
    cascade = Cascade(args.data, cfg, device=args.device)
    print(f"S1 {cfg.s1_tag} (max_len {cascade.s1_max_len}) | "
          f"S2 {cfg.s2_tag} ({cascade.s2_variant}, max_len {cascade.s2_max_len}) | "
          f"retrieval {cfg.retriever_variant} K={cfg.k} | shortlist {cfg.shortlist} "
          f"({cfg.shortlist_mode}) | tau {cfg.tau}")
    if args.save_config:
        cascade.save_config()
        print(f"Wrote {args.data / 'cascade_config.json'}")

    if not args.evaluate:
        return
    df = pd.read_parquet(args.data / "items.parquet")
    df = df[df.split == args.split].reset_index(drop=True)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=17).reset_index(drop=True)

    s3 = None
    if args.with_llm:
        p = args.data / args.with_llm
        if not p.exists():
            avail = sorted(x.name for x in args.data.glob("stage3_*.csv"))
            raise SystemExit(f"{p} not found — run stage3_agent.py first."
                             + (f" Available: {', '.join(avail)}" if avail else ""))
        s3 = pd.read_csv(p)
        mode = s3.shortlist_mode.iloc[0] if "shortlist_mode" in s3 else "unknown"
        if mode != cfg.shortlist_mode:
            print(f"WARNING: {p.name} was produced against a '{mode}' shortlist but this "
                  f"cascade builds '{cfg.shortlist_mode}'. The agent was scored on a list "
                  "this pipeline would never hand it.")
        print(f"Folding in {len(s3):,} Stage 3 decisions from {p.name} "
              f"(only items it actually covered)")
    evaluate(cascade, df, s3)


if __name__ == "__main__":
    main()
