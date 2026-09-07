# Long-Tail Cascade Classifier — Stage 2 retrieval gate

Design doc: https://claude.ai/code/artifact/97782eb2-444f-4322-8a8b-2aba479ce914

This repo implements **build steps 1–2** of the design: the data audit and the retrieval
gate. That gate is the ceiling on the whole cascade — no downstream stage can recover a
label the retriever never surfaced — so it is both the first thing to measure and the first
thing worth showing.

**No training. No API spend. Runs on a laptop.**

## The result this produces

Macro-averaged recall@20: **label-document variant × label-frequency stratum**, over ABO's
576 `product_type` leaves, using a frozen off-the-shelf embedder.

The claim it tests: *a taxonomy leaf is retrievable in proportion to how much its own text
says about it.* A leaf named `ACCESSORY` is unfindable. The same leaf carrying its ancestor
path and three real product titles is findable — including at zero training examples, which
is the case a softmax classifier cannot handle at all.

## What ABO actually contains (verified, not assumed)

| | |
|---|---|
| Records | 147,702 across 16 gzipped files, 84 MB |
| Leaves (`product_type`) | 576 |
| English `item_name` | 83.1% — the rest are dropped |
| Has a browse `node` | 95.3% (72.6% with an English path) |
| **Largest leaf** | **`CELLULAR_PHONE_CASE` = 44% of the corpus** |
| Top 5 leaves | 63% of the corpus |
| Leaves with <10 items | 231 of 576 |
| Singleton leaves | 58 |

**Two consequences worth understanding before you read any number this repo prints.**

*Accuracy is meaningless here.* Predicting `CELLULAR_PHONE_CASE` for everything scores 44%.
Every metric is reported both micro-averaged (per item) and macro-averaged (per leaf), and
**macro is the one to quote** — micro is close to a report on that single class. The gate
uses macro.

*ABO is a phone-case dataset with a long tail attached.* That makes it a good stress test
for the tail and a poor proxy for a balanced production taxonomy. Say so before an
interviewer notices it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# install torch for your platform FIRST, from https://pytorch.org
pip install -r requirements.txt
```

The `source` line is the one people skip. Your prompt should show `(.venv)`.

## Download ABO listings (84 MB)

Metadata only — the 3D models and turntable images are ~100 GB and are not needed.

```bash
mkdir -p data && cd data
curl -O https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/archives/abo-listings.tar
tar -xf abo-listings.tar
cd ..
```

`prepare.py` searches recursively for `listings_*.json.gz`, so the exact nesting under
`--raw` does not matter.

ABO is CC BY-NC 4.0 — non-commercial use, attribution required. `.gitignore` keeps `data/`
out of the repo; do not remove that line.

## Run

```bash
python src/prepare.py    --raw data --out data          # ~3-5 min
python src/label_docs.py --data data                    # seconds
python src/recall.py     --data data --limit 2000       # smoke test, ~2 min
python src/recall.py     --data data                    # real numbers
```

`recall.py` downloads Qwen3-Embedding-0.6B on first run (~1.2 GB, once). It picks CUDA,
then MPS, then CPU. `--limit` subsamples for a plumbing check — the rare strata end up with
too few items to trust, so never quote numbers from a limited run.

Optional, ~$0.60 and an `ANTHROPIC_API_KEY`, adds a fourth label-doc variant:

```bash
python src/label_docs.py --data data --describe
```

## What each step gives you

| Step | Output | Why it matters |
|---|---|---|
| `prepare.py` | `data/items.parquet`, `data/strata.csv`, `figs/frequency.png` | The long tail, drawn. Confirms the strata are populated enough to measure anything. |
| `label_docs.py` | `data/label_docs.json` | Four variants of the highest-leverage artifact in the system. |
| `recall.py` | `data/recall.csv` + printed table | **The gate.** macro@20 ≥ 0.95 overall and ≥ 0.85 on tail means the retriever needs no fine-tuning. |

## Two things the data forced

Both were found by checking ABO rather than trusting its documentation, and both are worth
mentioning if anyone asks how carefully this was built.

**Node paths carry no language tag.** All 169,347 `node` entries lack the `language_tag`
key that ABO's other localised fields use. About 30% of paths are Spanish, German, Japanese,
French, Dutch, Swedish or Turkish (`/Categorías`, `/カテゴリー別`, `/Kategorien`, …), and the
non-English ones are frequently the longest. Selecting the longest path would have silently
mixed languages into the label documents. `prepare.py` whitelists English roots instead.

**The zero-shot stratum is constructed, not found.** Every `product_type` in ABO has at
least one item, so no true zero-shot stratum exists. `prepare.py` builds one: it deletes
every *training* row for 30 leaves while keeping their val/test rows and their label
documents. Those leaves become unreachable by any supervised classifier and reachable only
through label text. `label_docs.py` prints the count of leaves with no prototypes as a
consistency check — it should equal 30.

## Reading the table in an interview

The interesting comparison is not the headline number, it is the **shape across columns**.
Expect name-only to hold up on head classes and fall away toward the tail, with the gap
between variants widening as frequency drops. That gap is the argument for treating labels
as documents rather than as output slots — the reason Stage 2 is retrieval and not a second
classifier.

If macro@20 is high everywhere, including name-only, say so plainly: ABO's `product_type`
names are unusually self-describing, the task is easier than a production taxonomy, and the
result will not transfer to a Korean corpus with terser category names. Volunteering that
caveat is worth more than a good number without it.

## Not implemented yet

Stage 0 (LLM labelling), Stage 1 (BERT + `OTHER`), Stage 2's cross-encoder rerank and
conformal calibration, Stage 3 (LLM adjudication), and the Korean arms. All specified in
the design doc.
