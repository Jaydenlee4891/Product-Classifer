# Long-Tail Cascade Classifier — Stage 2 retrieval gate

Design doc: https://claude.ai/code/artifact/97782eb2-444f-4322-8a8b-2aba479ce914

This repo currently implements **build steps 1–2** of the design: the data audit and the
retrieval gate. That gate is the ceiling on the whole cascade — no downstream stage can
recover a label the retriever never surfaced — so it is both the first thing to measure and
the first thing worth showing.

**No training. No API spend. Runs in under an hour on a laptop.**

## The result this produces

A recall@20 table: **label-document variant × label-frequency stratum**, over ABO's 576
`product_type` leaves, using a frozen off-the-shelf embedder.

The claim it tests: *a taxonomy leaf is retrievable in proportion to how much its own text
says about it.* A leaf named `ACCESSORY` is unfindable. The same leaf carrying its ancestor
path and three real product titles is findable — including when it has zero training examples,
which is the case a softmax classifier cannot handle at all.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is deliberately not pinned in `requirements.txt` — install the build that matches your
CUDA version first, from https://pytorch.org.

## Download ABO listings (~100 MB compressed)

Metadata only. The 3D models and turntable images are ~100 GB and are not needed.

```bash
mkdir -p data/raw && cd data/raw
curl -O https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/archives/abo-listings.tar
tar -xf abo-listings.tar          # -> listings/metadata/listings_[0-f].json.gz
cd ../..
```

If the archive path 404s, browse https://amazon-berkeley-objects.s3.amazonaws.com/index.html
and grab `listings/metadata/listings_*.json.gz` individually. `prepare.py` takes whatever
directory contains those files.

ABO is CC BY-NC 4.0 — non-commercial use, attribution required.

## Run

```bash
python src/prepare.py     --raw data/raw --out data
python src/label_docs.py  --data data
python src/recall.py      --data data --batch-size 16
```

`prepare.py` prints the record schema it found before parsing anything, and stops with a clear
error if ABO's field names differ from what it expects. Read that output on the first run.

## What each step gives you

| Step | Output | Why it matters |
|---|---|---|
| `prepare.py` | `data/items.parquet`, `figs/frequency.png` | The long tail, drawn. Confirms the strata are populated enough to measure anything. |
| `label_docs.py` | `data/label_docs.json` | Four variants of the highest-leverage artifact in the system. |
| `recall.py` | `data/recall.csv` + printed table | **The gate.** recall@20 ≥ 0.95 overall and ≥ 0.85 on tail means the retriever needs no fine-tuning. |

## The zero-shot stratum is constructed, not found

Every `product_type` in ABO has at least one item, so a true zero-shot stratum does not exist
in the raw data. `prepare.py` builds one: it holds out every training example for 30 leaves
while keeping their test items and their label documents. Those leaves are then unreachable by
any supervised classifier and reachable only through their label text.

This is stated in `prepare.py`'s output and should be stated in any write-up. A zero-shot
number that quietly came from leaves the model had seen would be worthless.

## Reading the table in an interview

The interesting comparison is not the headline number, it is the **shape across columns**.
Expect the name-only variant to hold up on head classes and collapse toward the tail, and the
gap between variants to widen as frequency drops. That gap is the argument for treating labels
as documents rather than as output slots — the whole reason Stage 2 is retrieval and not a
second classifier.

If recall@20 is high everywhere even for name-only, say so plainly: it means ABO's
`product_type` names are unusually self-describing, the task is easier than a production
taxonomy, and the result will not transfer to a Korean corpus with terser category names.
That caveat is worth more in an interview than a good number without it.

## Not implemented yet

Stage 0 (LLM labelling), Stage 1 (BERT + `OTHER`), Stage 2's cross-encoder rerank and conformal
calibration, Stage 3 (LLM adjudication), and the Korean arms. All specified in the design doc.
