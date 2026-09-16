# Serving parity: does the endpoint reproduce the offline numbers?

`src/serve/` runs the cascade as a routed graph behind an HTTP endpoint. The published
result — micro 0.895, macro 0.647 on 12,200 held-out items — came from the offline path.
This is the check that the served path is the same classifier, and the record of what
turned out not to be.

**Short answer.** Routing is identical. Retrieval is not, and the reason is that the
cached embeddings were computed on a different machine than everything else. The cost is
bounded at **≤1.0% of micro** (95% upper bound). Nothing here is a defect in the serving
layer; it is a property of the artefacts it was checked against.

Reproduce with:

```
python src/serve/verify.py --data data --n 400 --deferred-only
python src/serve/diagnose_embedding.py --data data
python src/provenance.py --data data
```

---

## What agrees

`verify.py`, 400 items sampled from those the offline run deferred, on MPS:

| check | result |
|---|---|
| S1 argmax mismatches | 0 / 400 |
| max &#124;p_max&#124; deviation | 2.44e-06 |
| **defer decision flips** | **0 / 400** |
| S2 top-1 agreement | 393 / 400 (98.25%) |

Zero defer flips is the important line. Every item reaches the same tier in the served
path as it did offline, so the stage-attribution table describes the endpoint. The
2.44e-06 on p_max is fp32 accumulation noise from batching 32 offline against 1 per
request, and it is three orders of magnitude too small to move a decision at τ=0.900.

## What does not

| check | result |
|---|---|
| identical shortlist ordering | 1 / 400 |
| identical shortlist membership | 33 / 400 |
| mean members shared | 8.46 / 10 |
| gold reachable, offline | 359 / 400 |
| gold lost by served path | 13 |
| gold gained by served path | 3 |
| gold-loss rate | 3.62%, 95% Wilson [2.13%, 6.10%] |
| **end-to-end micro impact** | **≤ 1.006% of all items** |

Ordering churn is not itself a problem: `stage3_agent` shuffles the candidates before
sending them, precisely because position anchors the model. A reordering upstream of a
shuffle is washed out before the model sees it. Membership is the part that can change an
answer, and only when the **gold** leaf is the member that moves — a leaf outside the ten
cannot be selected. Net reachability falls from 359/400 to 349/400, or 2.5 points.

## Why

Four hypotheses were tested and three were wrong.

**Precision on the serving side — no.** `--retriever-fp16` forces the bi-encoder to half
precision on any backend. It moved identical-membership 33→34 and top-1 393→395, and left
gold-loss at exactly 13. Matching the dtype does not match the machine.

**Text construction — no.** `recall.embed_items` and the graph's `s1` node both build
their input with `recall.item_text`. Same function, same 1000-character truncation.

**Batch size and pooling — no.** Qwen3-Embedding pools the last token, which is only
correct under left padding, and the tokenizer's default `padding_side` is `'right'` — so
this looked like the answer. It is not. `diagnose_embedding.py` measures the behaviour
instead of reading the attribute: the same item encoded alone and in batches of 2, 8 and
16 padded by the longest item in the split differs by at most **2.19e-04**, *identical*
at every batch size, and exactly 0.00e+00 for items that are themselves the longest.
That is padding arithmetic under correct masking, not pooling over pad tokens, which
would have shown ~1e-1 and grown with batch size. sentence-transformers sets the padding
side at encode time; the tokenizer attribute is not the behaviour.

**The machine — yes.** Neither batch size reproduces the cache:

```
              cache vs b=1   cache vs b=16
                  4.48e-04        4.30e-04
                  4.89e-04        4.09e-04
                  1.56e-04        1.56e-04
```

Both columns sit at ~4e-04 regardless. The cache is not a local encoding at any batch
size. `provenance.py` says why: every `emb_*.npy` has an fp16 fingerprint in every value
while `probs_*.npy` and `scores_*.npy` are fp32. `recall.load_model` requests fp16 only
on CUDA, so **the embeddings were produced on the CUDA card and the rest on the Apple
GPU** — and both `run_config.json` files recorded `"device": "auto"`, the flag passed,
not the device it resolved to. Nothing on disk said so; it was recovered from bit
patterns.

A ~4e-4 per-vector difference is small in isolation. In a 530-way retrieval where
adjacent candidates are separated by very little, it reorders the top-50 enough to change
the tail of a fused top-10 on 92% of deferred items.

## Why this is not fixed

There is no single machine that reproduces the offline result, because the offline result
was not produced on one. Serving on the CUDA card would match the retrieval cache and
then diverge on S1 and the cross-encoder, whose arrays are fp32 and which autocast to
fp16 there. The trade would be one divergence for another.

The principled fix is to regenerate every embedding on one machine in one precision. That
invalidates `cands_*.npy`, the cross-encoder scores (their candidate pools change), and
the Stage 3 answers (produced against shortlists that would no longer exist), so it costs
a full re-run of stages 2 and 3 and moves the published numbers. It is the right thing to
do before any claim about retrieval is made in a paper. It is not worth doing to remove a
bounded ≤1% effect from a demo.

## What changed as a result

- `recall.save_array` writes a `<name>.meta.json` beside every saved array recording the
  resolved device, dtype, library versions and platform.
- `stage1.py` and `stage2_rerank.py` record `resolved_device` and `fp16_autocast` in
  `run_config.json` alongside the requested flag.
- `src/provenance.py` audits `data/`, reading sidecars where they exist and inferring
  precision from the fp16 bit-pattern test where they do not.

## A note on measurement

Two numbers in this document's history were artefacts of how they were measured, and both
looked like findings first.

Shortlists were initially compared with ordered equality, which reported 100% mismatch on
a path whose routing was identical — `fused` interleaves two orderings, so one swapped
pair marks a whole row unequal. And the S1↔S2 alignment was keyed on `item_id`, which ABO
duplicates; a dict keeps the last occurrence, so 27 of 12,200 test rows (0.22%) were
scored against a different item's vector, which produced a 1.08e-01 maximum cosine
distance that vanished once alignment moved to row position.

`stage3_agent.compose` and `sweep.compose` align the same two slices by `item_id` and
carry the same latent flaw. It does not corrupt their numbers — they intersect slices
rather than compare vectors elementwise — but 56 of those duplicates are filed under two
different leaves, and it should be fixed.
