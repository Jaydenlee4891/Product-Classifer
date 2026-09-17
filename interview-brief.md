# Long-Tail Cascade Classifier — interview preparation brief

## HOW TO USE THIS FILE (instructions for the assistant reading it)

You are tutoring Jayden, an Applied Statistics student at Yonsei preparing for **agentic AI
engineering** interviews at Korean startups. This file is the complete factual record of a
project he built. Your job is not to praise it or summarise it back to him.

**Act as a demanding technical interviewer and a tutor, in that order.**

- Ask him questions from §11 and invent harder ones. Do not accept a vague answer — if he
  says "it improves the tail," ask *which* tail stratum, by how much, measured how, and
  against what baseline. The numbers are all in this file; he should be able to produce
  them or reason to them.
- When he gets something wrong, say so directly and show him the number that contradicts
  him. He has explicitly said he wants unvarnished assessment and pushes back on inflated
  claims about his own work.
- Distinguish **recall** from **understanding**. If he recites "macro recall@50 is 0.977,"
  ask him why macro rather than micro, and what would change if one leaf were 90% of the
  corpus instead of 53%. Parroting a number is not defending it.
- When he cannot answer, teach the concept from first principles, then re-ask it later in
  the session. Concepts he must own are listed in §10.
- He is an applied statistics student — lean on that. He should be more comfortable with
  McNemar and conformal prediction than with CUDA. Push the ML systems side harder.
- Do **not** let him overclaim. §12 sets out precisely what this project does and does not
  support. If he starts saying "I built a production classifier" or "I beat the baseline,"
  correct him.

**The roles he is targeting ask for: hands-on LLM experience, LangChain, and prompt
engineering.** §12 maps the project to those three. **This file was revised in September 2026
after he built and shipped a serving layer** — the LangChain gap described in earlier versions
is closed, and §12.3 now describes what he actually built rather than what he planned. §12.4
covers LangSmith and §12.5 covers serving the system from Django. Drill §12 as hard as the
rest; it is where the interview will actually go.

A note on how to use the new material. The serving work produced several findings that are
*more* interesting than the serving layer itself — a cross-device reproducibility defect in
his own artefacts, a prompt-cache marker that never fired, and two occasions where a
measurement was wrong before the code was. Those are in §9.4 and §12.3. They are the
strongest material in this file, because they are evidence of a habit rather than a result:
he measured instead of inferring, and the inference lost every time.

Suggested opening: ask him to give the 60-second pitch (§2) cold, without looking. Then
attack it.

---

## 1. What the system is, in one paragraph

A four-tier cascade that assigns products to a 530-leaf taxonomy where the label
distribution is Zipfian: one leaf holds 53% of the training data and 90 leaves have **zero**
training examples. Each tier handles what it can and defers the rest, so cost and capability
rise only for items that need them. A cheap BERT classifier answers 83.5% of items; frozen
retrieval over label *text* generates candidates for the rest; an LLM agent picks the final
label from a 10-candidate shortlist or abstains.

Corpus: Amazon Berkeley Objects (ABO), English listings, 121,133 items, `product_type` as
the target label.

---

## 2. The 60-second pitch

> Product taxonomies are long-tailed. In my corpus one category is 53% of the data and 90
> categories have no training examples at all — and a softmax classifier can't *represent* a
> label it never saw, so its score on those is zero by construction, not by weakness.
>
> I built a four-stage cascade. Stage 1 is a DistilBERT classifier over the 69 categories
> that have enough data, plus an explicit `OTHER` class; it answers 83.5% of items at 97.7%
> accuracy and defers the rest on two triggers — predicting `OTHER`, or max-softmax below a
> threshold I fitted on validation. Stage 2 is a frozen 0.6B embedding model retrieving over
> label *documents* — category name, ancestor path, example titles — which is what makes
> zero-shot categories reachable at all, plus a fine-tuned cross-encoder that rescores them.
> Stage 3 is an LLM agent that reads a 10-candidate shortlist and picks one via a forced tool
> call, or abstains.
>
> End to end: micro 0.895, macro 0.647 on held-out test. A TF-IDF linear SVM baseline gets
> micro 0.919, macro 0.507 — so I lose micro by 0.024 and win macro by 0.140, and the entire
> win is in the two rarest strata. On zero-shot the baseline scores exactly 0.000 and I get
> 0.407.
>
> The most interesting thing I found is that every component trained on the observed label
> distribution learns to suppress the labels it didn't observe — I can show that three
> separate ways in the same system. The frozen retriever is load-bearing precisely because
> it never learned the distribution.

---

## 3. Why a cascade at all — the load-bearing argument

Interviewers will ask "why not just train one model?" The answer has two parts and he must
give both.

**Part 1 — representational, not statistical.** A softmax classifier has one output unit per
class seen in training. For the 90 zero-shot leaves there is no unit. No amount of data for
other classes creates one. The TF-IDF baseline scores **0.000** on zero-shot, and that is not
a tuning failure — it is the definition of the model class. Retrieval sidesteps this by
scoring an item against *label text* rather than against learned weights, so a category that
has never been seen is still reachable if its name and description say what belongs in it.

**Part 2 — economic.** 83.5% of items are answered by a 66M-parameter model at 97.7%
accuracy. Only the residual 16.5% touches a 600M retriever, a cross-encoder and an LLM. The
cascade is a way of spending capability where it changes the answer.

**The honest counter he must be ready for:** the baseline comparison shows this cascade is
*over-engineered* for four of five strata. See §9.1.

---

## 4. Architecture and data flow

```
item ──▶ S1  DistilBERT, 69 head classes + OTHER
         │   answers 83.5% at 0.977 accuracy, 0.394 macro
         │   defers when  argmax == OTHER  OR  p_max < 0.900
         ▼
        S2  frozen Qwen3-Embedding-0.6B → top-50 label documents
         │   fine-tuned MiniLM cross-encoder rescores those 50
         │   the two orderings are INTERLEAVED → 10-candidate shortlist
         ▼
        S3  LLM reads the shortlist, picks one leaf via forced tool call, or abstains
```

**Key structural fact:** in the shipped configuration S2 never emits a final label. Its
entire contribution is the shortlist. It is a *candidate generator*, not a classifier. In the
final stage-attribution table S2's share is 0%.

---

## 5. The data — know these cold

| | value |
|---|---|
| ABO raw | 147,702 records, 576 leaves, 84 MB |
| After English-title filter | **121,133 items, 530 leaves** |
| English `item_name` | 83.1% |
| Splits (80/10/10 per leaf, seed 17) | train 96,857 / val 12,076 / test 12,200 |
| Largest leaf | `CELLULAR_PHONE_CASE` — 44% of raw, **53.5% of train** |
| Duplicate `item_id`s | 716 (216 exact repeats, 56 filed under two leaves) |
| Duplicates spanning splits | **0** — assignment is keyed on `item_id` |

### Strata (thresholds chosen in advance; distribution is ABO's)

| stratum | rule | leaves | actual train counts |
|---|---|---|---|
| head | ≥ 1000 | 7 | 1,149 – 51,808 |
| torso | 100 – 999 | 62 | 100 – 970 |
| tail | 10 – 99 | 182 | 10 – 97 |
| few_shot | 1 – 9 | 189 | 1 – 9 |
| zero_shot | 0 | 90 | 0 |

**Defence of the thresholds:** every boundary falls in an empty gap — nothing exists between
971 and 1,148, or between 98 and 99. The cuts can move inside those gaps and not one leaf
changes stratum. The strata are not an artifact of where the knife went.

**Two constructed things he must disclose before being asked:**

1. **zero_shot is manufactured.** ABO has no zero-shot leaves. `prepare.py` deletes all
   *training* rows for 30 leaves while keeping their val/test rows and label documents; 60
   more arrive naturally as singletons whose only item went to test. 90 total.
2. **Every leaf gets ≥1 test row** by rule, not by proportional sampling — 245 leaves have
   <10 items and would otherwise contribute nothing to the test set, making the tail strata
   unmeasurable.

---

## 6. Stage-by-stage numbers

### Stage 1 — DistilBERT + OTHER

Config: `distilbert-base-uncased`, 69 classes + `OTHER`, max_len 128, batch 32, lr 3e-5, 3
epochs, AdamW (weight decay 0.01), linear warmup 6%, grad clip 1.0, fp16 + GradScaler.
9,078 optimizer steps = `floor(96,857 / 32) × 3`.

| metric | value |
|---|---|
| τ (fitted on **val**, 98% precision target) | **0.900** |
| test: answered / accuracy / deferral recall | **83.5% / 0.977 / 0.849** |
| val: answered / accuracy / deferral recall | 83.8% / 0.980 / 0.866 |
| macro-F1 over the 69 classes | 0.830 (median 0.854, zero classes at F1=0) |
| head cut | ≥100 train items = 69 leaves = head(7) + torso(62) |

**`OTHER`'s training set is the whole story:** 6,397 rows — **5,793 tail, 604 few_shot, 0
zero_shot**. It is a learned class whose training data *is* the tail. It recognises unseen
items of *seen* tail leaves. It cannot recognise an unseen *leaf*.

**Who defers what, at τ = 0.90:**

```
   stratum      n  by OTHER   by tau    kept | gold in S1's label space
      head  17557      0.9%     4.0%   95.1% |      100.0%
     torso   4718      4.0%    25.3%   70.7% |      100.0%
      tail   1242     79.1%    15.4%    5.6% |        0.0%
  few_shot    334     72.8%    18.0%    9.3% |        0.0%
 zero_shot    425     31.1%    28.2%   40.7% |        0.0%
```

**Confidence when S1 is wrong and does not say `OTHER`:**

```
   stratum  n wrong  mean p_max  p_max>0.9  p_max>0.99
      head      288       0.742      24.7%        1.7%
 zero_shot      293       0.845      59.0%        9.2%
```

The model is **most confident exactly where it is always wrong**.

### Stage 2 — frozen retriever + fine-tuned cross-encoder

Retriever: `Qwen3-Embedding-0.6B`, **frozen**, instruction prompt on the query side only,
documents embedded bare. Exact flat search (530 leaves — no ANN needed).

Label-document variants, macro recall@50 (ALL):

| variant | content | macro@50 |
|---|---|---|
| v1_name | leaf name only | 0.837 |
| v2_path | + ancestor path | 0.945 |
| **v3_proto** | + prototype titles from train | **0.977** |
| v4_desc | + LLM-generated description | untested (see §9.4) |

macro@20 = 0.946, macro@50 = 0.977 → **K = 50**, because widening is free and no downstream
stage can recover a leaf the retriever never surfaced.

Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M params), 2 epochs, lr 2e-5, listwise
softmax over groups of 8 (1 gold + 7 hard negatives mined from the retriever's own top-50),
cap 200 items per leaf, max_len 192.

**Training pool after capping: 18,435 of 96,857.** 39 leaves hit the cap (86,222 rows →
7,800); 401 leaves kept everything (10,635 rows). Composition: **torso 57.7%, tail 31.4%,
head 7.6%, few_shot 3.3%, zero_shot 0%.**

**The reranker fails its own gate.** Macro recall@10:

| stratum | bi (frozen) | cross (fine-tuned) | **fused** |
|---|---|---|---|
| head | 0.740 | 0.887 | 0.877 |
| torso | 0.884 | 0.962 | 0.958 |
| tail | 0.920 | 0.931 | 0.937 |
| few_shot | 0.958 | 0.825 | 0.952 |
| zero_shot | 0.876 | **0.124** | 0.860 |
| **ALL** | 0.919 | **0.760** | **0.931** |

0.124 on zero-shot is *below* the 0.200 you would get by shuffling the 50 candidates at
random. To reach 90% recall the bi-encoder needs m=7; the cross-encoder needs m=34.

**Fused** interleaves the two orderings and dedups: take bi's next, then cross's next, skip
anything already taken, stop at 10. Costs 0.006 on few_shot and 0.016 on zero_shot; buys
0.137 on head and 0.074 on torso.

### Stage 3 — the agent tier

Claude via the **Batch API** (50% discount, ≤100k requests / 256MB per batch, ~1 hour).
2,015 escalated items, **~$2.70**. A 300-item pilot cost ~$0.40.

Design choices, all deliberate:

| choice | reason |
|---|---|
| **Forced tool call** (`tool_choice: {type: "tool", name: "record_category"}`) | guarantees structured output; no JSON parsing, no retry loop. **0 hallucinated leaf ids in 2,015 answers** |
| **Categorical** confidence enum, not numeric | LLMs produce poorly calibrated numbers; an ordinal enum is what they can actually distinguish |
| **Candidates shuffled** | controls position/anchoring bias — the model must read the documents, not trust rank 1 |
| **2-shot, one example being an abstention** | teaches the refusal explicitly; without it abstention is almost never used |
| **`NONE_OF_THESE`** as a first-class output | the shortlist misses gold ~9.4% of the time; anything but abstention there is a guaranteed wrong label |
| **No rationale tokens** | output is `{leaf_id, confidence}` only — rationale roughly triples output cost for no measured gain |

**Results, paired McNemar (exact, two-sided) against simply taking the shortlist's top
candidate — the correct baseline, because that is what the system would do with no agent:**

```
   stratum     n  agent   top1     net  a>t  t>a        p
      head   408  0.034  0.076  -0.042    7   24   0.0033 **
     torso   708  0.455  0.304  +0.151  148   41   0.0000 ***
      tail   581  0.719  0.670  +0.050   72   43   0.0087 **
  few_shot   169  0.775  0.734  +0.041   21   14   0.3105
 zero_shot   149  0.584  0.430  +0.154   39   16   0.0027 **
       ALL  2015  0.482  0.408  +0.074  287  138   0.0000 ***
```

Ceiling (gold leaf present in the shortlist): **0.906**. The agent captures ~70% of it
macro-averaged.

**Abstention** — the weakest component, and he should say so first: 7.6% rate, 190
unanswerable items, **25.3% recall, 31.4% precision, 105 answerable items refused** (5.2% of
the run discarded for nothing).

---

## 7. End-to-end results

| stratum | n | leaves | micro | macro |
|---|---|---|---|---|
| head | 8,792 | 7 | 0.951 | 0.805 |
| torso | 2,355 | 62 | 0.817 | 0.794 |
| tail | 620 | 182 | 0.674 | 0.662 |
| few_shot | 189 | 189 | 0.693 | 0.693 |
| zero_shot | 244 | 90 | 0.357 | 0.407 |
| **ALL** | **12,200** | **530** | **0.895** | **0.647** |

### Stage attribution — the table he should walk an interviewer through

```
      stage   share   micro   macro
         S1   83.5%   0.977   0.394
         S3   15.3%   0.522   0.715
 S3-abstain    1.3%   0.000   0.000
```

**Why this table is the point:** S1 is a per-item winner and a per-leaf loser — 97.7% of the
items it answers are right, but averaged over the *leaves* in its accepted stream it is at
0.394, because the rare items it wrongly keeps are spread across many leaves that each score
zero. S3 is the mirror image. A system reporting only micro would show 97.7% at the first
tier and nobody would build the rest.

### Against the baseline, same 12,200 items

TF-IDF: word 1–2grams ∪ char_wb 3–5grams (280,267 features, 525 nonzero/doc) → LinearSVC
over all 440 trainable leaves.

| stratum | TF-IDF micro | cascade | TF-IDF macro | cascade | Δ macro |
|---|---|---|---|---|---|
| head | **0.981** | 0.951 | **0.917** | 0.805 | −0.112 |
| torso | **0.874** | 0.817 | **0.849** | 0.794 | −0.055 |
| tail | **0.721** | 0.674 | **0.696** | 0.662 | −0.034 |
| few_shot | 0.439 | **0.693** | 0.439 | **0.693** | **+0.254** |
| zero_shot | 0.000 | **0.357** | 0.000 | **0.407** | **+0.407** |
| **ALL** | **0.919** | 0.895 | 0.507 | **0.647** | **+0.140** |

Majority-class baseline: micro 0.531, macro 0.002.

---

## 8. Findings that generalise beyond this project

These are what make it interesting. Each is measured, not asserted.

### 8.1 Anything trained on the observed label distribution suppresses the unobserved tail

Three independent demonstrations in one system:

- `OTHER` catches 79% of tail and 73% of few-shot but only **31%** of zero-shot — because
  its training set contains zero zero-shot rows.
- The fine-tuned reranker drops zero-shot to macro recall@10 = **0.124**, below chance.
- `sqrt` class weighting improved head/torso deferral and made tail deferral **worse**
  (0.796 → 0.696).

**Consequence:** the frozen retriever is load-bearing *because* it never learned the label
distribution. Freezing it was not a concession to a 6GB GPU.

### 8.2 "Cross-encoders beat bi-encoders" is a claim about comparable models

His cross-encoder is 22.7M params trained on MS MARCO web passage ranking. His retriever is
~595M, instruction-tuned for retrieval. A 26× parameter gap plus a task mismatch. Joint
attention does not cover that.

Proof it is capacity and not fine-tuning damage: the **off-the-shelf** cross-encoder, never
trained on ABO, already loses everywhere (ALL 0.728 vs 0.915; zero_shot 0.153).

Fine-tuning then *steepens an existing tilt*. Mean position in the 50-candidate list by the
leaf's training frequency:

```
                 zs   1-10   -100    -1k    >1k     span
bi (control)   27.3   28.0   25.4   21.8   15.1     12.2
off-shelf CE   32.4   24.8   20.9   19.5   17.9     14.5
fine-tuned CE  40.0   28.4   17.6    8.8    3.3     36.7
```

### 8.3 Max-softmax is the wrong instrument for out-of-vocabulary detection

τ's largest contribution is **torso**, not the tail. Zero-shot errors carry the *highest*
confidence (mean 0.845, 59% above p=0.9). Max-softmax measures how peaked the distribution is
over classes the model knows; it says nothing about whether the true answer is among them.

**Division of labour: `OTHER` is the tail router, τ is the torso trimmer, and nothing in S1
is a novel-leaf detector.** The known fix he did not have time to test: energy score
`-logsumexp(logits)` or max-logit, which preserve the magnitude information softmax
normalises away.

### 8.4 The metric that describes a classifier is not the metric that describes its role

`sqrt` class weighting scores **+0.005 macro-F1** (0.830 → 0.835) and loses **every** cascade
metric at the operating point — fewer items answered, lower accuracy, lower deferral recall,
all at once. Selecting on macro-F1 would have shipped the worse model.

### 8.5 A 26-item table said the opposite of the truth

An early Stage 3 pilot on 300 items showed zero-shot at −0.106 versus the shortlist top-1. It
was written up as a regression, a mechanism was proposed, and a $0.40 ablation was run to fix
it. The same measurement at two sample sizes:

```
  n= 26 : -0.106   discordant  3 vs  6   p=0.508
  n=149 : +0.154   discordant 39 vs 16   p=0.0027
```

Same system, same configuration, **opposite conclusion** — the original rested on three
items. Macro averaging hid it: with 18 leaves inside 26 items, one item moving shifts macro by
0.04.

**The lesson is not "small samples are noisy."** It is: a per-stratum table without its
discordant counts beside it is not a result. He now reports paired tests.

---

## 9. Weaknesses — he must raise these before the interviewer finds them

### 9.1 The cascade loses to a linear SVM on three of five strata

Head, torso and tail all go to TF-IDF, on both micro and macro. The entire macro win is
few_shot and zero_shot.

**How to answer:** state it first, then give the design finding it forces —

> S1's head cut was set at ≥100 training items from DistilBERT's per-class F1 knee. That knee
> is a property of DistilBERT, not of supervised learning: TF-IDF still gets 0.696 macro on
> leaves with 10–99 examples, where my cascade — which defers all of them — gets 0.662. The
> cut was calibrated to the wrong model. What my own ablation argues for is a cheap
> supervised tier over all 440 trainable leaves, with retrieval and the agent reserved for
> the 90 leaves that genuinely have no training data.

Being able to name the simpler system your own evidence supports is worth more than a clean
win. Very few candidates do this.

### 9.2 The agent is significantly *worse* than its own shortlist top-1 on head

−0.042, p=0.0033, 7 vs 24 discordant. Small absolutely (17 items of 408) and on a subset
where both options are poor (0.034 vs 0.076) because these are exactly the head items S1
could not handle — but significant, and the only negative sign in the table. Obvious fix,
untested: route head-stratum deferrals around S3.

### 9.3 Abstention targeting is a coin flip

31.4% precision, 25.3% recall, 105 answerable items refused. This is the largest single pool
of recoverable error in the system.

### 9.4 Methodology bugs found and fixed — say these proactively, they are credibility

- **τ was fitted on val+test pooled** and reported on the same pool. Refit on val alone: τ
  moves 0.925 → 0.900, test lands at 0.977 against the 0.980 val met. Selection optimism =
  0.003 accuracy, 0.017 deferral recall.
- **The retrieval gate credited the wrong variant.** `recall.csv` predates the description
  run, so macro@50 = 0.977 belongs to `v3_proto`; `v4_desc` has never been gated.
- **Three tiers were sharing one label-document rendering.** Retrieval needs the full
  untruncated document; the reranker needs the variant recorded in its own checkpoint; the
  agent needs its prompt rendering. Collapsing them silently swaps a measured tier for an
  unmeasured one.
- **The shortlist was implemented twice** and the two drifted — a results file produced
  against a cross-only shortlist was folded into a fused-mode evaluation before it was
  caught. Now one implementation, with `shortlist_mode` written into every results file.
- **Duplicate ASINs broke the batch.** 716 duplicate `item_id`s; the Batch API requires
  unique `custom_id`s and rejected 2,015 requests over 6 collisions. Fixed by keying on row
  position + ASIN, plus a pre-flight uniqueness check.

**Added September 2026, from building the serving layer. These are the best items in this
file — make him tell each one as a story with a number at the end.**

- **`data/` was built on two machines and nothing recorded it.** Every `emb_*.npy` has an
  fp16 fingerprint in *every* value; `probs_*.npy` and `scores_*.npy` are fp32. The
  embeddings were produced on a CUDA card, the rest on an Apple GPU. Both `run_config.json`
  files stored `"device": "auto"` — the flag passed, not the device it resolved to — so
  there was no record anywhere. It was recovered by testing whether each stored float was
  exactly representable in fp16, which a genuine fp32 computation satisfies with probability
  ~5e-4 per element and these arrays satisfied at 1.0000.
  **Consequence:** re-encoding the same item on a different machine moves it ~4e-4 cosine,
  which reshuffles a 530-way top-50 enough to change fused top-10 membership on 92% of
  deferred items and costs 2.5 points of gold-leaf reachability — bounding end-to-end micro
  impact at **≤1.0%** (95% Wilson). Fixed forward: `recall.save_array` stamps resolved
  device, dtype and library versions beside every array; `src/provenance.py` audits `data/`.
  *Ask him:* why can't he just re-run everything on one machine? (He can, but it invalidates
  the candidate pools, the reranker scores and the Stage 3 answers — a full re-run of stages
  2 and 3, and the published numbers move.)

- **A prompt-cache marker that never fired.** `cache_control` was placed on the system block:
  882 tokens, under Claude Sonnet 4.5's 1,024-token minimum. Below the minimum the API
  declines to cache **and does not say so** — no error, and both cache counters return 0.
  Moving the marker to the last worked example (1,282 tokens) made it real: 68% off input
  in warm steady state, verified from the response's own `cache_read` field.
  *The generalisable point:* an optimisation that fails silently can be believed
  indefinitely. Ask him what other silent-failure modes he can name.

- **Two numbers that were measurement artefacts before they were findings.** Shortlists were
  first compared with ordered equality, which reported 100% mismatch on a path whose routing
  was provably identical — `fused` interleaves two orderings, so one swapped pair marks a
  whole row unequal. And the S1↔S2 alignment was keyed on `item_id`, which ABO duplicates;
  a dict keeps the last occurrence, so 27 of 12,200 test rows were scored against a
  different item's vector and produced a 1.08e-01 maximum cosine distance that vanished when
  alignment moved to row position.
  *Ask him:* how did he tell the artefact from the finding? (The magnitude was wrong for the
  proposed mechanism. Precision noise is ~1e-6; 1e-1 is not precision. When the number
  doesn't match the story, the measurement is the suspect.)

- **Three wrong hypotheses, killed by cheap direct measurement.** Precision (falsified by
  forcing fp16 on the serving side: gold-loss stayed at exactly 13). Text construction
  (falsified by reading the code: both paths call `recall.item_text`). Padding and pooling —
  this one *looked* certain, because Qwen3-Embedding pools the last token, which is only
  correct under left padding, and the tokenizer's default `padding_side` is `'right'`.
  Measuring the behaviour instead of reading the attribute showed the encoder is
  batch-invariant to 2.19e-04, identical at batch 2/8/16 — padding arithmetic under correct
  masking, three orders of magnitude too small. *The lesson he should state out loud:* the
  tokenizer attribute is not the behaviour.

### 9.5 Scope limits

**A serving path now exists** (§12.3) — HTTP endpoint, per-tier latency, LangSmith tracing.
What is still absent: nothing aggregates or alerts on the timings, there is no load test, no
autoscaling story, and no authentication on the endpoint. "I built a serving layer" is true;
"I ran this in production" is not, and he must not blur them.
Retrieval parity with the offline arrays is **bounded, not clean** — ≤1.0% of micro, for the
cross-device reason in §9.4.
`stage2_conformal.py` (split conformal prediction sets) is implemented and **never executed**.
ABO is a phone-case dataset with a long tail attached: a good tail stress test, a poor proxy
for a balanced production taxonomy.

---

## 10. Concepts he must be able to explain from scratch

Quiz him on these. If he cannot derive or motivate it, teach it.

**Statistics / evaluation**
- micro vs macro averaging, and why macro is the honest one under extreme imbalance
- paired vs unpaired comparison; **McNemar's test** — why only discordant pairs carry
  information, and why that matters at n=26
- selective prediction / risk–coverage; deferral recall as "of what the model gets wrong,
  what fraction escapes"
- **split conformal prediction**: the finite-sample `⌈(n+1)(1-α)⌉/n` quantile; why Mondrian
  groups must be observable at inference time (he cannot condition on stratum — that requires
  knowing the true label)
- why selecting a threshold on the test set inflates the reported number

**Retrieval / encoders**
- bi-encoder (independent encoding, cacheable, ANN-able) vs cross-encoder (joint attention,
  no caching, O(n) per query) — and the cost asymmetry that makes retrieve-then-rerank the
  standard shape
- why label documents make zero-shot possible at all
- hard-negative mining from the retriever's own top-K, and why random negatives teach the
  wrong task ("shoes vs lawnmowers" instead of "shoes vs other shoes")
- listwise (softmax-over-group) vs pointwise training
- recall@k as a *ceiling* on everything downstream

**Training mechanics**
- AdamW and decoupled weight decay (why it differs from L2 inside Adam)
- linear warmup + decay, and why the total step count must be known up front
- fp16 + GradScaler, and why bf16 is unavailable below compute capability 8.0
- class weighting schemes and what they do to a catch-all class like `OTHER`

**Agentic engineering — the most relevant to the job**
- forced tool call vs JSON mode vs free-text parsing, and what each guarantees
- why categorical confidence beats a numeric score from an LLM
- position/anchoring bias in list selection and how shuffling controls it
- abstention as a first-class output; how to *measure* abstention quality (precision and
  recall against "gold not in the shortlist")
- the **distillation trap**: if an LLM labels your training data and an LLM evaluates it, you
  are measuring agreement, not accuracy
- Batch API economics: 50% discount, `custom_id` uniqueness, prompt caching thresholds
- how to evaluate an agent *tier* — against the no-agent baseline it replaces, not in
  isolation

---

## 11. Interview questions to drill him on

Ask these cold. Push until he either produces the number or admits he does not know.

**Architecture**
1. Why four stages? What breaks if you delete Stage 2? Stage 1?
2. Your cross-encoder is worse than the retriever it was meant to improve. Why is it still in
   the system?
3. Why is the retriever frozen? Give me the reason that isn't "6GB of VRAM."
4. Walk me through what happens to a goat cheese listing whose category has zero training
   examples.
5. In your final table Stage 2's share is 0%. Is it dead code?

**Metrics**
6. Your micro is 0.895 and your macro is 0.647. Which do I care about and why?
7. S1 is at 0.977 micro and 0.394 macro on the same items. Explain how both are true.
8. How did you pick τ = 0.900? What would have gone wrong if you'd picked it differently?
9. What is deferral recall and why is it the gate rather than accuracy?
10. A linear SVM beats you on head, torso and tail. Defend the project.

**Statistics**
11. Why McNemar and not a two-sample t-test?
12. You reported a −0.106 regression that turned into +0.154. What changed, and what should
    you have done the first time?
13. Your conformal module has never been run. What would it give you that τ doesn't?
14. Your zero-shot stratum is constructed by deleting training rows. Isn't that cheating?

**Agentic engineering**
15. Why a forced tool call instead of asking for JSON?
16. You shuffle the candidates. What does that cost you, and how do you know?
17. Your agent abstains 7.6% of the time at 31.4% precision. Is that good? What would you do?
18. How do you know the LLM isn't just memorising Amazon's taxonomy from pretraining?
19. How would you stop this system from getting more expensive as the catalogue grows?
20. What's the failure mode if the shortlist is wrong but confident?

**Engineering judgment**
21. What's the most expensive mistake you made, and how did you catch it?
22. If you had two more weeks, what would you do — and what makes that the top of the list?
23. What would you cut from this system if you had to ship it Monday?

---

## 12. Mapping to the job requirements: LLM, LangChain, prompt engineering

### 12.1 Hands-on LLM experience — strong, and concrete

Everything here is a specific artefact, not a claim:

| what he did | why an interviewer cares |
|---|---|
| Designed and shipped an **agent tier** inside a larger ML system, not a chatbot demo | shows LLMs used as a component with an interface and a measured contract |
| **Forced tool call** (`tool_choice: {type:"tool", name:"record_category"}`) with a typed schema | 0 hallucinated leaf ids in 2,015 answers — a hard guarantee, not a hope |
| **Batch API** orchestration: submit, poll `processing_status`, stream results, join on `custom_id` | production cost engineering — 50% discount, 2,015 items for ~$2.70 |
| Hit and fixed a real batch bug: **duplicate `custom_id`s** rejected the whole submission | shows he has actually run this at scale, not just called the API once |
| **Abstention** as a first-class output with measured precision and recall | most people never measure refusal at all |
| **Ablation harness** (`--ablate rank-order`, `--ablate no-fewshot`, `--dry-run`) | LLM behaviour treated as something to test, not to vibe-check |
| **Paired McNemar** of agent vs the no-agent baseline it replaces | he can prove the LLM tier earns its cost |
| Named and avoided the **distillation trap** | if an LLM labels your data and an LLM evaluates it, you measure agreement, not accuracy |

The single best line for this requirement:

> I didn't just call an LLM — I measured whether the LLM tier was worth its cost, per
> frequency stratum, with a paired significance test against the system that would exist
> without it. It's worth +0.074 overall, p<0.0001, and it's significantly *negative* on head.

### 12.2 Prompt engineering — the deepest part of the project

Every one of these was a decision with a reason, and several were tested:

- **Label documents as the core artefact.** A leaf named `ACCESSORY` is unfindable; the same
  leaf carrying its ancestor path and three real product titles is findable — including at
  zero training examples. Four cumulative variants, measured: name 0.837 → +path 0.945 →
  +prototypes 0.977 macro recall@50. **Prompt engineering as a measurable intervention.**
- **Truncation that protects the expensive part.** `v4_desc` is built as
  `<v3_proto> <description>`, so a naive `[:limit]` deletes the generated description first —
  exactly the part that cost money. `compose_doc` re-orders to put the description straight
  after the path and lets the example titles absorb the cut.
- **Few-shot including a worked abstention.** Without an abstention example the model almost
  never refuses; with one it refuses 7.6% of the time against ~9.4% unanswerable.
- **Candidate shuffling** to control position/anchoring bias, with `--ablate rank-order` as
  the A/B that tests what shuffling costs.
- **Categorical confidence enum** rather than a numeric score — LLMs are poorly calibrated on
  numbers and can only really distinguish ordinal buckets.
- **No rationale tokens.** Output is `{leaf_id, confidence}`; rationale roughly triples output
  cost for no measured gain.
- **Instruction on the query side only.** Qwen3-Embedding is instruction-aware and the task is
  asymmetric — items get `Instruct: … \nQuery: `, label documents are embedded bare.
- **`--dry-run`** prints one complete assembled prompt and spends nothing. He should mention
  this: it is the habit of someone who has wasted money on a bad batch before.

### 12.3 LangChain — he has now shipped it, and the interesting part is what he checked

**Revised September 2026. Earlier versions of this file said he had not used LangChain.
That is no longer true and he should not use the old humble answer.**

What he built, in a week, on top of the existing system rather than instead of it:

| artefact | what it demonstrates |
|---|---|
| The cascade as a **LangGraph `StateGraph`** — nodes `s1 → retrieve → rerank → stage3`, conditional edges on the deferral gate | the routing that was implicit in a script is now an explicit, inspectable graph |
| **Nothing that produces a number was reimplemented** — every node calls the same `Cascade` methods the offline evaluation calls | a serving layer that re-implements the model path is one whose numbers you must re-earn |
| Stage 3 as a **swappable provider**: `anthropic` (LangChain) / `raw` (SDK) / `cached` / `ollama` / `none` | the framework is a boundary he chose, not one he inherited |
| **FastAPI** endpoint, weights loaded once in the lifespan handler, per-tier timings in every response | he knows why loading a model per request is the classic demo-latency lie |
| **46 assertions that need no weights**, covering routing, abstention, and the HTTP contract | the new logic is routing, so that is what is tested |

**The single best thing he did here, and the thing to lead with:**

> LangChain and the raw SDK encode the same conversation differently — LangChain wraps a
> tool result in a `ToolMessage`, the Anthropic SDK puts it in a user turn. The rendered text
> is identical because both import from the same module, but the message structure isn't, and
> that's a difference the model can in principle see. So I ran the same items down both paths
> and compared. 100% agreement. Now I can say the serving layer didn't change the model's
> input, instead of assuming it.

That is `--parity`, and it is a better answer than any amount of LCEL fluency. Most
candidates adopt a framework and trust it. He adopted one and built the control arm.

**Framework-boundary judgement he can now defend with specifics:**

- LangGraph earned its place: the cascade *is* conditional routing with shared state, which
  is exactly `StateGraph` with conditional edges. Tiers 1 and 2 are model calls, not agents.
- Most of LangChain proper did **not**. He deliberately did not wrap his retrieval in
  LangChain's vector-store or retriever abstractions — he already had a measured retrieval
  path, and the abstraction would have added indirection over code whose numbers were known.
- What the framework bought him concretely: `init_chat_model` / `bind_tools` made the LLM
  tier a config flag across Anthropic and local Ollama, and LangGraph gave him node-level
  tracing for free. That is the honest accounting — integrations and observability, which is
  the standard answer, but he can point at the two places it actually paid.

**Questions to drill him on:**

1. Why did you not use `create_agent`? (Tier 3 is one forced tool call with a closed
   candidate set — there is no loop, no tool selection, nothing to decide. An agent executor
   would add a control loop that has nothing to control.)
2. Your offline Stage 3 uses the Batch API. What breaks when you serve it? (Batch latency is
   hours. A serving path needs a synchronous call — which changes both cost per item and the
   latency profile, and it lands entirely on the ~18% of requests that escalate.)
3. What does your graph do when Stage 3 abstains? (Ends with **no label**, `stage="S3-abstain"`
   — never a silent fallback to S2's top-1, which would inflate every served accuracy number
   relative to the offline report while looking like an improvement. This is asserted in the
   tests.)
4. How do you know the served path is the same classifier? (`verify.py`: 0 defer-decision
   flips in 400 items, S1 p_max to 2.4e-06 — and the retrieval divergence he *did* find, with
   its bound. See §9.4.)

### 12.4 LangSmith — what he wired, and what it is and is not good for

LangSmith is LangChain's hosted observability and evaluation platform. It is a separate
product from the library, instrumented through `langchain-core`'s callback system, so
enabling it is environment variables and no code change.

**The three things it does**

1. **Tracing.** Every chain, graph node and model call becomes a span in a nested tree, with
   inputs, outputs, token counts, latency and errors. For his cascade a single request renders
   as `s1 → retrieve → rerank → stage3`, so the routing is a picture rather than a log line.
2. **Datasets and evaluation.** Save examples, run a chain over them, score with graders
   including LLM-as-judge. He should notice this is a hosted version of what
   `stage3_agent.py --ablate` already does by hand — and say so, because his version carries
   judgement the generic tooling does not: candidate shuffling to defeat position anchoring,
   and abstention scored against genuine retrieval misses rather than against a gold label.
3. **Prompt management.** Versioned prompts and a playground, so prompt edits are not only in
   source control.

**What he actually wired, and what it showed**

- `LANGSMITH_TRACING` + key + project, plus **`LANGSMITH_ENDPOINT`**, and that last one is the
  operational story worth telling. His account is in the **APAC** region. LangSmith regions are
  fully isolated: a regional key posting to the default US ingest endpoint fails `403 Forbidden`
  on `/runs/multipart`, and the US dashboard shows an empty account even after ingestion works.
  He diagnosed it by probing all three regions for an HTTP status, then confirmed via
  `/api/v1/sessions?include_stats=true` that the project existed with **206 runs** — proving the
  data was present and the problem was navigation, not ingestion.
  *Why this is worth telling:* it separates "the write failed" from "I'm looking at the wrong
  page," which are the same root cause wearing two faces, and he found it with an API call
  rather than by guessing.
- The tracer **fails loudly and the program continues** — six error lines per run and valid
  results. He should contrast that with the prompt-cache marker in §9.4, which failed
  *silently* and was believed for an afternoon. Loud failure is the better bug, and he now has
  one clean example of each.

**What LangSmith cannot show him, and he should say so unprompted**

- Provider `raw` calls the Anthropic SDK directly, outside LangChain, so **half of every parity
  pair is invisible** in the UI. The check that proves his serving layer is faithful is the one
  thing the dashboard cannot display.
- `s1`, `retrieve` and `rerank` carry no token counts — they are PyTorch forward passes, not
  model API calls. Only `stage3` has usage data.
- The cache and token numbers he quotes come from the **API response body**, which is ground
  truth. LangSmith renders the same fields with a UI and history on top. Useful, not more
  authoritative. Given that two of his findings came from measuring the response rather than
  trusting a dashboard-shaped abstraction, this distinction is one he has earned the right to
  make.

**Questions to drill him on:**

1. Your prompts go to LangChain's servers when tracing is on. When is that not acceptable?
   (ABO is public so it is fine here; customer listings, PII, or anything under a DPA are not.
   He should also note he deliberately did *not* enable it for a full 12,200-item run.)
2. What is the difference between a trace and a metric, and which one do you need at 3am?
   (Traces debug one request; metrics tell you something is wrong across many. He has
   per-request traces and per-tier latency but nothing aggregating or alerting — §9.5.)
3. Where would LangSmith actually have saved you time on this project? (Honestly: the τ>0.9
   run, as one place to see which items were slow, which abstained, and total spend, without
   writing the aggregation. Not the parity or cache work, which the response body answered
   better.)

### 12.5 Serving this system from Django — the answer for a Django shop

He built the endpoint on FastAPI. A Korean backend team is as likely to be on Django, and
"why not Django?" is a fair question with a real technical answer. **The answer is not
preference. It is process model.**

**The core problem: Django's default deployment shape multiplies his memory by the worker
count.**

FastAPI's lifespan handler loads the models once per process, and a uvicorn deployment is
typically one process. Django's equivalent hook is `AppConfig.ready()` in `apps.py` — but
`ready()` runs **once per worker process**, and a conventional Gunicorn/WSGI deployment runs
several sync workers. His resident set is roughly 3GB: DistilBERT (~265MB fp32), the MiniLM
cross-encoder (~90MB), Qwen3-Embedding-0.6B (~2.4GB fp32), plus the label embedding matrix.

Size it with his own numbers. Mean end-to-end service time is 409ms. At 10 requests/sec the
offered load is 10 × 0.409 ≈ 4.1 workers busy on average; to keep the tail from exploding you
want utilisation well under 1, so call it **6 sync workers**. That is **~18GB of model weights
for one service**, six copies of the same frozen tensors.

Two follow-ups he should raise before the interviewer does:

- `ready()` also fires during `manage.py migrate` and `collectstatic`, so an unguarded model
  load there makes every management command pay 10.8 seconds and pull 3GB. Guard it.
- Gunicorn's `--preload` loads the app *before* forking, which looks like the fix — it is not
  a safe one here. CUDA and MPS contexts do not survive `fork()` reliably, and PyTorch after
  fork is a known hazard. Copy-on-write also stops helping the moment refcounts touch the
  pages.

**The second problem: one tier blocks for 1.8 seconds.**

His measured distribution is p50 25ms, p95 2,182ms — an 87× spread, because 82% of requests
stop at S1 and 18% make a synchronous LLM call averaging 1,797ms. Under sync WSGI workers,
each of those occupies a whole worker for two seconds, and a small pool is head-of-line
blocked by its own slow tail. Django async views (3.1+, ASGI) help only the *I/O* part: the
Anthropic call can await, but the PyTorch forward passes are compute-bound and will block the
event loop unless pushed through `sync_to_async(thread_sensitive=False)` into a thread pool —
at which point torch does release the GIL during ops, so threads overlap usefully, while the
Python-level pre/post-processing does not.

**What he should actually propose.** Not "port it to Django" — separate the concerns:

> Django owns HTTP, auth, persistence and the admin. The models live in their own process,
> either the FastAPI service I already built, called over localhost, or a Celery worker pool
> holding the weights. The service boundary goes at the model, not inside it, because the
> thing Django is good at and the thing a 3GB model is good at have opposite process
> requirements — many cheap workers versus one expensive one.

Then the trade-off, which is the part that shows judgement: an in-process call is lowest
latency but couples the deployments and forces one memory profile on both; a Celery hop
decouples the 2-second tier and lets it scale independently, at the cost of queue latency and
a much harder end-to-end trace. Given 82% of his requests finish in 25ms, he would keep S1 in
the request path and only offload the escalated tier — which is exactly what τ already
partitions for him. **The gate he built for cost turns out to be the same gate you would draw
for deployment.** That is the sentence to end on.

**Where LangSmith attaches in Django, including the gotcha.** Tracing context is per-request:
a small middleware opens a `tracing_context` and threads the Django request id in as metadata,
so a trace can be tied back to an access log line. The gotcha is Celery — trace context does
**not** cross a process boundary automatically. The parent run id has to travel in the task
payload and be re-attached inside the worker, or the LLM call shows up as an orphan trace with
no route back to the request that caused it.

**Questions to drill him on:**

1. Why is `AppConfig.ready()` not equivalent to FastAPI's lifespan? (Per worker, not per
   service; and it runs for management commands.)
2. You have 6 Gunicorn workers and one GPU. What happens? (Six processes contending for one
   device, each with its own context and memory; on a 6GB card they will not all fit. Another
   argument for a single model process.)
3. Your p95 is 2.2 seconds and your p50 is 25ms. What does that do to a thread-per-request
   server? (Head-of-line blocking; the 18% dictates pool sizing for the 82%.)
4. If you moved Stage 3 to Celery, what would you lose? (End-to-end tracing, unless you
   propagate context by hand; plus queue latency added to a tier already at 1.8s.)

### 12.6 What to build next if he wants a second, more agentic project

His current system has *one* LLM call per item — powerful but architecturally simple. If he
has time before interviews — and note the serving layer has already answered part of
this, since the graph makes the control flow explicit — the highest-leverage addition is a
**multi-step** agent: let it
request more information (fetch the full listing, retrieve sibling categories, ask for a
deeper shortlist) before committing. That converts "I used an LLM" into "I designed an agent
loop with tools, state and a stopping criterion," which is what an agentic-engineering role
is actually hiring for. The Korean arm is the natural vehicle.

---

## 13. Honest framing of what this project supports

He has been explicit about this himself, and the framing matters. The claim is **not** "I
built a state-of-the-art classifier."

The claim is:

> I can fine-tune encoders and engineer an agent tier into a system that handles a long-tail
> label space, and I can tell you which parts of it earned their place — with the measurement
> for each, including the ones that failed.

Every tier has a documented failure attached: the cross-encoder fails its own gate alone,
`OTHER` cannot see novel leaves, the agent is significantly worse than its own shortlist on
head, and the whole cascade loses to n-grams on four strata out of five. It wins overall macro
by 0.140 anyway, because the failures do not overlap and because one stratum is a regime the
baseline cannot enter at all.

**Do not let him claim** production-readiness, a Korean-language system (that is a separate
project, not started), a human-labelled gold set (the evaluation uses ABO's own labels), or
that the cascade beats the baseline outright (it does not — it wins macro and loses micro).

If asked directly how the code was written, the truthful answer is that he designed the
architecture, specified every stage, set the decision criteria, and directed and reviewed the
implementation with AI assistance — and that he can explain and defend every measurement in
it. That last part is what this file is for, and it is the part that is actually tested.

---

## 14. Repository

```
src/prepare.py           ABO → items.parquet, splits, strata, zero-shot construction
src/label_docs.py        4 cumulative label-document variants
src/recall.py            retrieval gate — recall@k per variant per stratum
src/baselines.py         majority + TF-IDF floor
src/stage1.py            S1 training, deferral table, funnel; --from-probs re-reports free
src/stage2_rerank.py     negative mining, listwise training, recall@m gate
src/stage2_conformal.py  split conformal prediction sets   ← never executed
src/stage3_agent.py      Batch API agent tier, ablations, --dry-run
src/shortlist.py         the one shortlist implementation, shared by both callers
src/pipeline.py          Cascade object + end-to-end evaluation with stage attribution
src/test_pipeline.py     21 assertions, no model required

  added Sept 2026 — the serving layer
src/sweep.py             tau frontier replayed from cached Stage 3 answers — costs nothing
src/provenance.py        which device and precision produced each array in data/
src/serve/graph.py       the cascade as a LangGraph state graph over the same Cascade object
src/serve/llm.py         Stage 3 swappable: anthropic | raw | cached | ollama | none
src/serve/app.py         FastAPI endpoint, weights loaded once at startup
src/serve/verify.py      served path vs the offline arrays
src/serve/bench.py       per-tier latency, cold start excluded and reported separately
src/serve/diagnose_embedding.py   padding, batch-invariance and cache-provenance probes
src/serve/test_graph.py  26 routing assertions, no weights required
src/serve/test_app.py    20 endpoint assertions, no weights required
docs/serving-parity.md   the full parity write-up, including what was measured wrongly first
```

ABO is CC BY-NC 4.0 — non-commercial, attribution required, corpus never redistributed.
