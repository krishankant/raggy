# Evaluating RAG: the metrics, worked by hand

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [12 Evaluating RAG](../concepts/12-evaluation.md).

This is the reference for week 2. The lesson (`lessons/week02-evals.md`) explains *why* we measure. This page explains *what each number means*, with one example small enough to check on paper. Every number below was printed by `raggy/evals/metrics.py`.

A RAG pipeline can fail in two places, so there are two families of metrics:

| Stage | Question | Metrics | Needs |
|---|---|---|---|
| Retrieval | Did the right evidence reach the model? | recall@k, MRR, nDCG@k | gold evidence per question |
| Generation | Did the model use it well? | correctness, faithfulness, citation precision, refusal accuracy | reference answers + a judge |

Measure retrieval first: if the evidence never reaches the model, no prompt can fix the answer.

---

## Gold evidence as spans, not chunk ids

Each question lists the evidence that answers it as a **verbatim quote** from the corpus:

```json
{"id": "f01", "question": "How long do refunds take to arrive?",
 "gold": [{"source": "returns-and-refunds.md",
           "quote": "Refunds are paid back to the original payment method within 14 days of the return arriving at our warehouse"}]}
```

At eval time the quote is located as a character span (`start`, `end`). A retrieved chunk is relevant if its own span overlaps the gold span. Chunks already carry `start`/`end` offsets from week 1, so this check is one line (`raggy/evals/dataset.py: chunk_gold`).

Why not label chunk ids? Because chunk ids are an output of the chunker. Week 4 rewrites the chunker, and every id-based label would silently break. A quote stays true whatever the chunk boundaries are.

A quote that can't be found raises `GoldNotFoundError` naming the question. If it were skipped silently, a typo in the dataset would show up as "recall 0", and you would go hunting for a retrieval bug that doesn't exist.

## The worked example

One question has **two** gold spans (g0 and g1). Retrieval returns four chunks:

| rank | chunk covers | new gold? |
|---|---|---|
| 1 | nothing | — |
| 2 | g0 | yes (g0) |
| 3 | g0 again | no: already seen |
| 4 | g1 | yes (g1) |

In code that's `retrieval_metrics([set(), {0}, {0}, {1}], n_gold=2, ks=(1, 3, 4))`:

```
{'recall@1': 0.0, 'ndcg@1': 0.0, 'recall@3': 0.5, 'ndcg@3': 0.3868..., 'recall@4': 1.0, 'ndcg@4': 0.6509..., 'mrr': 0.5}
```

### recall@k: did we find the evidence at all?

> recall@k = (gold items covered by the top k) / (number of gold items)

- top 1 covers nothing → **0.0**
- top 3 covers g0 → 1/2 = **0.5**
- top 4 covers g0 and g1 → **1.0**

The answer model only sees the top `RAGGY_TOP_K` chunks (4 by default), so **recall@4 is the ceiling on answer quality**. Evidence at rank 5 might as well not exist.

### MRR: how high is the first useful result?

> reciprocal rank = 1 / (rank of the first chunk covering any gold); MRR is its mean over questions

The first useful chunk is at rank 2, so the reciprocal rank is **1/2 = 0.5**. MRR cares only about the *first* hit. That is the right lens when one passage is enough, and the wrong one when an answer needs two passages (use recall for those).

### nDCG@k: are the useful results near the top?

Each position is worth less than the one above it. A useful chunk at rank *r* earns `1 / log2(r + 1)`:

| rank | discount |
|---|---|
| 1 | 1.000 |
| 2 | 0.631 |
| 3 | 0.500 |
| 4 | 0.431 |

> DCG@k = sum of discounts at ranks that bring a *new* gold item
> IDCG@k = the best possible DCG: one new gold item at each of the top min(n_gold, k) ranks
> nDCG@k = DCG@k / IDCG@k

For k = 4: DCG = 0.631 (rank 2) + 0.431 (rank 4) = 1.062. IDCG = 1 + 0.631 = 1.631. nDCG@4 = **0.651**.

For k = 3: DCG = 0.631. IDCG = 1.631 (two gold items fit in three slots). nDCG@3 = **0.387**.

Rank 3 earns nothing even though it is "relevant", because it repeats g0. raggy counts each gold item once on purpose. Three copies of the same paragraph waste three of the model's four context slots, and a metric that rewarded that would reward a defect. Week 8's context packing deduplicates for the same reason.

### Chunk-level vs document-level scoring

The fictional suite scores **chunks against spans**, because "the right paragraph" matters there.

SciFact's labels are per **document** (an abstract is relevant or not). raggy retrieves chunks, keeps each document's best chunk (`collapse_to_documents`), and scores the resulting document ranking. That matches how BEIR and MTEB score SciFact, so our number can be checked against the published one. A harness that can't reproduce a known result can't be trusted on an unknown one.

---

## Answer metrics

### Refusal accuracy (no judge needed)

The prompt tells the model to reply with one fixed sentence, `I couldn't find this in the documents.`, when the sources lack the answer. `is_refusal` checks for that sentence anywhere in the reply, ignoring case and curly apostrophes. Three numbers come out:

- `unanswerable_refused`: how often it refused when it should. Higher is better. A low value means it is making things up.
- `answerable_refused`: how often it refused when the answer *was* there. Lower is better. A high value means it is over-cautious or retrieval failed.
- `refusal_accuracy`: both combined, over every question.

The two kinds of error trade against each other. A prompt that says "refuse if unsure" pushes the first number up *and* the second number up. You need both numbers to see that.

### Correctness (judge + reference answer)

The judge sees the question, the expert's reference answer and the candidate answer, and replies `correct` (1), `partial` (0.5) or `incorrect` (0). It doesn't see the sources. Correctness asks "is this the right answer?", not "is it grounded?". An answerable question that the model refused scores 0 without calling the judge.

### Faithfulness and citation precision (judge + sources)

A second call shows the judge the numbered sources the model saw, plus the answer:

- **faithful**: is every claim stated in the sources? An answer can be *correct and unfaithful*, when the model knew the fact from training. Inside a company that's a defect, because next time its "knowledge" will be stale or invented.
- **citation precision**: of the `[n]` markers the answer used, the fraction where source *n* really supports the sentence. A marker the judge didn't mention counts as unsupported.

### Judge prompts are versioned

Rewording a judge prompt can move every score, so `JUDGE_VERSION` ("judge-v1") is saved in each run, and `raggy compare` warns when two runs used different versions. Treat a judge change like a change to a ruler: re-measure the baseline with the new ruler before you compare.

---

## Latency: p50 and p95, never the average

Ten requests (ms): `120, 95, 101, 98, 2300, 105, 99, 97, 110, 102`. The mean is 323 ms, which describes none of them. raggy's `percentile` gives:

- **p50 = 101.5 ms**: the typical request
- **p95 = 1319 ms**: the slow tail (here, the one 2.3 s outlier)

Users remember the slow tail. Retrieval and generation are timed separately, because they're fixed by different weeks (5 and 12 for retrieval speed, 8 and 12 for generation).

## Small suites: how big a change is real?

The fictional suite has 33 answerable questions, so one question is worth 0.03 of recall. A change of +0.03 is a single question flipping. Before believing a gain, open the run JSON (`questions[].retrieved`) and look at *which* questions changed. SciFact has 300 questions, so one question there is 0.003, but its questions are harder to read by eye. Use both suites: the small one to understand the change, the large one to confirm it.
