# RAG metrics, explained with examples

**[← Docs hub](README.md)** · **[Evaluating RAG (week 2 guide)](evaluating-rag.md)**

Every `raggy eval` run prints a block of numbers: recall@1, recall@4, MRR, nDCG, correctness, faithfulness, citation precision, refusal rates, p50 and p95. This guide explains each one with small worked examples you can check by hand: what it measures, how it's computed, how to read it, and what to do when it's low.

The metrics come in three families:

| Family | Question it answers | Needs a model to grade? |
| --- | --- | --- |
| [Retrieval](#part-1-retrieval-metrics) | Did search find the passages that contain the answer, and rank them high? | No: exact and repeatable |
| [Answer](#part-2-answer-metrics) | Is the answer right, honest, properly cited, and refused when it should be? | Yes: an LLM judge |
| [Cost and speed](#part-3-cost-and-speed-metrics) | How long did it take, and how much did the model read? | No |

---

## The setup: questions with gold evidence

Every eval question carries **gold evidence**: the exact passage, or passages, that contain its answer. In raggy, gold is a *quote* from a document, which becomes a character span. A retrieved chunk "hits" the gold if its text covers that span.

One running example:

> **Question:** "How long do refunds take?"
> **Gold:** the sentence *"Refunds are paid within 14 days of the return arriving at our warehouse."* in `returns-and-refunds.md`

raggy searches and returns a ranked list:

| Rank | Retrieved chunk | Covers the gold? |
| --- | --- | --- |
| 1 | `shipping.md`: "Orders ship within 2 working days…" | ✗ |
| 2 | `returns-and-refunds.md`: "Refunds are paid within 14 days…" | ✓ |
| 3 | `warranty.md`: "The frame is covered for 10 years…" | ✗ |
| 4 | `returns-and-refunds.md`: "To start a return, fill in the form…" | ✗ |

Every retrieval metric is a different way of scoring this one list.

---

## Part 1: Retrieval metrics

### 1.1 recall@k: "Is the answer in the top k?"

```text
recall@k = (gold passages found in the top k) ÷ (gold passages the question has)
```

**Example A: the running example (one gold passage, at rank 2)**

| k | Top k contains the gold? | recall@k |
| --- | --- | --- |
| 1 | no (rank 1 is shipping) | 0 ÷ 1 = **0.0** |
| 4 | yes (rank 2) | 1 ÷ 1 = **1.0** |
| 10 | yes | **1.0** |

**Example B: a question that needs two passages**

> "What's the Ebb's battery warranty, and who replaces it?"
> Gold 1: `warranty.md` "The battery is covered for 2 years." Gold 2: `service.md` "Battery replacements are done at the Fenwick workshop."

Search finds gold 1 at rank 1, and gold 2 only at rank 7.

| k | Gold found | recall@k |
| --- | --- | --- |
| 1 | gold 1 | 1 ÷ 2 = **0.5** |
| 4 | gold 1 | 1 ÷ 2 = **0.5** |
| 10 | both | 2 ÷ 2 = **1.0** |

With k = 4, the answer model sees only half of what it needs, so its answer will be half right at best.

**Example C: a suite of five questions.** The suite's recall@k is the average over questions:

| Question | First gold at rank | recall@1 | recall@4 |
| --- | --- | --- | --- |
| Q1 | 1 | 1 | 1 |
| Q2 | 3 | 0 | 1 |
| Q3 | 2 | 0 | 1 |
| Q4 | not in top 10 | 0 | 0 |
| Q5 | 1 | 1 | 1 |
| **Suite** | | **2 ÷ 5 = 0.40** | **4 ÷ 5 = 0.80** |

**How to read it:**

- **recall@k at the k your pipeline uses** (raggy's `top_k` is 4) is a ceiling on answer quality. A passage outside the top k never reaches the model.
- **The gap between recall@1 and recall@4** tells you about ordering. Week 7 measured recall@4 = 0.97 but recall@1 = 0.85 on the fictional suite: the answer was almost always found, but often not first. A reranker closes exactly this gap (recall@1 went to 0.97).
- **recall@100** measures reach, not order. A reranker can only reorder what the retriever found, so if recall@100 is low, fix retrieval or chunking first.

**Two details:**

- Unanswerable questions have no gold, so they're left out of recall.
- Two retrieved chunks that cover the *same* gold span count once. Finding the same passage twice doesn't find more of the answer.

### 1.2 MRR (mean reciprocal rank): "How high is the first right passage?"

For each question, take one divided by the rank of the **first** chunk that covers gold, or 0 if none does. Then average.

```text
reciprocal rank = 1 ÷ (rank of the first gold hit)
```

| First gold hit at | Reciprocal rank |
| --- | --- |
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 4 | 0.250 |
| 10 | 0.100 |
| not found | 0 |

**Example: the same five questions as above**

| Question | First gold at | Reciprocal rank |
| --- | --- | --- |
| Q1 | 1 | 1.000 |
| Q2 | 3 | 0.333 |
| Q3 | 2 | 0.500 |
| Q4 | — | 0 |
| Q5 | 1 | 1.000 |
| **MRR** | | (1 + 0.333 + 0.5 + 0 + 1) ÷ 5 = **0.567** |

**Why MRR exists alongside recall@4:** these two systems have the same recall@4, but different MRR.

| | Q1 rank | Q2 rank | Q3 rank | recall@4 | MRR |
| --- | --- | --- | --- | --- | --- |
| System A | 1 | 1 | 1 | 1.0 | **1.00** |
| System B | 4 | 4 | 4 | 1.0 | **0.25** |

Both "found everything". But System B always puts the answer last of four, where models use it least (the "lost in the middle" effect, week 8). MRR tells them apart; recall@4 can't.

**Limit:** MRR looks only at the *first* gold hit. For a question with two gold passages, it ignores where the second one is. That's what nDCG adds.

### 1.3 nDCG@k: "How good is the whole ranking?"

nDCG (normalised discounted cumulative gain) gives credit for every relevant chunk in the top k, with less credit the lower it ranks. It then divides by the best score possible, so the result is between 0 and 1.

**Step 1: the discount for each rank.** A relevant chunk at rank *i* earns 1 ÷ log₂(i + 1):

| Rank | log₂(rank + 1) | Credit |
| --- | --- | --- |
| 1 | 1.000 | **1.000** |
| 2 | 1.585 | **0.631** |
| 3 | 2.000 | **0.500** |
| 4 | 2.322 | **0.431** |
| 5 | 2.585 | **0.387** |
| 10 | 3.459 | **0.289** |

**Step 2: DCG** is the sum of the credits of the relevant chunks in the top k.

**Step 3: IDCG** (the ideal DCG) is the DCG of the best possible ranking, with every relevant chunk at the top.

**Step 4: nDCG = DCG ÷ IDCG.**

**Example A: the running example.** One gold passage, found at rank 2, with k = 4.

```text
DCG   = 0.631                 (gold at rank 2)
IDCG  = 1.000                 (best case: gold at rank 1)
nDCG@4 = 0.631 ÷ 1.000 = 0.631
```

With a single gold passage, nDCG is just the discount of the rank it was found at.

**Example B: two gold passages, different placements.** IDCG is the same for all of these: both gold at ranks 1 and 2 = 1.000 + 0.631 = **1.631**.

| Gold found at ranks | DCG | nDCG@4 | recall@4 |
| --- | --- | --- | --- |
| 1 and 2 | 1.000 + 0.631 = 1.631 | **1.000** | 1.0 |
| 1 and 4 | 1.000 + 0.431 = 1.431 | **0.877** | 1.0 |
| 2 and 4 | 0.631 + 0.431 = 1.062 | **0.651** | 1.0 |
| 3 and 4 | 0.500 + 0.431 = 0.931 | **0.571** | 1.0 |
| 1 only (other at rank 9) | 1.000 | **0.613** | 0.5 |

All of the first four rows have recall@4 = 1.0, yet nDCG spreads them from 0.57 to 1.00. It sees the whole ranking.

**Example C: graded relevance.** Public benchmarks sometimes grade relevance (0 = not relevant, 1 = partly, 2 = fully). The credit becomes relevance ÷ log₂(i + 1).

Retrieved relevance grades by rank: [1, 2, 0, 0]

```text
DCG   = 1/1.000 + 2/1.585 = 1.000 + 1.262 = 2.262
ideal = [2, 1, 0, 0] → 2/1.000 + 1/1.585 = 2.000 + 0.631 = 2.631
nDCG@4 = 2.262 ÷ 2.631 = 0.860
```

The fully relevant passage was second, so the score is below 1.

**Where you'll see it:** SciFact and FiQA report **nDCG@10**, because that's what the BEIR benchmark publishes. raggy's SciFact score of 0.712 can be compared directly with published papers.

### 1.4 precision@k (not reported by raggy, but worth knowing)

```text
precision@k = (relevant chunks in the top k) ÷ k
```

In the running example, precision@4 = 1 ÷ 4 = 0.25. With one gold passage per question, precision@4 can never exceed 0.25, so it says little that recall@4 doesn't. That's why raggy doesn't report it. It matters when questions have many relevant passages, and you care how much of the context is noise.

### 1.5 Chunk scoring against document scoring

- **Chunk-scored suites** (fictional, fictional-web, manual, rich, images, media): a hit must cover the gold *span*. Retrieving the right document but the wrong part of it doesn't count. This is strict, and it's what matters for RAG: the model only sees the chunk.
- **Document-scored suites** (SciFact, FiQA): a hit is any chunk from a relevant *document*, because the benchmark's labels are per document. Two chunks of one document count once.

### 1.6 Diagnosing with retrieval metrics

| What you see | What it means | Try |
| --- | --- | --- |
| Low recall@100 | The retriever never finds it | Ingestion (is the text there?), chunking, hybrid search (week 6), another embedding model |
| High recall@10, low recall@4 | Found, but ranked too low to be used | Reranking (week 7), a larger `top_k` |
| High recall@4, low recall@1 / MRR | In the context, but not first | Reranking (week 7) |
| One question type low (by tag) | A blind spot: codes, paraphrases, tables, pictures | Keyword search for codes (week 6), layout loaders for tables (week 9), images (week 10) |

---

## Part 2: Answer metrics

These grade the final answer. raggy generates the answer with the configured model, then asks a **judge** model two separate questions: is it *correct* (against the reference answer), and is it *grounded* (against the sources)?

### 2.1 Correctness: "Is the answer right?"

The judge compares the answer with a reference answer written by a person:

| Verdict | Score | Meaning |
| --- | --- | --- |
| correct | 1.0 | Every key fact of the reference, nothing contradicting it |
| partial | 0.5 | Right as far as it goes, but missing a key fact |
| incorrect | 0.0 | A wrong key fact, a contradiction, or a refusal |

**Examples.** Question "How long do refunds take?", reference "Within 14 days of the return arriving."

| Answer | Verdict | Why |
| --- | --- | --- |
| "Refunds are paid within 14 days of the return arriving [1]." | correct (1.0) | Same facts |
| "You'll get your money back within two weeks." | correct (1.0) | Different words, same meaning |
| "Within 14 days, paid to your original card [1]." | correct (1.0) | An extra detail that doesn't contradict is fine |
| "Refunds are paid after the return arrives." | partial (0.5) | Right, but missing the key fact (14 days) |
| "Refunds take about a month." | incorrect (0.0) | Wrong key fact |
| "I couldn't find this in the documents." | incorrect (0.0) | Refused an answerable question |

The suite score is the average. Five answers scoring 1, 1, 0.5, 0 and 1 give (1 + 1 + 0.5 + 0 + 1) ÷ 5 = **0.70**.

**A trap with references** (found in week 9): the reference must hold *exactly what the question asks*. If the question is "What does a 7-bike order cost per bike?" and the reference says "4,700 credits, with a 4-week lead time", a perfect answer ("4,700 credits") is marked partial for "missing" the lead time.

### 2.2 Faithfulness: "Did every claim come from the sources?"

The judge sees the retrieved passages, not the reference, and answers true or false: is every factual claim stated in, or directly implied by, the sources?

**Example.** The only source says *"Refunds are paid within 14 days of the return arriving."*

| Answer | Faithful? | Why |
| --- | --- | --- |
| "Refunds take 14 days from when your return arrives." | ✓ | All in the source |
| "Refunds take 14 days, and you'll get a confirmation email." | ✗ | The email isn't in the source, even if it's true in real life |
| "Refunds take 14 days. Most shops take 30." | ✗ | Outside knowledge |

**Correct and faithful aren't the same thing:**

| | Faithful | Unfaithful |
| --- | --- | --- |
| **Correct** | The goal | The model knew it from training, not from your documents. Right this time, but it can't be trusted next time |
| **Incorrect** | The sources were wrong or incomplete: a retrieval or data problem | The model made something up |

The suite score is the share of answers judged faithful.

### 2.3 Citation precision: "Does each [n] point at a source that supports it?"

```text
citation precision = (citations whose source supports their sentence) ÷ (all citations used)
```

**Example.** The sources are [1] the refunds policy and [2] the shipping policy.

> "Refunds take 14 days [1]. Shipping is free over 50 credits [1]."

- The first [1] is supported: passage 1 says 14 days.
- The second [1] isn't: free shipping is in passage 2, not passage 1.

Citation precision = 1 ÷ 2 = **0.5**.

The facts can be right while the citations are wrong. That matters, because a user who clicks [1] to check the shipping claim finds nothing about it. raggy's week 8 citation check verifies this on every answer, before the user sees it.

### 2.4 Refusal metrics: "Does it say 'not in the documents' at the right times?"

Each fictional suite includes **unanswerable** questions, whose answer isn't in the documents ("What's the price of the Kestrel Gravel?", a bike the shop doesn't sell). A good system refuses those and answers the rest. It's a 2 × 2 table:

| | Answerable question | Unanswerable question |
| --- | --- | --- |
| **Answered** | ✓ good | ✗ made something up |
| **Refused** | ✗ wrongly refused | ✓ good |

raggy reports three numbers from it.

**Example:** a suite with 33 answerable and 8 unanswerable questions. The model refuses 7 of the 8 unanswerable ones and 1 of the 33 answerable ones.

| Metric | Formula | Example | Ideal |
| --- | --- | --- | --- |
| `unanswerable_refused` | unanswerable refused ÷ unanswerable | 7 ÷ 8 = **0.875** | 1.0 |
| `answerable_refused` | answerable refused ÷ answerable | 1 ÷ 33 = **0.030** | 0.0 |
| `refusal_accuracy` | right decisions ÷ all questions | (7 + 32) ÷ 41 = **0.951** | 1.0 |

The one unanswerable question it *answered* is the dangerous case: a confident answer to something your documents never said. `unanswerable_refused` below 1.0 is usually the first thing to fix.

The two error rates trade off. Week 8's refusal gate refused unanswerable questions without calling the model, but it wrongly refused 1 answerable question in 17, so it stayed off.

### 2.5 Noise: how big a difference is real?

Answer metrics come from two models (the one answering and the judge), and both sample. Run the same eval twice and the numbers move. In week 8, three runs that gave the model *identical* passages scored 0.879, 0.924 and 0.939 correctness on 33 questions.

A rough guide: with *n* questions and a score near *p*, the run-to-run spread is about √(p × (1 − p) ÷ n).

```text
p = 0.9, n = 33:   √(0.9 × 0.1 ÷ 33) = √0.00273 ≈ 0.05
```

So on a 33-question suite, a change of 0.05 in correctness can be pure noise. One question flipping between correct and incorrect is 1 ÷ 33 = 0.03 by itself.

What to do about it:

- **Trust retrieval metrics more.** They're deterministic: the same index and question always give the same ranking.
- **Run answer evals twice**, and believe only differences larger than the run-to-run spread.
- **Use more questions:** the spread shrinks with √n. 4× the questions gives half the noise.
- **Look at per-question changes,** not just the mean: which questions flipped, and why?

### 2.6 Diagnosing with answer metrics

| What you see | What it means | Try |
| --- | --- | --- |
| Good recall@4, low correctness | The answer was in the context but not used | The prompt, a stronger model, packing (week 8) |
| Low correctness, low recall@4 | It never had the answer | Fix retrieval first (part 1) |
| Low faithfulness | It adds outside knowledge | A stricter prompt; the citation check with `refuse` (week 8) |
| Low citation precision | Right facts, wrong [n] | The citation check with `strip` (week 8) |
| Low `unanswerable_refused` | It makes up answers | The prompt's refusal rule; a calibrated gate (week 8) |
| High `answerable_refused` | It gives up too easily | Better retrieval; a less strict prompt |

---

## Part 3: Cost and speed metrics

### 3.1 Latency percentiles: p50 and p95

Time per question, per stage (transform, retrieve, rerank, generate), summarised by percentiles instead of the average.

**Example:** ten questions' generate times, sorted:

```text
110  120  125  130  140  150  160  180  250  900   (ms)
```

| Statistic | Value | What it says |
| --- | --- | --- |
| **p50** (median) | **145 ms** (halfway between 140 and 150) | The typical question |
| **p95** | **≈ 608 ms** | The slow end: 1 in 20 questions is at least this slow |
| mean | 226 ms | Pulled up by one slow question: describes no real question |

The mean is misleading: no question took 226 ms, and one 900 ms outlier inflates it. p50 is the typical experience; p95 is what users complain about.

raggy compares each stage's p95 with a budget (week 7: retrieve 100 ms, rerank 500 ms, transform 2,000 ms, generate 5,000 ms) and flags stages over it. The week 7 reranker was kept off partly because its p95 (512–708 ms) was over its 500 ms budget.

### 3.2 Context tokens

How many tokens of passages the answer model read per question, reported as p50 and p95. More context costs more, takes longer, and isn't automatically better: the answer can get lost among irrelevant passages. Week 8's packing measured whether merging overlapping passages lowered it.

### 3.3 Token usage and cost

Every run records input tokens, output tokens and cached input tokens. With prices set (`RAGGY_PRICES`, week 12), raggy turns them into money:

```text
cost = (input − cached) × input price + cached × cached price + output × output price
       (prices per million tokens)
```

**Example:** 1,200 input tokens (400 of them cached) and 60 output tokens, at input 2.00, cached 0.20 and output 8.00 per million:

```text
(800 × 2.00 + 400 × 0.20 + 60 × 8.00) ÷ 1,000,000
= (1,600 + 80 + 480) ÷ 1,000,000
= 0.00216 per answer  →  21.60 per 10,000 answers
```

---

## Part 4: Cheat sheet

| Metric | Range | Ideal | Measures | Deterministic? |
| --- | --- | --- | --- | --- |
| recall@k | 0–1 | 1 | Is the answer in the top k? | Yes |
| MRR | 0–1 | 1 | How high is the first right passage? | Yes |
| nDCG@k | 0–1 | 1 | How good is the whole top-k order? | Yes |
| correctness | 0–1 | 1 | Is the answer right? | No (judged) |
| faithfulness | 0–1 | 1 | Is every claim in the sources? | No (judged) |
| citation precision | 0–1 | 1 | Does each [n] support its sentence? | No (judged) |
| unanswerable_refused | 0–1 | 1 | Does it refuse what isn't there? | Nearly (the model decides) |
| answerable_refused | 0–1 | 0 | Does it refuse what is there? | Nearly |
| refusal_accuracy | 0–1 | 1 | Both refusal decisions together | Nearly |
| p50 / p95 latency | ms | low, within budget | Typical / slow-end speed | Varies with load |
| context tokens | tokens | as low as quality allows | How much the model reads | Yes |

**The order to read a run in:**

1. **recall@4.** Did the answer reach the model? If not, nothing downstream can fix it.
2. **recall@1 and MRR.** Was it first?
3. **unanswerable_refused.** Does it make things up?
4. **correctness and faithfulness.** Did it use what it had, and only that? Remember the noise.
5. **p95 per stage and cost.** Can you afford it?

## Further reading

- [`docs/evaluating-rag.md`](evaluating-rag.md): week 2's guide, with span-level gold, how the judge prompts are written and versioned, and why a weak judge gives noisy scores.
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), Thakur et al. (2021): where SciFact and FiQA's nDCG@10 numbers come from.
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), Es et al. (2023): faithfulness and answer relevance judged by LLMs.
