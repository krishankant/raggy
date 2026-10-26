# Week 6: Hybrid search — keywords and vectors together

**[← Week 5: Scale](week05-scale.md)** · **[Course Plan](../COURSE_PLAN.md)** · **[Hybrid search (guide)](../docs/hybrid-search.md)** · **[Week 7: Reranking →](week07-rerank-and-query-transforms.md)**

---

**Time:** about 4 hours (1 reading, 1.5 reading and running the code, 1.5 on the experiments).
**Code for this week:** `git checkout week-06`
**Companion guide:** [`docs/hybrid-search.md`](../docs/hybrid-search.md): BM25 and reciprocal rank fusion, worked by hand.

**You're done when:**

1. `uv run pytest` passes (433 tests, offline, no keys needed),
2. you've run `raggy search "BW-CH-1142" --mode vector` and `--mode keyword` and can explain the difference, and
3. you can say why hybrid search is *not* raggy's default, although it wins on four of the five suites.

## 1. Read first

| Read | Look for |
| --- | --- |
| [`docs/hybrid-search.md`](../docs/hybrid-search.md) (this repo) | BM25's three ingredients (idf, saturation, length), and why ranks are fused instead of scores. |
| [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114) (Cormack, Clarke & Büttcher, SIGIR 2009) | The whole method is one formula. The paper is short. Note that k = 60 was tuned on their data, not derived. |

## 2. Two retrievers that fail differently

Embeddings match meaning, and keywords match words. Each misses what the other finds:

- **Codes, names and rare words.** An embedding model splits `BW-CH-1142` into word pieces and can't tell it apart from `BW-CH-1210`. BM25 matches the exact token and weighs it heavily because it's rare.
- **Paraphrases.** "Money back" and "refund" share no words, which is fatal for BM25 and routine for an embedding.

Hybrid search runs both and fuses the rankings. This week builds the keyword side (a BM25 index in the same LanceDB table as the vectors), the fusion (reciprocal rank fusion, 31 lines), and a switch: `--mode vector | keyword | hybrid` on `search` and `ask`, or `RAGGY_RETRIEVAL`. Then it measures all three on every suite.

The result is more interesting than "hybrid wins". Hybrid is better on four of the five suites. On the fifth, FiQA, it's worse than vectors alone, and the rule written before the runs says that's enough to keep vectors as the default.

## 3. What you're building this week

```mermaid
flowchart LR
    Q[question] --> V[vector search<br/>top 50]
    Q --> K[BM25 search<br/>top 50]
    V --> F[RRF: Σ 1 / (60 + rank)]
    K --> F
    F --> P[top k → parents]
```

| File | Lines | What it does |
| --- | --- | --- |
| `retrieve/fusion.py` | 31 | `rrf(rankings, k=60)`: sums 1/(k + rank) per chunk; ties break by best rank, then id |
| `store/lance_store.py` | +42 | A `fts_text` column (the same text that's embedded: section path + chunk), a native LanceDB FTS index (lower-cased, stemmed, stop words removed), and `keyword_search()` |
| `pipeline.py` | +42 | `search(..., mode=)`, candidates from each side, and parent expansion that now fetches more until it has k distinct parents |
| `config.py`, `cli.py` | +12 | `RAGGY_RETRIEVAL`, `RAGGY_RRF_K` (60), `RAGGY_FUSION_CANDIDATES` (50), and `--mode` |

No new dependencies: LanceDB 0.40's built-in full-text index does the BM25.

## 4. Walkthrough

### 4.1 The keyword index

`LanceStore` rows gain `fts_text`, which is `embed_text(chunk)`: the context header plus the chunk. A heading word like "Warranty" is then findable by keyword, just as it's part of the embedding. `finish()`, which runs after every ingest, creates the full-text index if it's missing, or folds new rows into it. An existing index gains it on the next `raggy ingest`. An index built before week 6 has no `fts_text` column at all, and keyword search on it raises a one-line "run `raggy ingest` again" instead of a LanceDB schema error.

Two details came from probing LanceDB before writing the code:

- **The question is passed as a `MatchQuery`, not a query string.** A user's question may contain `?`, quotes, `AND` or a stray `(`. As a `MatchQuery` they're words, never query syntax. A question that's all stop words ("what is it?") returns no keyword hits, and hybrid falls back to the vector ranking.
- **Week 5's filters work on both sides.** The same `where` clause prefilters the BM25 search, so `--type pdf --mode hybrid` fuses two rankings of PDFs only (`test_hybrid_respects_filters`).

### 4.2 Fusing ranks

`rrf()` gives each chunk Σ 1/(60 + rank) over the rankings it appears in. A chunk found by both sides is one entry, never two. The guide works through why the scores themselves can't be added: BM25 scores are unbounded and change scale from query to query, while cosine similarities sit between about 0.5 and 0.9.

Each side contributes `max(k × 4, 50)` candidates. With parents on, children are fused first, then swapped for their parents.

### 4.3 A fix that the hybrid test found

`test_hybrid_with_parents` asked for 3 parents and got 2. The children of one parent rank together, so `k × 4` children can come from fewer than k parents. This was week 4's deferred minor ("fewer than k parents can come back"), and the hybrid test finally made it bite. `search` now doubles the child fetch until it has k distinct parents (capped at `k × 64`).

## 5. The numbers

All runs are retrieval only, with week 4's chunking and week 5's store.

### 5.1 First, check the ruler

Keyword-only search on the two public benchmarks, against the BM25 numbers BEIR published:

| | raggy keyword nDCG@10 | BEIR BM25 |
| --- | --- | --- |
| SciFact | 0.678 | 0.665 |
| FiQA | 0.242 | 0.236 |

Both are within 0.013. raggy's index chunks documents and adds section paths, and BEIR's indexes whole documents, so small differences are expected. A large one would have meant a broken tokenizer.

### 5.2 Vector vs keyword vs hybrid

| suite | metric | vector | keyword | hybrid |
| --- | --- | --- | --- | --- |
| fictional (33) | recall@4 / MRR | 0.970 / 0.915 | 0.970 / 0.937 | **1.000 / 0.949** |
| fictional-web (33) | recall@4 / MRR | 1.000 / 0.932 | 0.970 / 0.952 | **1.000 / 0.955** |
| manual (24) | recall@1 / MRR | 0.667 / 0.806 | 0.750 / 0.854 | **0.792 / 0.872** |
| SciFact (300) | nDCG@10 / recall@100 | 0.712 / 0.960 | 0.678 / 0.914 | **0.724 / 0.968** |
| FiQA (648) | nDCG@10 / recall@100 | **0.385** / 0.675 | 0.242 / 0.540 | 0.359 / **0.695** |

Retrieval latency at the eval's depth (p50 / p95; document-scored suites fetch about 1,200 chunks per question):

| suite | vector | keyword | hybrid |
| --- | --- | --- | --- |
| manual | 5.9 / 8.0 ms | 1.8 / 2.7 ms | 8.0 / 9.5 ms |
| SciFact | 34.6 / 41.8 ms | 28.3 / 30.9 ms | 65.4 / 84.0 ms |
| FiQA | 62.3 / 79.9 ms | 43.9 / 49.2 ms | 107.3 / 123.9 ms |

### 5.3 Which questions moved

On the fictional suites, BM25 alone was surprisingly strong. Questions tagged "paraphrase" still share a proper noun with their answer: f07 ("next-day delivery on a new bike") and f19 (the "Arlen Islands"), and f30 ("Ride to Work") went from rank 2+ under vectors to rank 1 under keywords and hybrid. Two went the other way: f06 (the Arlen Islands shipping cost) and f08 (the Ebb battery guarantee) were rank 1 under vectors and rank 2–4 under hybrid.

**f05**, week 4's one regression ("Why is the first frame line called Heron?"), is found at rank 1 by keywords and is back in hybrid's top 4. "Heron" is a rare word, which is exactly where BM25 is strong. On the manual, hybrid lifted three questions to rank 1: m09, m13 (an `E09` error code) and m20.

FiQA tells the other story. Keyword search is weak there (0.242): forum questions and answers use different words for the same thing. RRF weighs both rankings equally, so a weak ranking pulls a good one down. nDCG@10 drops by 0.026, although recall@100 rises (0.675 → 0.695): fusion still finds more of the relevant posts, but ranks them lower.

### 5.4 The decision

The rule, written into the plan before the runs: hybrid becomes the default only if (a) mean recall@4 over the three fictional suites rises, (b) SciFact and FiQA nDCG@10 don't fall by more than 0.005, and (c) FiQA p95 latency stays under 100 ms.

- (a) passes: 0.990 → 1.000.
- (b) fails on FiQA: −0.026.
- (c) fails: 124 ms. That's at eval depth; at `raggy ask` depth both modes are a few milliseconds, as week 5 showed. Condition (b) decides it on its own.

**So `RAGGY_RETRIEVAL` stays `vector`.** Hybrid is one switch away and is the better choice for collections like the fictional ones (product names, codes, policy pages). Try it on your own documents with `raggy eval` before switching. The FiQA result is the reason for the rule: a technique that helps on average can still hurt on a whole kind of collection.

Experiment 3 asks you to fix FiQA's drop with a weighted fusion. RRF's equal weights are a choice, not a law.

## 6. Why each choice

| Choice | Why | What it costs |
| --- | --- | --- |
| BM25 in the same LanceDB table | One store, one filter syntax, one ingest | Index is LanceDB-only (`RAGGY_STORE=numpy` can't do keyword search) |
| Index `fts_text` = section path + chunk | Headings become keywords, matching what the embedding sees | Heading words count once per chunk |
| `MatchQuery`, not query strings | User text can't become query syntax | No phrase or boolean queries from users |
| RRF with k = 60 | Score-scale-free; the published default | Equal weights: a weak side drags the other (FiQA) |
| 50 candidates per side | Enough for top-10 fusion; cheap | Deep fusion at eval depth costs ~2× latency |
| Vector stays the default | Rule written before the runs: FiQA −0.026 | The fictional suite loses 0.03 recall@4 (one question) |

## 7. Experiments and notes

Write your answers in `NOTES.md`.

1. `raggy search "BW-CH-1142" -k 3` in each mode on the fictional docs. Then try `"BW-CH-1210"`. Which mode tells them apart, and why can't the embedding?
2. Change `RAGGY_RRF_K` to 1, then to 1000, and rerun the fictional suite. What happens to MRR, and why? (Work out 1/(k+1) against 1/(k+2) for both values.)
3. Weighted fusion: change `rrf()` to take one weight per ranking and give the vector side 2× weight. Rerun FiQA and the fictional suite. Can one weight fix FiQA without losing the fictional gains?
4. Find a question in your own documents that only keyword search answers, and one that only vector search answers. What do they have in common?
5. Set `RAGGY_FUSION_CANDIDATES=10` and `=200`. How do recall@100 and latency move on SciFact?

---

**Tested with:** Python 3.13.14, lancedb 0.40.0 (native full-text index), fastembed 0.9.0, on an Apple M2 with 8 GB of RAM. All numbers are from the runs committed in `evals/baselines/week06/`.
