# Hybrid search: BM25, embeddings, and fusing them by rank

**[← Docs hub](README.md)** · **[Week 6 lesson](../lessons/week06-hybrid-search.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [07 Keyword search and BM25](../concepts/07-keyword-search-bm25.md) · [08 Hybrid search and rank fusion](../concepts/08-hybrid-search.md).

This guide sits beside week 6. It explains why embeddings and keywords fail on different questions, how BM25 scores a document (worked by hand), and why hybrid search fuses *ranks* instead of scores. The numbers in the worked examples were computed with the formulas shown.

---

## 1. Two kinds of miss

**Embeddings** turn meaning into geometry: "How long until I get my money back?" lands near "Refunds are paid within 14 days" although they share no words. But an embedding model has never seen most of your product codes, part numbers and surnames, so it splits them into word pieces (`BW-CH-1142` → `b ##w - ch - 114 ##2`) and their vectors carry little of their identity. A question about `BW-CH-1142` can land closer to a chunk about `BW-CH-1210`, because the pieces look alike.

**Keywords** (BM25) match exact words, so `BW-CH-1142` finds exactly the chunks containing it, and rare words count most. But keywords know nothing about meaning: "money back" doesn't match "refund".

Hybrid search runs both and merges the results. The bet is that a chunk relevant to the question is usually found by at least one of them.

## 2. BM25, worked by hand

BM25 scores a document D for a query by adding, over each query term t:

```
idf(t) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × |D| / avgdl))
```

- **tf**: how often t appears in D.
- **idf(t) = ln(1 + (N − n + 0.5) / (n + 0.5))**: rarer terms weigh more. N is the number of documents and n the number containing t.
- **k1 (1.2)**: term-frequency saturation. The 10th "refunds" adds less than the 2nd.
- **b (0.75)**: length normalisation. A long document mentioning t once is weaker evidence than a short one.

Three tiny documents, after stop words are dropped:

| doc | text (stop words removed) | length |
| --- | --- | --- |
| d1 | refunds paid 14 days return arriving | 6 |
| d2 | bw ch 1142 chain fits heron commuter | 7 |
| d3 | refunds faulty parts paid credits refunds 14 days | 8 |

avgdl = 7. Query: **"refunds 14"**. Both terms appear in 2 of the 3 documents, so idf = ln(1 + 1.5/2.5) = **0.470** for each.

- **d1:** tf 1 and 1, slightly short (6/7). Score **0.998**.
- **d3:** tf 2 for "refunds", 1 for "14", slightly long (8/7). Score **1.065**.
- **d2:** neither term. Score **0**.

d3 mentions "refunds" twice, but it doesn't score twice as much. Saturation, as tf × (k1+1) / (tf + k1) with no length effect: tf 1 → 1.00, 2 → 1.38, 3 → 1.57, 5 → 1.77, 10 → 1.96. It approaches 2.2 and never passes it. Repetition can't buy a top rank.

In raggy, LanceDB's full-text index computes these scores. It lower-cases, folds accents, stems ("refunds" → "refund") and drops stop words before indexing. The text it indexes is the same text the embedder reads: section path + chunk.

## 3. Why fuse ranks, not scores

A BM25 score for one question might be 12.4 and for the next 3.1. A cosine similarity is about 0.5–0.9 and is compressed differently by every model. Adding the two lets whichever side happens to have larger numbers decide, and which side that is changes from query to query. Normalising each list to 0–1 helps less than it seems, because the top score of a list says nothing about whether the list is any good.

**Reciprocal rank fusion** (Cormack, Clarke & Büttcher, 2009) ignores scores entirely:

```
RRF(c) = Σ over rankings  1 / (k + rank(c))        k = 60, ranks start at 1
```

| rank | 1 | 2 | 3 | 4 | 5 | 10 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / (60 + rank) | 0.0164 | 0.0161 | 0.0159 | 0.0156 | 0.0154 | 0.0143 | 0.0091 |

The curve is flat on purpose. A chunk ranked 3rd by both sides scores 0.0318, which beats a chunk ranked 1st by one side and missing from the other (0.0164). Agreement between the two methods counts more than a strong opinion from one. Worked example with rankings [a, b, c] and [c, a]:

- a = 1/61 + 1/62 = **0.03252**
- c = 1/63 + 1/61 = **0.03227**
- b = 1/62 = **0.01613**

The order is a, c, b, and `test_rrf_sums_and_dedupes` pins exactly these values.

Two implementation details. A chunk found by both sides is one entry with both contributions summed, never two entries. And ties break by best single rank, then by id, so the same question always gives the same order.

## 4. Filters and parents in hybrid search

- **Filters go to both sides before ranking.** "The 4 best PDFs" filters inside the vector search *and* inside the BM25 search, then fuses. Filtering after fusion could return fewer than 4.
- **Fusion happens on the small chunks, then parent expansion** (week 4) swaps each one for its parent. One parent can hold several top children, so raggy fetches more children until it has k distinct parents.

## 5. What to expect

On public benchmarks, BM25 and dense retrieval each win on different datasets. On BEIR, BM25 scores nDCG@10 0.665 on SciFact and 0.236 on FiQA. Hybrid search is usually at least as good as the better of the two, and is best on collections that mix natural-language questions with exact identifiers. The week 6 lesson measures it on all five of raggy's suites, and the rule for whether hybrid becomes the default was written down before the first run.

## Further reading

- S. Robertson, H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009): where k1 and b come from.
- G. Cormack, C. Clarke, S. Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009).
- N. Thakur et al., "BEIR" (2021), table 2: BM25 against dense retrievers on 18 datasets.
