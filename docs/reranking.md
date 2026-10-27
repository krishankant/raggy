# Reranking and query transforms

**[← Docs hub](README.md)** · **[Week 7 lesson](../lessons/week07-rerank-and-query-transforms.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [09 Reranking and cross-encoders](../concepts/09-reranking.md) · [10 Query transformation](../concepts/10-query-transformation.md).

This guide sits beside week 7. It explains why a second, slower ranker over the top few results is worth having, how cross-encoders and LLM rankers differ from the embedding model that retrieved the candidates, what the three query transforms change, and how to budget latency per stage.

---

## 1. Two kinds of model, two kinds of cost

**A bi-encoder** (the embedding model, week 1) encodes the question and each passage *separately* into vectors. Passages are encoded once, at ingest. A search is one question encoding plus dot products, so a million passages cost milliseconds. The catch: the passage vector was computed without knowing the question. Everything the model knows about relevance has to survive being squeezed into 384 numbers per passage.

**A cross-encoder** takes the question and one passage *together* as a single input, so its attention layers can match each question word against each passage word, and it outputs one relevance score. It's much more precise. But nothing can be computed ahead of time: scoring N passages costs N full model passes at query time. Over a whole index that's minutes; over the retriever's top 20, it's about 0.4 s on a laptop CPU for MiniLM-L-6.

The two-stage pattern follows directly: **retrieve wide and cheap, then rank narrow and precise.** The retriever's job becomes recall (get the right passage *somewhere* in the top 20). The reranker's job is precision (put it first).

## 2. What reranking can and can't fix

A reranker reorders the candidates; it can't add one. If the right passage is at retrieval rank 35 and you rerank 20, it's gone. So:

- Reranking lifts **recall@1 and MRR**, the order at the top. In week 7, mean recall@1 over the fictional suites went from 0.80 to 0.95 (MiniLM) and to 1.00 (an LLM).
- It barely moves **recall@k** for k near the candidate count, because those passages were already found.
- The candidate count is a trade-off: more candidates give the reranker more to rescue, and cost proportionally more time.

## 3. Listwise LLM ranking

A cross-encoder scores each passage alone (*pointwise*). An LLM can see all 20 at once and return an order (*listwise*), which lets it compare passages to each other: "this one states the refund period; that one only mentions refunds". raggy asks for `{"ranking": [3, 1, 7, ...]}` and repairs what comes back:

| Reply | Handling |
| --- | --- |
| An id that wasn't a candidate | Dropped |
| An id twice | Kept at its first position |
| Candidates left out | Appended in retrieval order |
| No parseable JSON | Retrieval order kept, failure counted |

It's the most accurate option measured (recall@1 = 1.000 on all three fictional suites) and the slowest: 7–10 s and one paid call per question.

## 4. Changing the question instead

The other lever is the query. Three transforms, each one LLM call per question:

**Rewrite.** Users write "hey so my bike's battery died after like a year, is that covered??". Documents say "The battery is covered for two years". A rewrite turns the first into "battery warranty coverage period", closer to the document's wording.

**Multi-query.** Three phrasings of the same question are each searched, and the rankings are fused with RRF (week 6). A passage that ranks well for several phrasings rises. This hedges against one unlucky wording.

**HyDE (hypothetical document embeddings).** Instead of embedding the question, an LLM writes a short passage that *would* answer it, and that passage is embedded. A question and its answer often look different ("Does X cause Y?" against "In a cohort of 4,000 patients, X was associated with..."). A fake answer looks like a real one, so it lands nearer to it. The risk is that the fake answer is confidently specific in the wrong direction and lands near the wrong section. In week 7, HyDE lost one manual question this way.

All three are cached by (model, prompt version, question), so repeated questions and eval reruns are free.

## 5. Budgets per stage

A latency target for the whole pipeline hides where time goes. raggy times each stage (`transform`, `retrieve`, `rerank`, `generate`) and compares each one's p95 with its own budget. An illustration, with numbers from two of week 7's fictional runs (rewrite and MiniLM):

```text
Stage        p50       p95    budget
transform   1756 ms   2333 ms  2000 ms  ⚠
retrieve      14 ms     22 ms   100 ms
rerank       399 ms    512 ms   500 ms  ⚠
```

Defaults (`RAGGY_BUDGETS="transform=2000,retrieve=100,rerank=500,generate=5000"`) were set for an 8 GB M2 before measuring. A stage over budget is a design question: a smaller model, fewer candidates, a cache, or dropping the stage.

## 6. When to turn these on

| Your situation | Try |
| --- | --- |
| The right passage is usually in the top 5 but rarely first | `RAGGY_RERANK=local` |
| Accuracy matters far more than seconds or cost | `RAGGY_RERANK=llm` |
| Users write long, chatty questions | `RAGGY_QUERY_TRANSFORM=rewrite` |
| Questions are phrased very unlike the documents (claims against abstracts) | `hyde`, then check with `raggy eval` |
| Short questions that already use the documents' words | Neither: measure first |

Whatever you try, measure it on your own questions with `raggy eval`, and check the ⚠ line for fallbacks.

## Further reading

- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), Nogueira & Cho (2019): the two-stage pattern.
- [RankGPT: Is ChatGPT Good at Search?](https://arxiv.org/abs/2304.09542), Sun et al. (2023): listwise LLM reranking with sliding windows.
- [HyDE](https://arxiv.org/abs/2212.10496), Gao et al. (2022).
- [Query Rewriting for Retrieval-Augmented LLMs](https://arxiv.org/abs/2305.14283), Ma et al. (2023).
