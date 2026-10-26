# raggy documentation hub

**[← Repository Home](../README.md)** · **[Course Plan](../COURSE_PLAN.md)**

Concept guides that sit beside the weekly lessons. Lessons walk through raggy's code; guides explain the idea behind it in more depth.

| Guide | Used in | What it covers |
| --- | --- | --- |
| [Embeddings and vector search](embeddings-and-vector-search.md) | Week 1 | What an embedding is, cosine similarity with a worked example, normalisation, query vs passage embeddings, exact vs approximate search |
| [Evaluating RAG](evaluating-rag.md) | Week 2 | Span-level gold evidence; recall@k, MRR and nDCG worked by hand; chunk vs document scoring; refusal accuracy, correctness, faithfulness, citation precision; versioned judge prompts; p50/p95 latency; how big a change is real |
| [Ingestion](ingestion.md) | Week 3 | Why boilerplate hurts retrieval; main-content extraction; Unicode normalisation; MinHash and LSH worked through; incremental re-ingest as a state machine; what "deleted" should mean |
| [Chunking](chunking.md) | Week 4 | Tokens vs characters; fixed, recursive and structure-aware splitting; overlap; contextual headers; parent/child retrieval and its two traps; contextual retrieval; comparing chunkers at equal context |
| [Vector search at scale](vector-search-at-scale.md) | Week 5 | Exact search as a matrix multiply; IVF, HNSW, SQ/PQ and refine; measuring ANN by overlap with exact; when you need ANN; embedding throughput |
| [Hybrid search](hybrid-search.md) | Week 6 | Why vectors and keywords miss different things; BM25 worked by hand (idf, saturation, length); reciprocal rank fusion worked by hand; filters and parents in hybrid search |
| [Reranking and query transforms](reranking.md) | Week 7 | Bi-encoders against cross-encoders; what reranking can and can't fix; listwise LLM ranking; rewrite, multi-query and HyDE; latency budgets per stage |

More guides arrive with each week.
