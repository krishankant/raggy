# raggy documentation hub

**[← Repository Home](../README.md)** · **[Course Plan](../COURSE_PLAN.md)**

Concept guides that sit beside the weekly lessons. Lessons walk through raggy's code; guides explain the idea behind it in more depth.

**Starting from zero?** The [concepts folder](../concepts/README.md) teaches the general ideas first (what RAG, embeddings, BM25, reranking, cross-encoders and the rest *are* and how they work), with diagrams, worked examples and reading links, and no raggy code. These guides then explain the choices raggy makes.

| Guide | Used in | What it covers |
| --- | --- | --- |
| [Embeddings and vector search](embeddings-and-vector-search.md) | Week 1 | What an embedding is, cosine similarity with a worked example, normalisation, query vs passage embeddings, exact vs approximate search |
| [RAG metrics, explained with examples](metrics-explained.md) | Every week | recall@k, MRR, nDCG (worked by hand, with graded relevance), precision@k; correctness, faithfulness, citation precision, refusal rates; run-to-run noise; p50/p95, context tokens, cost; what to fix when each is low |
| [Evaluating RAG](evaluating-rag.md) | Week 2 | Span-level gold evidence; recall@k, MRR and nDCG worked by hand; chunk vs document scoring; refusal accuracy, correctness, faithfulness, citation precision; versioned judge prompts; p50/p95 latency; how big a change is real |
| [Ingestion](ingestion.md) | Week 3 | Why boilerplate hurts retrieval; main-content extraction; Unicode normalisation; MinHash and LSH worked through; incremental re-ingest as a state machine; what "deleted" should mean |
| [Chunking](chunking.md) | Week 4 | Tokens vs characters; fixed, recursive and structure-aware splitting; overlap; contextual headers; parent/child retrieval and its two traps; contextual retrieval; comparing chunkers at equal context |
| [Vector search at scale](vector-search-at-scale.md) | Week 5 | Exact search as a matrix multiply; IVF, HNSW, SQ/PQ and refine; measuring ANN by overlap with exact; when you need ANN; embedding throughput |
| [Hybrid search](hybrid-search.md) | Week 6 | Why vectors and keywords miss different things; BM25 worked by hand (idf, saturation, length); reciprocal rank fusion worked by hand; filters and parents in hybrid search |
| [Reranking and query transforms](reranking.md) | Week 7 | Bi-encoders against cross-encoders; what reranking can and can't fix; listwise LLM ranking; rewrite, multi-query and HyDE; latency budgets per stage |
| [Grounded generation](grounded-generation.md) | Week 8 | Packing (merge, order, budget); lexical citation support worked by hand; LLM checks; refusal gates and split-half calibration; run-to-run noise; when prompt caching pays |
| [Rich documents](rich-documents.md) | Week 9 | What a PDF really stores; rebuilding headings and ruled tables; OCR's costs; locations for pages, slides and sheets; what retrieval metrics can't see about tables |
| [Images](images.md) | Week 10 | OCR vs a vision model vs CLIP; why OCR text is kept beside a description; the header as citation and gold; caching by image bytes; what gets skipped |
| [Audio and video](audio-video.md) | Week 11 | Whisper's timed segments and failure modes; windows by time and speaker; time labels as citations; keyframes and repeat skipping; diarization; captions first |
| [Running RAG in production](production.md) | Week 12 | Caching at every layer and how to key it; streaming; parallel reads, serial writes; cost per answer; what a trace must hold; profiles; a checklist before real users |
| [Agentic and graph RAG](agentic-and-graph-rag.md) | Bonus | Single-hop vs multi-hop and bridge entities; the retrieval loop and its design choices; GraphRAG's pipeline; a minimal graph index for raggy; when each pays |

One guide per week, weeks 1–12, plus the bonus.
