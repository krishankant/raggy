# raggy

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#setup)
[![Models](https://img.shields.io/badge/answers-Ollama%20%7C%20Claude%20%7C%20OpenAI-6f42c1.svg)](lessons/week01-naive-rag.md#36-one-interface-three-model-providers)
[![Tests](https://img.shields.io/badge/tests-offline%20pytest-2ea44f.svg)](#commands)

**[Course Plan](COURSE_PLAN.md)** · **[Week 1: Naive RAG](lessons/week01-naive-rag.md)** · **[Bonus: Agentic RAG](lessons/bonus-agentic-and-graph-rag.md)** · **[Concepts (start here)](concepts/README.md)** · **[Docs hub](docs/README.md)** · **[Metrics explained](docs/metrics-explained.md)**

---

**raggy** is a retrieval-augmented generation (RAG) pipeline that grows one week at a time, from a roughly 770-line naive version to a system that answers questions over thousands of documents of any kind (text, web pages, PDFs, Office files, images, audio and video), with citations back to the exact source.

It's a course you **read and run**, not one you fill in. Every week adds working code and a lesson that explains what was added, why it's built that way and what it changed, with eval numbers from real runs. Each week is preserved on its own branch, so you can check out any stage of the build.

> **RAG in one sentence.** Instead of trusting what a language model memorised, *retrieve* the passages from your own documents that are relevant to a question, then have the model *generate* an answer from those passages only, citing them.

No RAG framework (no LangChain or LlamaIndex): only focused libraries and the official model SDKs, so every stage is code you can read.

## Concepts: start here if you're new to RAG

Fifteen pages that teach each idea from the ground up, independent of raggy's code, with diagrams, worked examples and reading links. Read a page before the week that uses it. Index and glossary: [`concepts/README.md`](concepts/README.md).

| # | Concept | What you'll learn | Used in |
| --- | --- | --- | --- |
| 01 | [What is RAG?](concepts/01-what-is-rag.md) | Why LLMs need retrieval; RAG vs fine-tuning vs long context; where RAG fails | Week 1 |
| 02 | [LLM basics for RAG](concepts/02-llm-basics.md) | Tokens, context windows, how generation works, temperature, hallucination, cost | Week 1 |
| 03 | [Document parsing](concepts/03-document-parsing.md) | What's inside a PDF, HTML boilerplate, OCR, tables, normalisation, de-duplication | Weeks 3, 9 |
| 04 | [Chunking](concepts/04-chunking.md) | Fixed, recursive, structural and semantic splitting; overlap; parent/child; contextual retrieval | Week 4 |
| 05 | [Embeddings](concepts/05-embeddings.md) | How text becomes a vector; contrastive training; pooling; query prefixes; choosing a model | Week 1 |
| 06 | [Vector similarity and search](concepts/06-vector-search.md) | Cosine worked by hand; exact search; IVF, HNSW, quantisation; vector databases | Weeks 1, 5 |
| 07 | [Keyword search and BM25](concepts/07-keyword-search-bm25.md) | Inverted indexes, TF-IDF, BM25 worked by hand, learned sparse retrieval | Week 6 |
| 08 | [Hybrid search and rank fusion](concepts/08-hybrid-search.md) | Why combine vectors and keywords; score normalisation; RRF worked by hand | Week 6 |
| 09 | [Reranking and cross-encoders](concepts/09-reranking.md) | Retrieve-then-rerank; bi- vs cross-encoders; ColBERT; LLM rerankers; costs | Week 7 |
| 10 | [Query transformation](concepts/10-query-transformation.md) | Rewriting, multi-query, HyDE, step-back, decomposition, routing | Week 7 |
| 11 | [Grounded generation](concepts/11-grounded-generation.md) | Prompt anatomy, context packing, lost-in-the-middle, citations, refusals | Week 8 |
| 12 | [Evaluating RAG](concepts/12-evaluation.md) | Test sets, retrieval vs answer metrics, LLM-as-judge, RAGAS, noise | Week 2+ |
| 13 | [Multimodal RAG](concepts/13-multimodal-rag.md) | Images (OCR, vision models, CLIP, ColPali), tables, audio, video | Weeks 9–11 |
| 14 | [RAG in production](concepts/14-production.md) | Latency, streaming, caching, cost per answer, tracing, security | Week 12 |
| 15 | [Agentic and graph RAG](concepts/15-agentic-and-graph-rag.md) | Agents and ReAct, Self-RAG, CRAG, multi-hop, knowledge graphs, GraphRAG | Bonus |

## Quick start

```bash
git clone https://github.com/krishankant/raggy.git && cd raggy
uv sync
ollama pull qwen3.5:4b                      # free local answers (or use Claude/OpenAI, below)

uv run raggy ingest samples/week01          # index the sample documents
uv run raggy ask "Which bike comes in an XS frame size?"
```

```
The only Brightwater model sold in an XS frame size is the **Kestrel Gravel**. It weighs **10.2 kg** [1].
Sources:
[1] bike-models.md — "# The Brightwater range (2026) ## Heron Commuter A steel city bike with an 8-speed hub gear, hub dyn…"
```

Point it at your own files, folders or web pages: `uv run raggy ingest ~/notes ~/papers/report.pdf https://example.com/faq`. Run it again any time: only what changed is re-indexed.

## Setup

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). Then `uv sync`.
- **Answer model**, pick one:
  - Local and free: [Ollama](https://ollama.com) plus `ollama pull qwen3.5:4b` (the default).
  - Claude: set `RAGGY_LLM=anthropic` and `ANTHROPIC_API_KEY` in `.env`.
  - OpenAI: set `RAGGY_LLM=openai`, `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`.
- `cp .env.example .env` and edit. Keys go in `.env`, never in your shell rc file.
- The embedding model (`BAAI/bge-small-en-v1.5`, 67 MB) downloads automatically on first use and runs locally.

## Commands

| Command | What it does |
| --- | --- |
| `uv run raggy ingest PATH_OR_URL... [--full]` | Index files, folders and URLs (`.md`, `.txt`, `.html`, `.pdf`). Re-runs skip unchanged sources, re-index changed ones and drop files deleted from a folder. `--full` re-reads every file. Folders honour `.raggyignore` and `.gitignore`. |
| `uv run raggy search "query" [-k N] [--mode vector\|keyword\|hybrid] [--type pdf] [--source DIR] [--after 2026-01-01]` | Show the chunks retrieval finds, with scores. Your first debugging step. The mode and filters also work on `ask`. |
| `uv run raggy ask "question" [-k N] [--show-context]` | Answer with numbered citations. |
| `uv run raggy sources` | List what's indexed: chunks, type, title, source, and which documents were skipped as duplicates. |
| `uv run raggy remove PATH_OR_URL` | Drop a source, or everything under a folder or URL, from the index. Never touches the files. |
| `uv run raggy eval [--suite fictional\|fictional-web\|manual\|rich\|images\|media\|multihop\|scifact\|fiqa] [--retrieval-only] [--no-judge] [--limit N] [--label L]` | Measure retrieval and answer quality; saves a JSON run to `evals/runs/`. |
| `uv run raggy compare BASE.json NEW.json` | Show each metric's change between two runs. |
| `uv run pytest` | Run the offline test suite (no keys, no downloads). |

| Setting (`.env`) | Default |
| --- | --- |
| `RAGGY_LLM` | `ollama` (`anthropic`, `openai`) |
| `RAGGY_LLM_MODEL` | `qwen3.5:4b` for Ollama, `claude-opus-5-5` for Claude |
| `RAGGY_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `RAGGY_INDEX_DIR` | `.raggy` |
| `RAGGY_TOP_K` | `4` |
| `RAGGY_JUDGE` / `RAGGY_JUDGE_MODEL` | `anthropic` (`claude-opus-5-5`) if `ANTHROPIC_API_KEY` is set, else `ollama` |
| `RAGGY_HTML_EXTRACT` | `main` (article text only; `full` keeps every visible word) |
| `RAGGY_DEDUPE` | `1` (skip exact and near-duplicate documents; `0` indexes them) |
| `RAGGY_CHUNKER` | `structure` (Markdown-aware; `recursive`; `fixed` = weeks 1–3) |
| `RAGGY_CHUNK_TOKENS` / `RAGGY_CHUNK_OVERLAP_TOKENS` | `256` / `32` (in the embedding model's tokens) |
| `RAGGY_PARENT_TOKENS` | `512` (search small chunks, answer with their parent; `0` = off) |
| `RAGGY_CHUNK_HEADERS` | `1` (embed "title › section" with each chunk) |
| `RAGGY_CHUNK_CONTEXT` | `none` (`llm`: contextual retrieval, one cached LLM call per chunk) |
| `RAGGY_STORE` | `lance` (on-disk LanceDB; `numpy` = week 1's in-memory matrix) |
| `RAGGY_EMBED_CACHE` | `1` (never embed the same text twice with the same model) |
| `RAGGY_ANN` / `RAGGY_ANN_NPROBES` / `RAGGY_ANN_REFINE` | `off` / `20` / `4` (approximate index; worth it well past 100k chunks) |
| `RAGGY_RETRIEVAL` | `vector` (`keyword` = BM25, `hybrid` = both fused with RRF) |
| `RAGGY_RRF_K` / `RAGGY_FUSION_CANDIDATES` | `60` / `50` |
| `RAGGY_RERANK` / `RAGGY_RERANK_MODEL` / `RAGGY_RERANK_CANDIDATES` | `none` (`local` cross-encoder, `llm`) / `Xenova/ms-marco-MiniLM-L-6-v2` / `20` |
| `RAGGY_QUERY_TRANSFORM` | `none` (`rewrite`, `multi`, `hyde`: one cached LLM call per question) |
| `RAGGY_BUDGETS` | `transform=2000,retrieve=100,rerank=500,generate=5000` (p95 ms per stage, reported by `raggy eval`) |
| `RAGGY_PACK` / `RAGGY_CONTEXT_TOKENS` | `0` (merge, order and budget the passages) / `3000` |
| `RAGGY_CITATION_CHECK` / `RAGGY_CITATION_ACTION` / `RAGGY_CITATION_THRESHOLD` | `off` (`lexical`, `llm`) / `report` (`strip`, `refuse`) / `0.5` |
| `RAGGY_PDF` / `RAGGY_OCR` | `layout` (headings, ruled tables, page labels; `text` = week 3's text layer) / `on` (OCR pages with no text layer) |
| `RAGGY_IMAGES` / `RAGGY_VISION` / `RAGGY_VISION_MODEL` | `off` (`ocr`, `vision`) / the answer model's provider / the answer model, or `qwen3.5:4b` for Ollama |
| `RAGGY_MEDIA` / `RAGGY_WHISPER_MODEL` | `off` (`on`: transcribe audio and video locally) / `base` (`tiny`, `small`, `medium`, `large-v3`) |
| `RAGGY_MEDIA_WINDOW` / `RAGGY_KEYFRAME_SECONDS` | `60` (seconds per transcript window) / `30` (video frames, read when `RAGGY_IMAGES` is on) |
| `RAGGY_PROFILE` | unset (`local`, `fast`, `accurate`: fill in settings you haven't set) |
| `RAGGY_AGENT` / `RAGGY_AGENT_STEPS` | `0` (`1`: the model decides what to search next, for multi-hop questions; or `raggy ask --agent`) / `4` |
| `RAGGY_ANSWER_CACHE` / `RAGGY_TRACE` / `RAGGY_INGEST_WORKERS` | `1` (repeat questions over identical passages cost nothing) / `0` (`1`: one JSON line per answer) / `4` |
| `RAGGY_PRICES` | empty (`openai:gpt-5.5=1.25/10/0.125,...`: input/output/cached per 1M tokens, for cost per answer) |
| `RAGGY_IMAGE_SEARCH` | `off` (`on`: CLIP vectors for every picture, `raggy images "query"`) |
| `RAGGY_REFUSAL_GATE` | empty = off (`vector=0.63`: refuse without a model call below that top score; calibrate with `evals/calibrate_gate.py`) |

## The course

See **[COURSE_PLAN.md](COURSE_PLAN.md)** for all 12 weeks. All twelve weeks and the bonus are ready:

| Week | Lesson | Code |
| --- | --- | --- |
| 1 | [Naive RAG, end to end](lessons/week01-naive-rag.md) | `git checkout week-01` |
| 2 | [Evals before optimisation](lessons/week02-evals.md) | `git checkout week-02` |
| 3 | [Ingestion: files, folders, URLs](lessons/week03-ingestion.md) | `git checkout week-03` |
| 4 | [Chunking: where to cut](lessons/week04-chunking.md) | `git checkout week-04` |
| 5 | [Scale: tens of thousands of documents](lessons/week05-scale.md) | `git checkout week-05` |
| 6 | [Hybrid search](lessons/week06-hybrid-search.md) | `git checkout week-06` |
| 7 | [Reranking and query transforms](lessons/week07-rerank-and-query-transforms.md) | `git checkout week-07` |
| 8 | [Grounded generation](lessons/week08-grounded-generation.md) | `git checkout week-08` |
| 9 | [Rich documents: PDF layout, tables, OCR, Office files](lessons/week09-rich-documents.md) | `git checkout week-09` |
| 10 | [Images: OCR, vision descriptions, picture search](lessons/week10-images.md) | `git checkout week-10` |
| 11 | [Audio and video: transcripts, keyframes, time citations](lessons/week11-audio-video.md) | `git checkout week-11` |
| 12 | [Production: streaming, caching, concurrency, cost, traces, profiles](lessons/week12-production.md) | `git checkout week-12` |
| Bonus | [Agentic and graph RAG: when one search isn't enough](lessons/bonus-agentic-and-graph-rag.md) | `git checkout bonus` |

## Repository layout

```
raggy/            the package: ingest/ chunking/ embed/ store/ generate/ evals/ pipeline.py cli.py
lessons/          one lesson per week
docs/             concept guides (embeddings, evals, ingestion, chunking, vector search at scale, hybrid search, reranking, grounded generation, rich documents, images, audio and video, production)
samples/          small demo corpora (fictional, so models can't answer from memory)
evals/            eval suites (fictional corpus, its web-site version, a long manual), committed baselines; data/ and runs/ are local
tests/            offline pytest suite, one folder per week
```
