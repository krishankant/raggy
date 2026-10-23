# raggy: a RAG course you read, run and keep (design)

**Date:** 2026-10-23
**Status:** draft, waiting for approval
**Repo:** `~/claudecode/raggy`, next to `harnessy`, to be pushed to the owner's GitHub like harnessy

## 1. Intent

The owner finished harnessy (an agent-harness course) and wants to learn RAG the same way: one Python codebase that grows one week at a time, with a lesson `.md` per week explaining the concept and the reasons behind each choice.

At the end it is a working tool, not a toy:

- You point it at a **file, a folder, or a webpage URL**.
- It ingests everything there, **thousands of documents**, in **any media**: text, web pages, PDFs, Office files, images, audio and video.
- It answers questions with **citations** back to the exact source, page or timestamp.

**Why now:** the agent-harness concepts are understood, and retrieval is the next layer.

## 2. Priorities (owner's words: "all three, accuracy first, then speed")

1. **Accuracy:** right answers, grounded in the sources, with citations that point to the passage actually used. Saying "I don't know" when the corpus has no answer counts as correct.
2. **Speed:** ingest and query stay fast at thousands of documents.
3. **Cost:** runs free and local by default. Hosted models are optional.

Every week's improvement is **measured, not assumed**. The eval harness arrives in week 2, and each later lesson shows before and after numbers.

## 3. How the course works (decided by the owner)

This is the main difference from harnessy:

- **No exercise stubs and no `solutions/` folder.** There is one finished, working codebase. The learner reads the lesson next to the real code.
- **Each week = working code + a lesson.** The lesson walks through the code that week added. It covers the concept, why it is built this way, the trade-offs and alternatives, what breaks without it, and the eval numbers before and after.
- **Proper code versions.** Each week has a snapshot branch `week-NN` holding the code exactly as it stood at the end of that week, so checking out `week-03` shows only weeks 1 to 3. `main` holds everything. **Snapshot branches are never fast-forwarded, merged or reset to main.** (This is the rule the owner set for harnessy.) A fix that belongs to one week is cherry-picked onto that week's branch only. A matching tag `week-NN` is also created.
- **Tests verify the code. They are not exercises.** `uv run pytest` passes on every week branch, and runs offline with no API keys. Network and model calls are faked in tests.
- **Every week has a runnable demo** through the CLI, so the learner can see the improvement for themselves.

## 4. Goals

- A CLI, `raggy`, used as `uv run raggy <cmd>`:
  - `raggy ingest <path-or-url>...` adds files, folders (recursively) or URLs. Re-running it is incremental: unchanged sources are skipped, changed ones are re-indexed, deleted ones are removed.
  - `raggy ask "<question>"` returns an answer with numbered citations (source, location, quoted snippet). It streams by the production week.
  - `raggy search "<query>"` shows raw retrieval results with scores, for learning and debugging.
  - `raggy eval [--suite ...]` runs the eval suite and prints a metrics table.
  - `raggy sources` / `raggy remove <source>` manage the index.
- A Python package `raggy/` with small, swappable parts, each behind an interface: loaders, chunkers, embedders, stores, retrievers, rerankers, LLMs and the answer generator.
- Provider-neutral answering through one small LLM interface, with adapters for **Anthropic, OpenAI and Ollama** (local), in the same spirit as harnessy week 1. The default works with only a local model installed. Hosted models are switched on in config.
- Local embeddings by default (an open-source model run on the machine). A hosted embedding adapter is optional.
- Handles **thousands of documents**. A benchmark corpus of several thousand docs is part of the eval suite, and the lessons report ingest throughput and p50/p95 query latency.
- A lesson per week plus companion concept guides in `docs/`, with Mermaid diagrams and the same navigation style as harnessy (breadcrumb links, a course-plan table, a README with a cheat sheet).

## 5. Non-goals

- **No RAG or agent framework** inside raggy: no LangChain, LlamaIndex, Haystack or similar. Use focused libraries (a PDF parser, an embedding model runtime, a vector DB, a transcription model) plus the official provider SDKs. The lesson for each week may compare its design with a framework's, but the code never imports one.
- No server-based vector DB that has to run separately (Postgres/pgvector, Qdrant server, Elasticsearch). The store is embedded and lives on disk.
- No web UI. The CLI is the interface. A tiny optional HTTP API may appear in the production week only if it costs little.
- No multi-user auth, no cloud deployment and no distributed ingestion.
- No model fine-tuning or embedding training.
- GraphRAG and fully agentic RAG are out of the main path. They may be a bonus week at the end (see section 7).

## 6. Anti-goals: it fails even if it "works" when...

- Claims of improvement are not backed by eval numbers from a real run.
- Citations point to sources that don't contain the claim.
- Code samples in lessons don't match the code on that week's branch, or were written from memory against an old library API. Every quoted command output in a lesson is copied from a real run, and the library versions used are named. (The owner's observation log records this exact failure in an earlier tutorial: an SDK renamed a class across a major version.)
- A week's branch doesn't run or its tests fail.
- Lessons explain *what* the code does but not *why* this approach beats the simpler one that came before.

## 7. Course outline (12 weeks + a bonus)

Each week names its idea, what gets built, and how it is measured. The order follows the priorities: baseline, then measurement, then accuracy layers, with scale introduced when the corpus grows, then media, then production.

| Week | Idea | Built | Measured by |
| --- | --- | --- | --- |
| **1. Naive RAG end to end** | Retrieve, then generate. The whole loop in roughly 200 lines. | Load `.txt`/`.md` from a folder; fixed-size chunks; local embeddings; numpy cosine top-k; prompt with numbered sources; answer with citations; LLM interface plus Anthropic/OpenAI/Ollama adapters; `ingest`/`ask`/`search` CLI; pickle or `.npy` on disk | It answers questions about a small sample corpus, demonstrated by hand |
| **2. Evals before optimisation** | You can't improve what you don't measure. | Eval suite: (a) a **fictional corpus** (so the model can't answer from memory) with questions, gold passages and reference answers, including unanswerable ones; (b) a **public retrieval benchmark** with relevance labels for a few thousand docs (for example a BEIR subset such as SciFact; the builder confirms license and size). Metrics: recall@k, MRR, nDCG for retrieval; answer correctness, faithfulness and citation precision through an LLM judge, with judge prompts versioned; refusal accuracy on unanswerable questions; latency. Results are saved as JSON per run so weeks can be compared | Baseline numbers for week 1's pipeline. Every later week reports a delta against them |
| **3. Ingestion: folders, URLs, metadata** | Garbage in, garbage out. | Loader registry by file type; recursive folders with ignore rules; URL fetch plus main-content extraction (boilerplate removed); HTML; basic PDF text; a document registry (SQLite) with content hashes so re-ingest is incremental (skip, update, delete); normalisation and near-duplicate removal; metadata (source, title, type, dates) | Ingest is idempotent; eval delta from cleaner text |
| **4. Chunking** | Chunk boundaries decide what can be found. | Structure-aware chunking (headings, paragraphs, code blocks), recursive splitting, overlap, token-based sizing; contextual chunk headers (document title and section path prepended); parent/child chunks (search small, return the bigger parent); optionally LLM-written chunk context ("contextual retrieval") as a config switch, because it costs money | Recall@k and answer correctness for each strategy, shown as a table |
| **5. Scale: a real vector store** | numpy brute force stops being good enough. | Move to an embedded vector DB (LanceDB is the default pick; the builder confirms it is still the best embedded option); batched and parallel embedding; an embedding cache keyed by content hash; ANN index plus the trade-off between exact and approximate search; metadata filters; ingest the thousands-of-docs benchmark | Ingest docs/sec, p50/p95 query latency, recall loss from ANN compared with exact search |
| **6. Hybrid search** | Vectors miss exact names, codes and rare words; keywords miss paraphrases. | BM25 / full-text index alongside the vectors; reciprocal rank fusion; query-time filters by source/type/date | Retrieval metrics: vector only vs keyword only vs hybrid |
| **7. Reranking and query transformation** | Retrieve wide and cheap, then rank narrow and precise. | Cross-encoder reranker (local model by default, hosted optional); query rewriting, multi-query and HyDE as switches; a latency budget per stage | Accuracy gain against added latency, per technique. Only what earns its place stays on by default |
| **8. Grounded generation** | Retrieval is half the job; the answer must stay inside the sources. | Context packing (dedupe, ordering, token budget); citation format plus a post-check that each cited passage supports its sentence; "not in the documents" refusal; prompt caching where the provider supports it | Faithfulness, citation precision, refusal accuracy |
| **9. Rich documents** | Real documents aren't plain text. | PDF with layout (headings, tables as Markdown, page numbers in citations); OCR for scanned pages; DOCX, PPTX (slide numbers), XLSX/CSV (tables, kept row-aware) | Eval questions whose answers sit in tables, slides and scanned pages |
| **10. Images** | Pictures carry facts that text extraction misses. | Images (standalone files and figures inside PDFs/slides) described by a vision model into searchable text; optional multimodal embeddings for image-to-image and text-to-image search; images cited by file and page | Eval questions answerable only from an image |
| **11. Audio and video** | Speech is text with timestamps. | Local transcription (a Whisper-family model) with timestamps; chunking by time and by speaker turns; video = audio transcript plus keyframes described by the vision model; citations as `file @ mm:ss`; YouTube/remote media URLs where the tooling allows | Eval questions over a podcast and a video, with citations checked by timestamp |
| **12. Production** | Make it fast, cheap and observable. | Query and answer caching; streaming answers; concurrent ingest with a progress display; cost and token accounting per query; traces (JSONL, one record per stage with timings); config profiles (`local`, `quality`, `fast`); a final end-to-end benchmark over all media at thousands of docs | Final report: every week's metrics in one table, which shows the course's story with numbers |
| **Bonus: Agentic and graph RAG** | When one retrieval pass isn't enough. | The model decides when and what to retrieve (multi-hop) as a tool loop, linking back to harnessy; a short comparison with GraphRAG ideas | Multi-hop eval questions |

The builder may merge or split weeks if a week turns out much heavier or lighter than about 5 hours of reading and running. Changes to this table are reported, not done silently.

## 8. Repo layout (target at the end)

```
raggy/
  README.md                 # what it is, quick start, cheat sheet, links
  COURSE_PLAN.md            # the week table, how to use the branches, setup
  pyproject.toml / uv.lock
  .env.example              # API keys for optional hosted providers
  raggy/                    # the package
    cli.py
    config.py
    types.py                # Document, Chunk, Hit, Answer, Citation …
    ingest/                 # loaders/, registry.py, pipeline.py
    chunking/
    embed/                  # local + hosted embedders, cache
    store/                  # numpy store (wk1) → vector DB (wk5), fulltext
    retrieve/               # dense, sparse, hybrid, rerank, query transforms
    generate/               # llm adapters, prompts, citations, grounding check
    media/                  # pdf, office, ocr, vision, transcription
    evals/                  # datasets, metrics, judge, runner, reports
    trace.py
  lessons/week01-naive-rag.md … week12-production.md
  docs/                     # concept guides (embeddings explained, vector search
                            # math, BM25, ANN indexes, chunking, eval metrics …)
  evals/                    # corpora, question sets, saved run results
  samples/                  # small demo documents of every media type
  tests/                    # offline pytest, per week
```

## 9. Constraints

- Python 3.11+, `uv`, pytest, matching harnessy.
- Must run on the owner's Mac (Apple Silicon) with no GPU requirement. Heavy local models (reranker, Whisper, vision) must have a small default size and a configurable larger size.
- Tests run offline and fast. Model downloads and API calls happen only in demos and evals.
- API keys come from `.env`, never the shell rc file, because non-interactive runs don't read it (a harnessy lesson).
- Every library choice is checked against its **current release at build time**, and its version is pinned in `uv.lock` and named in the lesson where it's introduced.
- Commits are small and reviewable, one week at a time. Each week branch is cut when that week is finished.

## 10. Decisions made by the assistant (owner can override)

- Name: **raggy**.
- Local-first defaults: local embedding model, local reranker, local Whisper, with Ollama as a possible answer model. Claude, OpenAI and hosted embeddings are optional upgrades in config. The default eval judge is a hosted model (judging quality matters), with a local fallback.
- An embedded vector DB rather than a server: LanceDB is the starting candidate.
- Two eval corpora: a fictional one for answer quality, and a public labelled benchmark for retrieval at scale.
- CLI only, no web UI.
- 12 weeks + bonus, ordered as in section 7.
- Snapshot branches **and** tags named `week-NN`.

## 11. Left to the builder

- Exact libraries and model names for each role (embedding model, reranker, PDF/Office parsers, OCR engine, transcription runtime, HTML extraction), chosen by current quality, licence and Mac support, and justified in the lesson.
- The exact public benchmark subset and the size of the fictional corpus.
- Prompt wording, default chunk sizes, top-k values and fusion constants. These are set from eval results, not guessed.
- Whether week 1 stores vectors as `.npy` or something equally simple.
