# raggy: 12-week course plan

**[← Repository Home](README.md)** · **[Docs hub](docs/README.md)**

---

Over 12 weeks (plus a bonus) at about 5 hours a week, raggy grows from a naive RAG pipeline into one that answers questions over **thousands of documents of any media type**, with citations. The priorities, in order: **accuracy, then speed, then cost**. Runs are local and free by default.

## How this course works

- **The code is already written.** Each week adds working code, and the lesson walks through it: the concept, why it's built this way, the alternatives, what breaks without it, and the measured before and after.
- **Every week is a branch.** `git checkout week-03` gives you the code exactly as it stood after week 3, nothing from later weeks. `main` holds everything. (Matching tags `week-NN` exist too.)
- **Improvements are measured, not assumed.** From week 2 on, every lesson reports eval numbers from real runs: retrieval hit rate, answer correctness, faithfulness, refusals, and latency.
- **Tests run offline.** `uv run pytest` passes on every branch with no API keys and no downloads.

For each week: check out its branch, read the lesson with the code open beside it, run the commands in the lesson, then do the experiments and write notes in `NOTES.md`.

## The weeks

| Week | Idea | Builds | Status |
| :---: | --- | --- | --- |
| **1** | [Naive RAG, end to end](lessons/week01-naive-rag.md) | load → fixed chunks → local embeddings → numpy cosine search → cited answer; Ollama, Claude and OpenAI adapters; CLI | ✅ `week-01` |
| **2** | [Evals before optimisation](lessons/week02-evals.md) | A fictional Q&A set (41 questions, 8 unanswerable) plus SciFact (5,183 abstracts, 300 labelled claims); recall@k, MRR, nDCG, answer correctness, faithfulness, citation precision, refusal accuracy, latency; `raggy eval` / `raggy compare`; committed baselines | ✅ `week-02` |
| **3** | [Ingestion: files, folders, URLs](lessons/week03-ingestion.md) | A loader per file type (Markdown, text, HTML main content, PDF text); folders with `.raggyignore`/`.gitignore`; URLs; normalisation; exact and near-duplicate removal (MinHash); a SQLite registry so re-ingest skips, updates and deletes; `raggy remove`; a `fictional-web` eval suite | ✅ `week-03` |
| **4** | [Chunking: where to cut](lessons/week04-chunking.md) | Token-sized recursive and Markdown structure-aware chunkers; section-path headers; parent/child retrieval (256-token children, 512-token parents, the new default); optional cached contextual retrieval; a long-document `manual` suite; fair comparison by context size and answer correctness | ✅ `week-04` |
| **5** | [Scale: tens of thousands of documents](lessons/week05-scale.md) | LanceDB as the store (incremental writes, filters); an embedding cache keyed by model + text; length-sorted batching (+41%); metadata filters (`--type/--source/--after`); an optional ANN index measured against exact search; the 57,638-document FiQA suite | ✅ `week-05` |
| **6** | [Hybrid search](lessons/week06-hybrid-search.md) | A BM25 full-text index in the same LanceDB table; reciprocal rank fusion; `--mode vector\|keyword\|hybrid`; filters on both sides; measured on five suites (hybrid wins four, loses FiQA, so vector stays the default) | ✅ `week-06` |
| **7** | [Reranking and query transforms](lessons/week07-rerank-and-query-transforms.md) | A local cross-encoder and a listwise LLM reranker over the top 20; rewrite, multi-query and HyDE (cached); per-stage latency budgets; fallbacks counted and flagged. Reranking lifts recall@1 from 0.80 to 0.95 but fails its latency and SciFact conditions, so it ships off | ✅ `week-07` |
| **8** | [Grounded generation](lessons/week08-grounded-generation.md) | Context packing; lexical and LLM citation checks (report / strip / refuse); a per-mode refusal gate calibrated on held-out questions; prompt caching and token usage on every adapter. Finds that identical runs differ by 0.06 in correctness; every technique ships off, caching on | ✅ `week-08` |
| **9** | [Rich documents](lessons/week09-rich-documents.md) | PDF layout (headings by font size, ruled tables as Markdown, page labels); OCR for scanned pages (RapidOCR, offline); DOCX, PPTX with speaker notes, XLSX, CSV; page/slide/sheet locations in citations; a `rich` suite. Layout + OCR become the defaults (rich recall@4 0.875 → 1.000, answers 0.875 → 1.000) | ✅ `week-09` |
| **10** | [Images](lessons/week10-images.md) | Picture files and figures in PDFs, slides and Word files become `[Image: …]` blocks via OCR or a vision model (local or hosted, cached); optional CLIP picture search; an `images` suite with pre-set rules (not yet measured, so defaults stay off) | ✅ `week-10` |
| **11** | [Audio and video](lessons/week11-audio-video.md) | Local transcription (faster-whisper, cached); windows by time and speaker; subtitles (.srt/.vtt with voice tags); video keyframes read as pictures; `file @ mm:ss` citations; media URLs via yt-dlp, captions first; a `media` suite (not yet measured, so media stays off) | ✅ `week-11` |
| **12** | Production | Caching, streaming, concurrent ingest, cost accounting, traces, config profiles, final benchmark | coming |
| *Bonus* | Agentic and graph RAG | The model decides when to retrieve (multi-hop), linking back to [harnessy](https://github.com/krishankant/harnessy) | coming |

## Setup (about 30 minutes)

- [ ] Python 3.11+ and `uv`; then `uv sync`
- [ ] Ollama with `ollama pull qwen3.5:4b`, **or** an Anthropic or OpenAI key in `.env` (see `.env.example`)
- [ ] `uv run pytest`: all green
- [ ] `uv run raggy ingest samples/week01` then `uv run raggy ask "How long do refunds take?"`

**Assumed background:** you can read Python and run commands in a terminal. No machine-learning background is needed; the concept guides in [`docs/`](docs/README.md) cover the maths you need.
