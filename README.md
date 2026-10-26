# raggy

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#setup)
[![Models](https://img.shields.io/badge/answers-Ollama%20%7C%20Claude%20%7C%20OpenAI-6f42c1.svg)](lessons/week01-naive-rag.md#36-one-interface-three-model-providers)
[![Tests](https://img.shields.io/badge/tests-offline%20pytest-2ea44f.svg)](#commands)

**[Course Plan](COURSE_PLAN.md)** · **[Week 1: Naive RAG](lessons/week01-naive-rag.md)** · **[Week 3: Ingestion](lessons/week03-ingestion.md)** · **[Docs hub](docs/README.md)**

---

**raggy** is a retrieval-augmented generation (RAG) pipeline that grows one week at a time, from a roughly 770-line naive version to a system that answers questions over thousands of documents of any kind (text, web pages, PDFs, Office files, images, audio and video), with citations back to the exact source.

It's a course you **read and run**, not one you fill in. Every week adds working code and a lesson that explains what was added, why it's built that way and what it changed, with eval numbers from real runs. Each week is preserved on its own branch, so you can check out any stage of the build.

> **RAG in one sentence.** Instead of trusting what a language model memorised, *retrieve* the passages from your own documents that are relevant to a question, then have the model *generate* an answer from those passages only, citing them.

No RAG framework (no LangChain or LlamaIndex): only focused libraries and the official model SDKs, so every stage is code you can read.

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
| `uv run raggy search "query" [-k N]` | Show the chunks retrieval finds, with scores. Your first debugging step. |
| `uv run raggy ask "question" [-k N] [--show-context]` | Answer with numbered citations. |
| `uv run raggy sources` | List what's indexed: chunks, type, title, source, and which documents were skipped as duplicates. |
| `uv run raggy remove PATH_OR_URL` | Drop a source, or everything under a folder or URL, from the index. Never touches the files. |
| `uv run raggy eval [--suite fictional\|fictional-web\|scifact] [--retrieval-only] [--no-judge] [--limit N] [--label L]` | Measure retrieval and answer quality; saves a JSON run to `evals/runs/`. |
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

## The course

See **[COURSE_PLAN.md](COURSE_PLAN.md)** for all 12 weeks. Ready so far:

| Week | Lesson | Code |
| --- | --- | --- |
| 1 | [Naive RAG, end to end](lessons/week01-naive-rag.md) | `git checkout week-01` |
| 2 | [Evals before optimisation](lessons/week02-evals.md) | `git checkout week-02` |
| 3 | [Ingestion: files, folders, URLs](lessons/week03-ingestion.md) | `git checkout week-03` |

## Repository layout

```
raggy/            the package: ingest/ chunking/ embed/ store/ generate/ evals/ pipeline.py cli.py
lessons/          one lesson per week
docs/             concept guides (embeddings, evals, ingestion, and more each week)
samples/          small demo corpora (fictional, so models can't answer from memory)
evals/            eval suites (fictional corpus, its web-site version, questions), committed baselines; data/ and runs/ are local
tests/            offline pytest suite, one folder per week
```
