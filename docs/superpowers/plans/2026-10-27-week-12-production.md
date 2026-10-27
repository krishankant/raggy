# Week 12: Production — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make raggy pleasant and safe to run every day: answers that stream, repeated questions that cost nothing, faster ingest of big folders, a cost for every call, a trace of every answer, named configuration profiles, and one benchmark command that measures the whole course.

**Architecture:**
- **Streaming:** each adapter gets `stream(system, user) -> Iterator[str]`; `generate_answer(on_token=)` uses it when given a callback. Citations are parsed and checked after the last token.
- **Answer cache:** `AnswerCache` (SQLite, `answer-cache.sqlite`) is keyed by normalised question + the passages' chunk ids and text hash + model + a hash of the system prompt + check settings. A changed index or prompt is a miss by construction. Eval runs never use it.
- **Concurrent ingest:** `_sync` stat-checks files in the main thread, loads the changed ones in a thread pool (`RAGGY_INGEST_WORKERS`, 4), and keeps registry and store writes on the main thread. The vision and transcript caches take a lock.
- **Cost:** `raggy/costs.py` has prices per million tokens from `RAGGY_PRICES` (no built-in price list, because prices change). `cost(usage, model)` returns None when the price is unknown. Ask and eval report tokens, plus cost when a price is set.
- **Traces:** `Tracer` appends one JSON line per `ask` to `index_dir/traces/YYYY-MM-DD.jsonl` when `RAGGY_TRACE=on`: question, settings, stage timings, hits (source, location, score), answer, citations, unsupported markers, usage, cost, cache hit. `raggy trace [-n N]` prints the latest.
- **Profiles:** `RAGGY_PROFILE=local|fast|accurate` fills in defaults for settings the environment doesn't set. An explicit setting always wins.
- **Benchmark:** `evals/benchmark.py` runs every suite retrieval-only under the default and `accurate` profiles and writes one Markdown table.

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 12 row).

## Global Constraints

- No new dependencies; tests are offline.
- `RAGGY_ANSWER_CACHE` defaults to on. A cached answer is the same model's output for byte-identical passages and question, so accuracy can't change by construction. `RAGGY_TRACE` defaults to off, because traces store questions. `RAGGY_INGEST_WORKERS` is 4. `RAGGY_PROFILE` is unset.
- Per the user's instruction this week, the benchmark is written but not run. The lesson gives the command.

## Review Focus

1. A streamed answer whose citation check strips a marker: the printed text already showed it, so the CLI prints the correction after the stream.
2. Two files loaded concurrently that share an image: the cache lock prevents an SQLite "recursive use" error.
3. A profile that names a setting the user also set: the user's value wins (test).
4. An unknown model in `RAGGY_PRICES`, or a malformed entry: unknown → no cost shown; malformed → a `ConfigError` naming the entry.
5. A trace directory that can't be written: the answer is still returned, and the CLI prints one warning.

### Task 1: streaming, answer cache, costs, traces (TDD)
### Task 2: concurrent ingest, profiles, benchmark script (TDD)
### Task 3: lesson, guide, docs; release
