# Week 12: Production — streaming, caching, concurrency, cost, traces, profiles

**[← Week 11: Audio and video](week11-audio-video.md)** · **[Course Plan](../COURSE_PLAN.md)** · **[Running RAG in production (guide)](../docs/production.md)** · **[Bonus: Agentic and graph RAG →](bonus-agentic-and-graph-rag.md)**

---

**Time:** about 4 hours (1 reading, 1.5 reading and running the code, 1.5 on the benchmark and experiments).
**Code for this week:** `git checkout week-12`
**Companion guide:** [`docs/production.md`](../docs/production.md): what to cache and how to key it, where concurrency is safe, cost per answer, what a trace must hold, and how to choose a profile.

**You're done when:**

1. `uv run pytest` passes (offline, no keys needed),
2. you've run `raggy ask "..." --stream` twice and seen the second answer come from the cache, and
3. you've run `uv run python evals/benchmark.py` and can read the course's whole story in its table.

## 1. Read first

| Read | Look for |
| --- | --- |
| [`docs/production.md`](../docs/production.md) (this repo) | Why the answer cache's key includes the passages' text, not just the question. Why only *loading* runs in threads. |
| Your own `raggy eval` runs from weeks 2–11 | Which settings won on your documents. This week turns those choices into profiles. |

## 2. From a course project to a tool you run every day

Eleven weeks built the pipeline: ingest anything, chunk it well, retrieve with vectors and keywords, rerank, answer with checked citations, and read pictures and speech. This week adds what makes it pleasant and safe to use daily:

| Need | This week |
| --- | --- |
| Answers feel slow | **Streaming**: words appear as they're written |
| The same questions come in again and again | **Answer cache**: a repeat over identical passages costs nothing |
| A folder of 5,000 PDFs takes an afternoon | **Concurrent ingest**: files load in parallel |
| "What did that cost?" | **Cost accounting**: tokens, and money when you set prices |
| "Why did it say that?" | **Traces**: one JSON line per answer, with everything that shaped it |
| Forty settings | **Profiles**: `local`, `fast`, `accurate` |
| "Is it any good?" | **The benchmark**: every suite, every profile, one table |

## 3. What you're building this week

| File | Lines | What it does |
| --- | --- | --- |
| `generate/*_llm.py`, `generate/answer.py` | +60 | `stream(system, user)` on each adapter; `generate_answer(on_token=)`; citations parsed and checked after the last token |
| `generate/answer_cache.py` | 49 | `AnswerCache` and `answer_key`: normalised question, each passage's id and text, model, system prompt, check settings |
| `pipeline.py` | +80 | `ask(on_token=)` with timings, usage, cost and an optional trace; `answer_from(use_cache=)`; ingest loads changed files in a thread pool |
| `costs.py` | 39 | `RAGGY_PRICES` parsing, `cost(usage, model)` |
| `trace.py` | 54 | `Tracer`: daily JSONL files under `index_dir/traces/`; `raggy trace` |
| `config.py` | +50 | `RAGGY_PROFILE` (`local`, `fast`, `accurate`), `RAGGY_ANSWER_CACHE`, `RAGGY_TRACE`, `RAGGY_INGEST_WORKERS`, `RAGGY_PRICES` |
| `evals/benchmark.py` | 77 | Every suite under each profile, into `evals/baselines/week12/benchmark.md` |

No new dependencies.

## 4. Walkthrough

### 4.1 Streaming

An answer takes 2–5 s to generate, and the first words exist after a few hundred milliseconds. Streaming shows them as they arrive. Each adapter's `stream()` yields text pieces: OpenAI and Ollama through `stream=True` with `include_usage`, Claude through the SDK's `messages.stream()` helper. Token usage is counted when the stream ends.

Citations are parsed after the last piece, because `[2]` can arrive split as `[` and `2]`. A citation check (week 8) also runs after the stream. If `strip` changed the text, the user has already seen the original, so `raggy ask --stream` prints the corrected version underneath. A stream can't be unsaid; it can only be corrected.

### 4.2 The answer cache

A repeated question over the same passages, with the same model and prompt, gives a fresh answer that differs only by sampling noise (week 8 measured that noise). Serving the stored one costs nothing and is consistent. The key:

```text
sha256( normalised question
      + for each passage: chunk id + the exact text the model saw
      + model name + citation-check settings
      + sha256(system prompt) )
```

Everything that could change the answer is in the key, so there's nothing to invalidate by hand. Re-ingest a document with new text, change the model, edit the prompt: each is a miss. That's why it's on by default (`RAGGY_ANSWER_CACHE=on`), unlike the techniques of weeks 6–11: it can't change what a fresh answer would have been, only skip paying for it.

Two exceptions:

- **Evals never use it** (`answer_from(use_cache=False)`). A cached answer would hide exactly the run-to-run noise that week 8 found.
- **Failed citation checks aren't cached**, so the next ask retries the check.

### 4.3 Concurrent ingest

Loading is the slow part of a big ingest: PDF layout, OCR, office parsing, image reading, transcription. It's independent per file, so `_sync` stat-checks every file first and submits the changed ones to a thread pool (`RAGGY_INGEST_WORKERS`, 4). It then collects them in the original order. Everything that writes (the registry, the vector store, the embedding batches) stays on the main thread, so the result is identical to a serial ingest (`test_concurrent_ingest_matches_serial`).

Threads, not processes: most of the heavy work (pdfplumber's C parts, ONNX for OCR and CLIP, CTranslate2 for Whisper, network calls for vision) releases Python's GIL. The vision and transcript caches now take a lock, because two files can share an image.

### 4.4 Cost

Every adapter has counted tokens since week 8. `cost(usage, model, prices)` turns them into money, with prices you set:

```bash
RAGGY_PRICES="openai:gpt-5.5=1.25/10/0.125, anthropic:claude-opus-5-5=5/25/0.5"
#              provider:model = input / output / cached input, per million tokens
```

There's deliberately no built-in price list. Prices change, and a stale number printed as a fact is worse than none, so an unknown model shows tokens only. `raggy ask` prints `Tokens: 1214 in (0 cached), 58 out, 0.0021` after each answer.

### 4.5 Traces

With `RAGGY_TRACE=on`, every `ask` appends one line to `index_dir/traces/2026-10-27.jsonl`:

```json
{"time": "...", "question": "How long do refunds take?",
 "settings": {"retrieval": "vector", "rerank": "none", "top_k": 4, "llm": "openai:gpt-5.5", ...},
 "timings": {"retrieve": 7.1, "generate": 1840.2, "total": 1851.0},
 "hits": [{"source": ".../returns-and-refunds.md", "location": null, "score": 0.83, "chunk": "..."}],
 "answer": "Refunds are paid within 14 days ... [1]", "citations": [1], "unsupported": [],
 "cached": false, "usage": {"calls": 1, "input": 1214, "output": 58, ...}, "cost": 0.0021}
```

That's enough to answer "why did it say that?" a week later: what was retrieved, with what scores, under which settings. `raggy trace -n 5` prints the latest. Traces store questions, so tracing is off by default. If the folder can't be written, the answer is still returned with one warning: a record should never cost the user their answer.

### 4.6 Profiles

Forty settings are too many to choose from cold. A profile fills in the settings you haven't set:

| `RAGGY_PROFILE` | Sets | From |
| --- | --- | --- |
| `local` | Ollama for answers and judging, OCR for pictures, local transcription | Everything on your machine; nothing leaves it |
| `fast` | vector search, no reranker, no transform, no check, no packing | The lowest latency: no extra model call or ranker |
| `accurate` | hybrid search, local reranker, packing, a lexical citation check (`report`), OCR, transcription | The measured accuracy wins, paying their latency: hybrid won 4 of 5 suites (week 6), the reranker lifted recall@1 from 0.80 to 0.95 (week 7) |

An explicit setting always wins: `RAGGY_PROFILE=accurate RAGGY_RERANK=none` is `accurate` without the reranker. No profile is the default; the defaults are each week's measured, rule-checked choices.

## 5. The benchmark

```bash
uv run python evals/benchmark.py                          # default vs accurate, retrieval only
uv run python evals/benchmark.py --profiles fast accurate --answers   # with answers (paid calls)
```

It runs every suite under each profile and writes `evals/baselines/week12/benchmark.md`:

| suite | default | accurate |
| --- | --- | --- |
| fictional | recall@1 / recall@4 / MRR (p95) | ... |
| ... | | |

Per the brief for this final release, the benchmark is written but wasn't run here. The table is yours to produce on your own machine. What earlier weeks' runs predict for it:

- **fictional, fictional-web, manual:** `accurate` should lift recall@1 most (week 7's reranker), with recall@4 already near 1.000.
- **rich:** identical; layout and OCR are already the defaults (week 9).
- **images, media:** zero under the default (images and media are off), and non-zero under `accurate`: those suites exist to measure exactly that switch.
- **SciFact:** hybrid +0.012 nDCG@10 (week 6), reranker about ±0 (week 7).
- **FiQA:** hybrid −0.026 (week 6). That's `accurate`'s known cost, and the reason it isn't the default.
- **Latency:** `accurate` adds about 0.4–0.7 s of reranking per question on an 8 GB laptop.

## 6. Why each choice

| Choice | Why | What it costs |
| --- | --- | --- |
| Stream, then check | Fast first words; citation parsing needs the whole text | A stripped citation is corrected after it was shown |
| Answer cache keyed on passage text | Correct by construction: no manual invalidation | A tiny edit to a passage re-pays the answer |
| Cache on by default, off for evals | It can't change a fresh answer's quality; evals measure noise | Repeats don't sample new answers |
| Threads for loading only | Heavy work releases the GIL; writes stay serial and ordered | Pure-Python parsing gains little |
| No built-in prices | Prices change; a stale fact misleads | You set `RAGGY_PRICES` yourself |
| Traces off by default, local files | They store questions | Turn on to debug |
| Profiles fill gaps, never override | Your explicit choice always wins | A profile can be partly overridden without notice |

## 7. Experiments and notes

Write your answers in `NOTES.md`.

1. `raggy ask "How long do refunds take?" --stream --timings`, twice. Compare the two timings lines. What does the second run still pay for?
2. Ingest a large folder with `RAGGY_INGEST_WORKERS=1` and then `8` (`raggy ingest --full`). Where does the speed-up stop, and why?
3. Set `RAGGY_PRICES` for your model and ask ten questions. Estimate the cost of 10,000 questions a month, with and without `RAGGY_CHUNK_CONTEXT=llm` at ingest.
4. Turn on `RAGGY_TRACE`, ask a question the system gets wrong, and use `raggy trace -n 1` to decide: was it retrieval (the right chunk wasn't in `hits`) or generation (it was, and the answer ignored it)?
5. Run the benchmark. Write your own profile for your own documents: which settings, and which week's evidence supports each?

## 8. Where to go next

The bonus week (agentic and graph RAG) lets the model decide when and what to retrieve, which helps questions that need several hops. The course plan has the outline. The habits of the twelve weeks carry over: write the rule before the run, and believe the measurement over the intuition.

---

**Tested with:** Python 3.13.14 on an Apple M2 with 8 GB of RAM.
