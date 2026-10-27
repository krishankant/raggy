# Running RAG in production

**[← Docs hub](README.md)** · **[Week 12 lesson](../lessons/week12-production.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [14 RAG in production](../concepts/14-production.md).

This guide sits beside week 12. It covers the engineering that turns a pipeline that answers well into one you can run every day: caching, streaming, concurrency, cost, tracing and configuration. It also has a checklist for taking raggy, or any RAG system, to real users.

---

## 1. Caching: every layer, keyed on its inputs

raggy now caches at five layers, and every key is built the same way: **everything that determines the output, and nothing else.**

| Cache | Key | Saves |
| --- | --- | --- |
| Embeddings (week 5) | model + exact text | Re-embedding unchanged chunks |
| Contextual retrieval (week 4) | document hash + chunk span + model + prompt version | One LLM call per chunk |
| Query transforms (week 7) | model + prompt version + question | One LLM call per question |
| Image text (week 10) | image bytes + mode + model + prompt version | OCR or a vision call per picture |
| Transcripts (week 11) | audio bytes + model + version | Transcribing again |
| Answers (week 12) | question + passages' ids and text + model + prompt + check settings | The answer call |

A key that includes everything that matters never needs manual invalidation: change any input and you miss. A key that leaves something out serves stale results silently. The classic mistake is keying answers by question alone: after a document changes, the cache keeps serving the old answer.

The provider's prompt cache (week 8) is a different thing: it makes a *call* cheaper when its prefix repeats, but the call still happens.

## 2. Streaming

Perceived latency is time to the first word, not time to the last. Streaming moves the first word from "after generation" (2–5 s) to "after retrieval and the first token" (a few hundred ms).

What streaming changes:

- **Post-processing waits for the end.** Citation parsing and citation checks need the whole text. A check that changes the text can only correct what the user already saw, so show the correction clearly.
- **Errors arrive mid-answer.** A connection drop after half an answer must be shown as an error, not as a short answer.
- **Usage arrives last.** Token counts come in the final event (OpenAI needs `stream_options={"include_usage": true}`).

## 3. Concurrency: parallelise reads, serialise writes

Ingest has three phases: load (parse, OCR, describe, transcribe), embed and write. Loading is independent per file and usually the slowest, so it parallelises well. Writing to the registry and the vector store is ordered and stateful, so keep it on one thread. raggy loads in a thread pool and consumes the results in the original order, so a parallel ingest gives byte-identical results to a serial one.

Threads work here because the heavy libraries (pdfplumber's parser, ONNX Runtime, CTranslate2, network I/O) release Python's GIL. Shared resources need care: an SQLite connection used from several threads needs a lock, and a model loaded lazily by two threads at once may load twice.

Embedding is already batched (week 5): one model call per 16 texts beats many concurrent single calls on a CPU.

## 4. Cost

Cost per answer = Σ over calls of (uncached input × input price + cached input × cached price + output × output price).

Where the calls are in raggy:

| Stage | Calls | When |
| --- | --- | --- |
| Contextual retrieval | 1 per chunk | Ingest, cached |
| Vision descriptions | 1 per picture | Ingest, cached |
| Query transform | 1 per question | Query, cached |
| LLM reranker | 1 per question | Query |
| Answer | 1 per question | Query, cached for repeats |
| LLM citation check | 1 per answer | Query |
| Judge (evals only) | 2 per question | Eval |

Ingest costs are paid once per document, and query costs once per question. For a fixed corpus asked many questions, query cost dominates; for a large, changing corpus asked few questions, ingest does. Set `RAGGY_PRICES` and let `raggy ask` print what each answer cost, instead of estimating.

## 5. What a trace must hold

A trace exists to answer "why did it say that?" later. It needs:

- **The input:** the question and the settings (retrieval mode, reranker, models, top k).
- **The evidence:** what was retrieved, with sources, locations and scores. This separates a *retrieval* failure (the right passage wasn't there) from a *generation* failure (it was, and the answer ignored it).
- **The output:** the answer, its citations, and any unsupported markers.
- **The costs:** stage timings, token usage, money, and whether it was cached.

Traces store user questions, so treat them as user data: local by default, off unless needed, and with a retention policy if you keep them.

## 6. Profiles and defaults

raggy's defaults are each week's measured, rule-checked choices: the safest settings for collections like the eval suites. Profiles are named starting points for other priorities:

- `local`: privacy first; nothing leaves the machine.
- `fast`: latency first.
- `accurate`: accuracy first, paying the measured latency costs.

A profile fills only the settings you haven't set, so you can start from one and override any part.

## 7. A checklist before real users

1. **Build an eval set from your own documents** (week 2): 30+ questions with gold spans, including unanswerable ones. Without it, every other step is a guess.
2. **Pick settings by measurement:** run `raggy eval` per candidate setting, or the benchmark per profile, and write the decision rule before the run.
3. **Run answer evals at least twice** (week 8): differences under the noise aren't real.
4. **Check refusals:** an unanswerable question must get "not in the documents", not a guess.
5. **Set prices and watch cost per answer.**
6. **Turn on traces for a pilot**, and read the failures: retrieval or generation?
7. **Re-ingest on a schedule:** the registry (week 3) re-reads only what changed.
8. **Re-run the eval after every change**: a model update, a prompt edit, a new document type.

## Further reading

- Anthropic, [Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) and [Streaming messages](https://docs.claude.com/en/docs/build-with-claude/streaming).
- OpenAI, [Streaming](https://platform.openai.com/docs/api-reference/streaming).
- Python, [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) and the GIL's interaction with C extensions.
