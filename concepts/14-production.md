# 14 · RAG in production

**[← 13 Multimodal RAG](13-multimodal-rag.md)** · **[Concepts](README.md)** · **Next: [15 Agentic and graph RAG →](15-agentic-and-graph-rag.md)** · **In the course:** [Week 12](../lessons/week12-production.md)

A RAG system that answers well on a laptop still has to answer **fast**, **cheaply**, **reliably** and **safely** for real users. This page covers the ideas that matter once people depend on it.

---

## 1. Where the time goes

A typical request, with illustrative timings:

```text
 0 ms ┬ question arrives
      ├─ embed question ............... 10 ms
      ├─ vector + keyword search ...... 20 ms
      ├─ rerank top 30 ................ 400 ms
      ├─ build prompt ................. 1 ms
      ├─ LLM: time to first token ..... 600 ms    ← the user sees text from here (if streaming)
      └─ LLM: rest of the answer ...... 2,500 ms
3,531 ms ┴ done
```

The LLM usually dominates. Two numbers matter to the user:

- **Time to first token (TTFT)**: how long before *anything* appears.
- **Total time**: how long until the answer is complete.

Report **p50 and p95** (the median, and the time 95% of requests beat), not the average: a few slow requests hide in an average, and users remember them.

## 2. Streaming

Since an LLM writes token by token ([02](02-llm-basics.md#3-how-generation-works)), you can show tokens as they arrive. Total time doesn't change, but **perceived** latency drops from seconds to well under a second.

```text
Without streaming:  [........ 3.5 s of nothing ........] full answer
With streaming:     [0.6 s] The battery is covered for two years [1]...  (keeps typing)
```

Catch: any check that needs the *whole* answer (citation verification) must run after the stream ends, or the UI must update afterwards.

## 3. Caching at every layer

| Layer | Key | Saves | Invalidate when |
| --- | --- | --- | --- |
| **Embedding cache** | hash(model, text) | Re-embedding unchanged chunks on re-ingest | Model changes |
| **Parse/vision/transcript cache** | hash(file or image bytes) | Expensive OCR, vision, ASR calls | Extractor version changes |
| **Query transform cache** | (model, prompt version, question) | Rewrite/HyDE calls | Prompt changes |
| **Retrieval cache** | (question, index version, settings) | Search + rerank | Index changes |
| **Answer cache** | (question, retrieved chunk ids, model, prompt version) | The whole LLM call | Anything in the key changes |
| **Provider prompt caching** | The provider caches a long, repeated prompt prefix | Input cost and TTFT on the cached part | Automatically, after minutes of disuse |

**Keys must include versions.** An answer cached under an old prompt or old index is a stale answer. Keying the answer cache by the *retrieved chunk ids* means a document update that changes what's retrieved automatically misses the cache.

**Semantic caching** matches *similar* questions ("refund time?" ≈ "how long do refunds take?") by embedding the question. Higher hit rates, but risk: "Is the Ebb covered?" and "Is the Kestrel covered?" are very similar strings with different answers. Use a strict threshold, or don't use it where small wording changes matter.

## 4. Cost per answer

Cost = (input tokens × input price) + (output tokens × output price), summed over every LLM call in the request: transforms, reranking, the answer, checks.

Worked example with illustrative prices ($3 / M input, $15 / M output):

| Call | Input tokens | Output tokens | Cost |
| --- | --- | --- | --- |
| Query rewrite | 200 | 20 | $0.0009 |
| LLM rerank of 20 passages | 4,000 | 50 | $0.0128 |
| Answer (5 passages) | 2,500 | 250 | $0.0113 |
| **Total** | | | **$0.0249** |

The LLM reranker costs more than the answer. A local cross-encoder ([09](09-reranking.md)) would cost $0 and about 0.4 s instead. Track cost per request and per stage, the same way as latency.

Levers, roughly in order of effect: fewer LLM calls per request; fewer/shorter passages; smaller models for side tasks (rewriting, judging); caching; batch APIs for offline work (evals, ingestion).

## 5. Ingestion at scale

- **Incremental re-ingest:** hash each file; re-process only new or changed files; delete the chunks of removed files.
- **Parallelism:** parsing, OCR and embedding are independent per file, so run them concurrently, but write to the index from one place (or use an index that supports concurrent writes).
- **Batching:** embedding 64 chunks per call is far faster than one at a time.
- **Backfills:** changing the embedding model means re-embedding everything; plan for it as a migration, building the new index beside the old one.

## 6. Observability: tracing

When an answer is wrong, you need to see *why*. A **trace** records, for each request:

```text
trace 7f3a…
  question:        "How long is the battery covered?"
  transform:       rewrite → "battery warranty period"         620 ms
  retrieval:       top 20 ids [c17, c03, c44, ...], scores       18 ms
  rerank:          top 5 ids  [c03, c17, ...], scores           410 ms
  prompt:          2,480 tokens, prompt version answer-v3
  answer:          "...two years [1]."  250 tokens             2,900 ms
  checks:          citations 1/1 supported
  cost:            $0.0113    cache: miss
  settings:        model, top_k, chunker, index version
```

With traces you can answer "was it retrieval or generation?" for any bad answer, replay it, and turn it into a test case ([12](12-evaluation.md)). Tools: OpenTelemetry-based tracers, Langfuse, LangSmith, Arize Phoenix, or a JSON line per request.

## 7. Security and permissions

| Risk | Example | Defence |
| --- | --- | --- |
| **Leaking documents across users** | An intern's question retrieves an HR salary file | Store access-control lists with each chunk; **filter at retrieval time** by the asking user's permissions. Never rely on the LLM to withhold text it was given |
| **Prompt injection** in documents | A web page says "ignore your instructions..." | Treat passages as data; no dangerous tools; vet sources ([11](11-grounded-generation.md#8-prompt-injection-through-documents)) |
| **Sensitive data in logs and caches** | Traces store questions with personal data | Redact, restrict access, set retention |
| **Stale or deleted content** | A withdrawn policy still answers | Deletion removes chunks *and* cache entries |

## 8. Reliability

- **Timeouts and retries** with backoff on every external call.
- **Fallbacks**: if the reranker fails, use retrieval order; if the main model is down, use a backup model, or refuse cleanly. A failed improvement must never become a failed answer.
- **Rate limits**: queue or shed load before the provider does it for you.
- **Configuration profiles**: named presets ("fast", "accurate", "local") instead of a dozen loose settings, so deployments are reproducible.

## 9. A checklist before real users

- [ ] Test set with answerable *and* unanswerable questions, run on every change
- [ ] p95 latency and cost per answer measured per stage
- [ ] Streaming on
- [ ] Caches keyed by versions; deletion clears caches
- [ ] Permissions enforced in retrieval, not in the prompt
- [ ] A trace for every request
- [ ] Fallbacks for every optional stage
- [ ] A feedback button, with feedback reviewed and turned into test cases

## 10. In raggy

- [Week 12](../lessons/week12-production.md) and the [production guide](../docs/production.md): streaming (`on_token`), answer cache (`raggy/generate/answer_cache.py`), cost per answer (`raggy/costs.py`), traces (`raggy/trace.py`, `raggy trace`), concurrent ingest, and profiles (`RAGGY_PROFILE`).

## 11. Further reading

| Read | Why |
| --- | --- |
| [Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) (Anthropic docs) | How provider prompt caching works and what it saves |
| [Streaming messages](https://docs.anthropic.com/en/docs/build-with-claude/streaming) (Anthropic docs) | Streaming in practice |
| [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/), Eugene Yan | Caching, guardrails, evals and feedback in production |
| [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) | Security risks including prompt injection and data leakage |
| [GPTCache](https://github.com/zilliztech/GPTCache) | A semantic cache, and a good way to see its trade-offs |
| [Langfuse documentation](https://langfuse.com/docs) | An open-source tracing tool: what a RAG trace looks like |
