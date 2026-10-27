# Agentic and graph RAG: answering questions that need more than one search

**[← Docs hub](README.md)** · **[Bonus lesson](../lessons/bonus-agentic-and-graph-rag.md)**

This guide sits beside the bonus week. It explains why some questions can't be answered by any single retrieval, describes the two families of fixes (search again at question time, or connect facts at index time), and shows how each would fit into raggy.

---

## 1. Single-hop and multi-hop questions

A **single-hop** question is answered by one passage that resembles it:

> "How long do refunds take?" → *"Refunds are paid within 14 days…"*

A **multi-hop** question needs several passages, where a later one is only findable through an earlier one:

> "Which partner centre can work on the motor of the bike whose box seats two children?"
> hop 1: "the bike whose box seats two children" → *bike-models.md: the Ebb E-Cargo… a box that seats two children*
> hop 2: "who works on the Ebb E-Cargo's motor" → *service-centres.md: Saltmarsh… works on the Ebb E-Cargo's motor*

The connecting name, "Ebb E-Cargo", is a **bridge entity**. It isn't in the question; it's in the *answer* to hop 1. A retriever scores passages by similarity to the question, and passage 2 isn't similar to the question, only to hop 1's answer. Better embeddings, hybrid search and reranking can't fix this, because they all rank by similarity to the same query.

Common multi-hop shapes:

| Shape | Example | Bridge |
| --- | --- | --- |
| Description → name → fact | "the bike that weighs 38 kg" → Ebb → battery warranty | a product |
| Fact → holder → another fact | "who fits the BW-BT-5000" → Saltmarsh → phone number | a company |
| Comparison | "is the cheapest bike's light set more than 30 credits?" | a superlative over a list |
| Chain of 3+ | assembled in Fenwick → Ebb → battery → fitter → opening hours | several |

## 2. Fix 1: search again (agentic retrieval)

Let a model read what the first search found and write the next search. This is the ReAct pattern (reason, act, observe) with search as the only action:

```text
1. search(question)
2. loop:
     show the model: question, searches so far, passages so far
     model decides: search(q) | answer | not_found
3. answer from all passages gathered
```

### Design choices, and raggy's picks

| Choice | Options | raggy |
| --- | --- | --- |
| How decisions come back | Native tool calls; JSON in text | JSON in text: one loop on every provider |
| First step | Ask the model first; search the question first | Search first: most questions need one hop |
| What the model sees | Passage ids only; full passages; truncated passages | Truncated (700 chars): enough to spot the bridge |
| When to stop | Model says so; step limit; no new passages; repeated query | All four, each recorded |
| Final answer | The agent writes it; the normal answer path | The normal path: citations and checks unchanged |
| Context | Last search only; everything gathered | Everything gathered, de-duplicated, ≤ 12 passages |

### Costs

Per question, the agent adds one decision call, plus one per extra search. A decision prompt holds the passages found so far, so it grows with each step. For a single-hop question, the cost is one short "answer" decision. Measure it: `raggy ask --agent --timings`, with `RAGGY_PRICES` set.

### Failure modes to watch

- **Searching past the answer.** The model keeps searching for confirmation. The step limit caps the cost; the prompt should say that "enough" means enough.
- **Answering too early.** It answers after hop 1 and guesses hop 2. The citation check (week 8) catches unsupported claims.
- **Drift on unanswerable questions.** It searches until something looks related, then answers from that. Unanswerable multi-hop questions are the test.
- **Comparing unfairly.** The agent's context is larger. Compare it with single-pass at the same number of passages (`RAGGY_TOP_K=8`), or you'll credit the agent for what is really more context.

## 3. Fix 2: connect facts in advance (graph RAG)

Instead of finding the bridge at question time, record it at index time.

### 3.1 The full GraphRAG pipeline (Edge et al., 2024)

1. **Extract.** An LLM reads each chunk and lists entities (with types and descriptions) and relations between them.
2. **Build.** Merge duplicate entities across chunks into one graph.
3. **Cluster.** Split the graph into communities of densely connected entities (the Leiden algorithm), at several levels.
4. **Summarise.** An LLM writes a report for each community.
5. **Query.**
   - **Local search:** find the entities in the question, collect their neighbourhoods and source chunks, and answer from those.
   - **Global search:** map the question over community reports, then reduce the partial answers. This handles "what are the main themes?" questions that no set of passages answers.

### 3.2 What a minimal version would look like in raggy

A small, cheap variant (local search only) would fit beside the existing index:

| Step | In raggy |
| --- | --- |
| Entity extraction | Per chunk, cached like contextual retrieval (week 4): `(chunk id, entity)` rows. Or a free heuristic: capitalised phrases and product codes |
| Graph | An SQLite table of entity ↔ chunk links; two entities are neighbours when they share a chunk |
| Query | Retrieve as usual; take the entities in the top chunks; add chunks that share those entities (one hop); rerank |

On the m01 example: the top chunk (bike-models, Ebb) carries the entity "Ebb E-Cargo"; the graph links it to the service-centres chunk that also names "Ebb E-Cargo"; that chunk joins the context, without an LLM call at question time.

### 3.3 When a graph is worth building

| Your questions | Better fit |
| --- | --- |
| Mostly single-hop | Neither: plain RAG (weeks 1–12) |
| Some 2–3 hop, over a corpus that changes | Agentic retrieval: nothing to rebuild |
| Many hops over stable, entity-rich data (contracts, org charts, product catalogues) | A graph: pay once, then every question is cheap |
| "Global" or summary questions over the whole corpus | GraphRAG's community summaries |

## 4. The two together

They combine well. An agent whose tool set includes "neighbours of entity X" as well as "search text" can follow a graph edge when there is one, and search when there isn't. That's where both research lines are heading: retrieval as a set of tools, with the model choosing between them.

## Further reading

- [ReAct](https://arxiv.org/abs/2210.03629), Yao et al. (2022).
- [IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning](https://arxiv.org/abs/2212.10509), Trivedi et al. (2022).
- [Self-Ask: Measuring and Narrowing the Compositionality Gap](https://arxiv.org/abs/2210.03350), Press et al. (2022).
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), Edge et al. (2024).
- [HotpotQA](https://arxiv.org/abs/1809.09600), Yang et al. (2018): the standard multi-hop QA benchmark, with bridge and comparison questions.
- [harnessy](https://github.com/krishankant/harnessy): the agent-harness course this loop comes from (weeks 2–3: the loop and tools).
