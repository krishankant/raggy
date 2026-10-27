# Chunking: where to cut, and how to compare cuts fairly

**[← Docs hub](README.md)** · **[Week 4 lesson](../lessons/week04-chunking.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [04 Chunking](../concepts/04-chunking.md).

This guide sits beside week 4. The lesson walks through raggy's chunkers and the numbers. This page explains the ideas: why sizes are measured in tokens, what each splitting strategy does, how parent/child retrieval works, and how to compare strategies without fooling yourself. Numbers marked *measured* come from raggy's own code with `BAAI/bge-small-en-v1.5`.

---

## 1. A chunk is the unit of retrieval *and* of evidence

Retrieval returns chunks, not documents. Two things follow:

- **A fact that isn't wholly inside one retrieved chunk can't be fully used.** If a boundary falls between "On the first Thursday of each month" and "the workshop stays open until 21:00", neither half answers "when?".
- **A chunk's vector is an average of everything in it.** One idea per chunk gives a sharp vector. Five ideas give a blurred one that's a little similar to many questions and very similar to none.

So there is a tension. Small chunks retrieve precisely but can cut facts apart. Big chunks keep facts together but retrieve vaguely. Every technique in week 4 is a way to get both.

## 2. Measure size in tokens, not characters

An embedding model reads tokens, and it has a hard limit: bge-small reads 512, including its two special markers. **Anything past the limit is silently dropped.** The vector is computed from the first 512 tokens, and nothing tells you.

Characters are a poor proxy, because the ratio changes with the text. *Measured* with the model's tokenizer:

| Text | Characters | Tokens | Chars per token |
| --- | --- | --- | --- |
| `Refunds are paid back within 14 days.` | 37 | 9 | 4.11 |
| `\| BW-CH-1142 \| 8-speed chain, 116 links \| Heron Commuter \| 24 credits \|` | 71 | 23 | 3.09 |
| `def embed_text(chunk: Chunk) -> str:` | 36 | 15 | 2.40 |
| the 12 fictional docs | 866 – 1,217 each | 181 – 327 | 3.50 – 4.88 |

Part codes, numbers and code split into many word pieces. `BW-CH-1142` alone is 7 tokens: `b ##w - ch - 114 ##2`. A character limit that's safe for prose can be over the token limit for a table or a code listing. raggy sizes chunks with the model's own tokenizer (`Embedder.count_tokens`). It keeps 64 tokens free for the contextual header, and it refuses to embed anything over the limit instead of letting the model truncate it.

## 3. Three ways to cut

### Fixed windows (weeks 1–3)

Every *n* characters, with *m* characters of overlap. It's simple and predictable, and blind: it cuts mid-word, mid-sentence and mid-table-row.

### Recursive splitting

Cut at the biggest natural boundary first: paragraphs, then lines, then sentences, then words, then characters only as a last resort. Then merge neighbouring pieces back up to the size limit. Chunks end at sentence ends, but the splitter knows nothing about headings. A chunk can start in the middle of one section and end in the next.

### Structure-aware (Markdown)

Parse the document into **blocks** (headings, paragraphs, lists, tables, fenced code) and pack whole blocks:

- **A sibling or higher heading starts a new chunk.** One chunk never mixes "Hours" with "Staff discount". A deeper heading (a subsection) may share its parent section's chunk.
- **A table bigger than a chunk is split between rows.** The header row travels with every piece, in the chunk's context, so the row `| E07 | Charging interrupted | … |` still says which column is which.
- **Code is split between lines**, never inside one.
- **Every chunk knows its section path**: `Ebb E-Cargo owner's manual › 5. Battery and charging › Charging in the cold`.

Week 3's HTML extractor writes Markdown, so web pages get the same treatment as `.md` files.

### Overlap

Overlap repeats the end of one chunk at the start of the next, so a fact near a boundary appears whole in at least one chunk. raggy overlaps by whole units (sentences, rows, lines), never by half a sentence. The cost: overlapping text is embedded twice, and two overlapping chunks can both be retrieved and take two context slots for one passage. That's why week 2's metrics give no credit for a gold span found twice.

## 4. Contextual headers

A chunk that reads "Wait at least one hour before charging" has lost the words *battery*, *cold* and *Ebb E-Cargo*, which sit in the headings above it. raggy embeds the section path in front of the chunk:

```
Ebb E-Cargo owner's manual › 5. Battery and charging › Charging in the cold

The battery only accepts charge between 5 °C and 40 °C. …
```

The header is **embedded and shown to the answer model, but it isn't part of the chunk's span**. `chunk.text` is still exactly `document[start:end]`, so citations and the eval's gold spans point at real characters in the document.

## 5. Parent/child: search small, answer with more

Index small chunks (the *children*: precise vectors). At query time, swap each retrieved child for its *parent*, the bigger span around it, so the model reads the neighbouring text too. When two children of one parent are found, the parent is returned once.

Two details decide whether it works. raggy got both wrong first, and the evals caught both:

1. **Parents must not stop at section boundaries.** Their job is to add the neighbouring text a section-bound child cut off. In the staff handbook, the *what* ("fix it with our tools, at no cost") and the *when* ("first Thursday of each month") are in two sibling sections. A parent that stops at the heading is just the child's section again.
2. **Every child must lie wholly inside its parent.** If parents are chunked separately, their boundaries don't line up with the children's, and a child can straddle two parents. Retrieve that child and you get a parent holding only half of it. raggy builds parents from the children instead (consecutive children grouped up to the parent size), so containment holds by construction. `test_every_child_inside_its_parent` checks it on random documents.

## 6. Contextual retrieval (LLM-written context)

The header gives a chunk its section path. An LLM can go further: given the whole document and one chunk, write one or two sentences saying what the chunk is about in context. Anthropic published this as "contextual retrieval" (2024), and it helps most when chunks are small and documents are long and self-referential ("the unit", "this setting").

It costs one LLM call per chunk at ingest time. raggy makes it a switch (`RAGGY_CHUNK_CONTEXT=llm`) and caches each result by document hash, chunk span, model and prompt version, so a re-ingest pays only for chunks that changed. The prompt puts the document first and the chunk last, so providers that cache prompt prefixes can reuse the document part across all its chunks.

## 7. Comparing strategies without fooling yourself

The trap: **bigger chunks raise recall@k for free.** Four 1,000-token parents contain more text than four 100-token chunks, so of course they contain the answer more often. The model then has to read 4,000 tokens, which is slower and costlier, and a needle in more hay.

So raggy's eval records **`context_tokens`**: the tokens the answer model is given for each question. Then compare strategies two ways:

- **at the same k** (what `recall@4` reports), which favours big chunks, and
- **at the same token budget**: for each strategy, how many hits fit in ~800 tokens, and the recall at that k. This is the fair comparison when context is what you pay for.

And because recall is only a proxy, the final decision uses **answer correctness** from the judge, on a suite where chunking actually matters. A corpus of 200-token documents can't tell chunkers apart: most documents are one chunk under any strategy.

Finally, keep a benchmark you didn't write as a veto, with its limit set *before* the runs. In week 4, 128-token children looked best on raggy's own suites, and SciFact's "no more than 0.01 nDCG@10 lost" rule rejected them (−0.019). The 256-token setting that passed was better on answers anyway.

## Further reading

- Anthropic, "Introducing Contextual Retrieval" (2024): the method in section 6, with its retrieval-failure numbers.
- The Pinecone and LlamaIndex chunking guides: recursive, structure-aware and "small-to-big" (parent/child) retrieval, compared in prose. raggy's lesson compares them in numbers.
