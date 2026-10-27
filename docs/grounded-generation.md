# Grounded generation: context, citations, refusals and caching

**[← Docs hub](README.md)** · **[Week 8 lesson](../lessons/week08-grounded-generation.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [11 Grounded generation](../concepts/11-grounded-generation.md).

This guide sits beside week 8. Retrieval decides which passages the model may use. This guide is about the step after: how the passages are laid out, how to check that each citation is honest, how to say "not in the documents" reliably, and when prompt caching saves money.

---

## 1. The context window is a layout problem

The model reads the passages in the order you give them, and it doesn't read every position equally well. Liu et al. (2023, "Lost in the Middle") found that with many passages, models use information at the start and end of the context better than in the middle. With raggy's default of 4 passages and about 1,000 tokens, the effect is small. With 20 passages and 10,000 tokens it isn't.

Packing does three things:

1. **Merge.** Two passages from the same document that overlap or touch are one passage. With parent expansion, two children of one parent arrive as the same parent twice; with overlap, neighbouring chunks repeat 32 tokens. Duplicates waste the budget and split one citation into two numbers.
2. **Order.** Sources by their best score, so the strongest evidence comes first. Within one source, by position, so a merged passage reads the way it was written.
3. **Budget.** Keep the best passages that fit `RAGGY_CONTEXT_TOKENS`. A single passage larger than the budget is cut at a sentence end, never dropped: an empty context is worse than a shortened one.

## 2. Checking citations

The model writes "Refunds take 14 days [2]". Is that sentence actually in passage 2? Week 2's judge answers this after the fact, for evals. A citation check answers it for every answer, before the user sees it.

### 2.1 Lexical support, by hand

Content words are the sentence's words minus stop words (*the, is, within, of*...). Numbers count as words. Support is the share of content words found in the cited passage:

> Sentence: "Refunds are paid within 14 days of the return arriving [2]."
> Content words: *refunds, paid, 14, days, return, arriving* (6)
> Passage 2: "...we pay refunds within 14 days of receiving your return..."
> Found: *refunds, 14, days, return* (4). Missing: *paid* (passage says "pay"), *arriving* ("receiving").
> Support = 4 / 6 = 0.67 ≥ 0.5 → **supported**.

Swap "14" for "30", and *30* is missing too: 3 / 6 = 0.5, at the edge. Cite a passage about shipping instead, and support drops near 0.

The weakness is visible in the example: "paid" against "pay", "arriving" against "receiving". Paraphrase lowers support even when the claim is right. In week 8's runs, both of the lexical check's flags were paraphrases the judge rated as supported.

### 2.2 An LLM check

An LLM reads each sentence beside its cited passage and says supported or not. It handles paraphrase and catches subtle contradictions. It costs a call per answer and adds latency, and it's a model grading a model, with the same noise as the judge (section 4).

### 2.3 What to do with an unsupported citation

| Action | Effect | When |
| --- | --- | --- |
| `report` | Keep the answer, list the markers | Logging, review queues, measuring the checker |
| `strip` | Remove the unsupported markers | When a missing citation is better than a wrong one |
| `refuse` | No supported citation left → the refusal sentence | High-stakes answers, where unsupported means unusable |

## 3. Saying "not in the documents"

Two ways to refuse:

- **The model refuses.** The prompt says to answer only from the passages and to use a fixed sentence otherwise. Week 2 measured this: refusal accuracy is high with a strong model.
- **A gate refuses.** If the best retrieved score is below a threshold, don't call the model at all. It's faster and free, but it's a single number deciding.

The gate's problem is overlap. Plot the top score of answerable questions and of unanswerable ones: the distributions overlap, because an unanswerable question is often *about something near* the documents. Any threshold trades wrong refusals of answerable questions against missed refusals of unanswerable ones.

### Calibrating without fooling yourself

If you choose the threshold on the same questions you measure it on, it will look perfect: you picked it to be. So split them:

1. On the calibration half, set the threshold to the lowest top score of any answerable question. That half loses no answerable question, by construction.
2. Measure on the held-out half: how many unanswerable questions are gated (good), and how many answerable ones are wrongly gated (bad).

In week 8: threshold 0.632, held out 1 of 4 unanswerable gated, and **1 of 17 answerable wrongly gated**. With the model already refusing all unanswerable questions, the gate only saves calls, at the cost of wrong refusals.

## 4. Noise: how much does a run move by itself?

An answer eval has two models in it: the one that answers and the judge. Both sample. In week 8, three runs that gave the model identical passages scored 0.879, 0.924 and 0.939 correctness on 33 questions, with six questions flipping between "correct" and "partial".

What that means for reading results:

- A difference under about 0.06 on a suite this size can be noise. Run twice before believing it.
- Retrieval metrics (recall@k, MRR, nDCG) are deterministic: the same index and question give the same ranking. That's why raggy's decision rules lean on them.
- For answer metrics, prefer large suites, repeated runs, or per-question agreement over the mean.

## 5. When prompt caching pays

Providers can cache the processed form of a prompt prefix and reuse it when the next request starts with the *byte-identical* prefix. Cached input is billed at a fraction of the normal rate and processed faster.

Three conditions:

1. **A long prefix that repeats.** Claude caches prefixes of at least 512 tokens (Opus 5.5); OpenAI from 1,024.
2. **The repeated part comes first.** A cache matches from the start of the prompt. Anything that varies (the question, the chunk) goes after it.
3. **Requests close together in time.** Entries expire after minutes.

| Workload | Repeated prefix | Pays? |
| --- | --- | --- |
| Contextual retrieval (week 4): one call per chunk, each with the whole document | The document | **Yes**: 67% of input tokens cached in week 8 |
| A long fixed system prompt on every question | The system prompt | Yes, if it's past the minimum |
| Answering questions | The retrieved passages | No: they differ per question |

Claude needs an explicit `cache_control` marker on the block to cache; OpenAI caches automatically. raggy's adapters take a `cache_prefix` argument and do the right thing for each.

## Further reading

- [Lost in the Middle](https://arxiv.org/abs/2307.03172), Liu et al. (2023).
- [ALCE: Enabling LLMs to Generate Text with Citations](https://arxiv.org/abs/2305.14627), Gao et al. (2023): measuring citation recall and precision.
- Anthropic, [Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching); OpenAI, [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
