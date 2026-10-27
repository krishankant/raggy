# Embeddings and vector search

**[← Course Plan](../COURSE_PLAN.md)** · **[Week 1 lesson](../lessons/week01-naive-rag.md)** · **[Docs hub](README.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [05 Embeddings](../concepts/05-embeddings.md) · [06 Vector similarity and search](../concepts/06-vector-search.md).

---

Retrieval in raggy rests on one idea: **turn text into a direction in space, so that "similar meaning" becomes "pointing the same way"**. This guide explains that idea without assuming any machine-learning background.

## 1. What an embedding is

An embedding model reads a piece of text and outputs a fixed-length list of numbers, a **vector**. raggy's default model, `BAAI/bge-small-en-v1.5`, outputs 384 numbers for any text, whether it's three words or a whole paragraph.

The numbers mean nothing individually. What matters is the training: the model was trained on huge numbers of (question, relevant passage) and (sentence, paraphrase) pairs, and pushed so that related texts produce vectors pointing in similar directions and unrelated texts point elsewhere. After training, "refund" and "money back" land near each other even though they share no letters.

```
"how long do refunds take?"          → [-0.074, -0.022,  0.048, … 381 more ]
"Refunds are issued within 14 days." → [-0.093, -0.042, -0.003, … ]
"The office cat is called Biscuit."  → [ 0.012, -0.034, -0.002, … ]
```

*(First three of 384 values from a real run. You can't see similarity by eye in three numbers out of 384; the cosine scores in section 3 are where it shows.)*

## 2. Measuring "pointing the same way": cosine similarity

Take two vectors **q** and **v**. Their **cosine similarity** is the cosine of the angle between them:

```
cos(q, v) = (q · v) / (|q| × |v|)
```

- `q · v` (the dot product) multiplies the vectors element by element and adds the results.
- `|q|` is a vector's length: the square root of the sum of its squared elements.
- The result runs from 1 (same direction) through 0 (unrelated, at right angles) to −1 (opposite).

**A worked example in 2 dimensions** (computed with numpy; real embeddings just have 384 dimensions instead of 2):

| Vector | Dot product with q = (3, 4) | Cosine with q |
| --- | --- | --- |
| a = (6, 8) | 50.0 | **1.0**: same direction, just longer |
| b = (4, 3) | 24.0 | **0.96**: nearly the same direction |
| c = (−4, 3) | 0.0 | **0.0**: at right angles |

Notice that `a` has a much bigger dot product than `b` simply because it's longer, yet its *direction* is identical to `q`. Length says nothing about meaning, which is why we compare angles, not raw dot products.

## 3. The normalisation trick

If every vector is first scaled to length 1 ("normalised"), the denominator `|q| × |v|` becomes `1 × 1`, and **cosine similarity is just the dot product**:

```
q = (3, 4) → normalised (0.6, 0.8)
b = (4, 3) → normalised (0.8, 0.6)
dot = 0.6×0.8 + 0.8×0.6 = 0.96     ← the same 0.96 as the cosine above
```

raggy normalises every vector once, when it's created (`raggy/embed/base.py`, `normalize`). Then scoring a query against *every* stored chunk is a single matrix multiplication:

```
scores = vectors @ query        # (n_chunks × 384) @ (384,) → n_chunks scores
```

On a real run with this repo's embedder:

| Query: "how long do refunds take?" | Score |
| --- | --- |
| "Refunds are issued within 14 days." | **0.845** |
| "The office cat is called Biscuit." | 0.333 |

The query and the matching sentence share only the word "refunds", but the model matched "how long … take" with "within 14 days" by meaning.

**Scores are relative, not absolute.** 0.333 for an unrelated sentence may look high. Each model has its own baseline, and with bge models even unrelated English text rarely scores below 0.3. Compare scores within one model and one query, and never treat 0.7 as a universal "relevant" threshold. (Week 2 measures how well the ranking works, and week 7 adds a reranker that produces better-calibrated scores.)

## 4. Queries are not documents

A question and its answer don't look alike: "how long do refunds take?" versus "Refunds are paid back within 14 days of the return arriving." Many retrieval models, bge included, were trained with an instruction prefix on the *query* side so the model knows which role each text plays. fastembed adds the right prefix when you call `query_embed` for questions and `passage_embed` for documents. Using the wrong one usually still works, but slightly worse. It's a free accuracy gain, so raggy always uses the right method.

## 5. Exact search vs approximate search (preview of week 5)

| | Exact (brute force), week 1 | Approximate nearest neighbour (ANN), week 5 |
| --- | --- | --- |
| How | Score every chunk, keep the top k | Build an index (graph or clusters) that jumps close to the answer without scoring everything |
| Cost per query | n × 384 multiply-adds | roughly logarithmic in n |
| Misses the true best match? | Never | Sometimes. Measured as *recall loss* |
| Good up to | tens of thousands of chunks on a laptop | millions and beyond |

Start exact. It's correct by construction, so when you move to ANN you can measure exactly how much accuracy you traded for speed.

## 6. Things that silently break embeddings

- **Mixing models.** Vectors from two different models live in unrelated spaces, and comparing them gives meaningless scores with no error. raggy stores the model name in the index and refuses to mix them.
- **Truncation.** bge-small reads at most 512 tokens (about 350 to 400 words). Anything longer is cut off silently. raggy's 800-character chunks stay well under that.
- **Language.** `bge-small-en` is an English model. Other languages need a multilingual model, and you'd re-ingest after switching.
