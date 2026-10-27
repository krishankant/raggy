# 07 · Keyword search and BM25

**[← 06 Vector search](06-vector-search.md)** · **[Concepts](README.md)** · **Next: [08 Hybrid search →](08-hybrid-search.md)** · **In the course:** [Week 6](../lessons/week06-hybrid-search.md)

Before embeddings, search engines ranked documents by the words they share with the query. The best-known formula, **BM25**, dates from the 1990s and is still the default in Elasticsearch, OpenSearch and Lucene. It's also still very good at what embeddings are bad at: exact names, codes and rare terms.

---

## 1. The problem keyword search solves

Ask an embedding-based search for part number `BW-BT-5000` and it may return `BW-BT-5001`, which is almost the same string and so almost the same vector. Keyword search treats `bw-bt-5000` as a token that either appears in the document or doesn't.

| Query type | Embeddings | Keywords |
| --- | --- | --- |
| "How fast do I get my money back?" (paraphrase) | ✅ | ❌ no shared words with "refund" |
| `BW-BT-5000` (code) | ⚠️ near-misses | ✅ exact |
| "Saltmarsh Cycles" (rare name) | ⚠️ if the model never saw it | ✅ |
| "error 0x80070057" | ⚠️ | ✅ |

## 2. The inverted index

The data structure behind keyword search is the **inverted index**: for every word, the list of documents (and positions) containing it. It's the index at the back of a book.

```text
Documents:                                    Inverted index:
D1 "the battery is covered by a 2 year warranty"   battery  → D1, D2, D3
D2 "refunds for a battery are paid within 14 days"  warranty → D1, D4
D3 "charge the battery fully before the first ride" refunds  → D2
D4 "the frame warranty lasts ten years"             frame    → D4
                                                    ...
```

To answer "battery warranty", look up two lists and score only the documents on them (D1–D4 here), instead of scanning every document. That makes keyword search fast over billions of documents.

Before indexing, text is **analysed**: lower-cased, split into tokens, often with **stop words** removed ("the", "is", "a") and **stemming** applied (`covered` → `cover`, `refunds` → `refund`). The query goes through the same analysis.

## 3. TF-IDF: the intuition

Two common-sense rules:

1. **Term frequency (TF):** a document that mentions "warranty" several times is probably more about warranties.
2. **Inverse document frequency (IDF):** a word that appears in *every* document ("the", or "bike" in a bike shop's docs) tells you nothing; a word in few documents ("warranty") is a strong signal.

TF-IDF multiplies the two. BM25 refines both halves.

## 4. BM25

For a query with terms q₁ … qₙ and a document D:

```text
score(D, Q) = Σ  IDF(qᵢ) × ( tf(qᵢ, D) × (k1 + 1) ) / ( tf(qᵢ, D) + k1 × (1 − b + b × |D| / avgdl) )
              i

IDF(q) = ln( 1 + (N − df(q) + 0.5) / (df(q) + 0.5) )
```

| Symbol | Meaning |
| --- | --- |
| tf(q, D) | How many times term q appears in D |
| df(q) | How many documents contain q |
| N | Number of documents |
| \|D\| | Length of D (in terms); avgdl is the average length |
| k1 (≈ 1.2) | **Saturation**: how quickly extra repeats stop helping |
| b (≈ 0.75) | **Length normalisation**: how much long documents are penalised |

Two refinements over TF-IDF:

- **Saturation.** The 10th mention of "warranty" adds much less than the 1st. With k1 = 1.2, a term's tf-part can never exceed k1 + 1 = 2.2, no matter how often it repeats. At average length, tf = 1 gives 1.0, and tf = 3 gives 3 × 2.2 / (3 + 1.2) = **1.57**, not 3.
- **Length normalisation.** A long document mentions everything a few times by chance; b scales tf down for documents longer than average and up for shorter ones.

## 5. BM25 worked by hand

Take the four documents above, after removing stop words:

| Doc | Terms | Length |
| --- | --- | --- |
| D1 | battery covered 2 year warranty | 5 |
| D2 | refunds battery paid within 14 days | 6 |
| D3 | charge battery fully before first ride | 6 |
| D4 | frame warranty lasts ten years | 5 |

N = 4, avgdl = 22 / 4 = 5.5. Query: **"battery warranty"**. k1 = 1.2, b = 0.75.

**Step 1: IDF.** "battery" is in 3 documents, "warranty" in 2.

```text
IDF(battery)  = ln(1 + (4 − 3 + 0.5) / (3 + 0.5)) = ln(1 + 0.429) = ln 1.429 = 0.357
IDF(warranty) = ln(1 + (4 − 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.0)   = ln 2     = 0.693
```

The rarer word is worth about twice as much.

**Step 2: the tf part.** Every tf is 1 here; only length differs.

```text
Length 5: 1 − 0.75 + 0.75 × 5/5.5 = 0.932;  tf part = 1 × 2.2 / (1 + 1.2 × 0.932) = 2.2 / 2.118 = 1.039
Length 6: 1 − 0.75 + 0.75 × 6/5.5 = 1.068;  tf part = 2.2 / (1 + 1.2 × 1.068)   = 2.2 / 2.282 = 0.964
```

Shorter documents get a slightly higher tf part for the same count.

**Step 3: sum per document.**

| Doc | battery | warranty | **BM25** | Rank |
| --- | --- | --- | --- | --- |
| D1 | 0.357 × 1.039 = 0.371 | 0.693 × 1.039 = 0.720 | **1.091** | 1 |
| D4 | — | 0.720 | **0.720** | 2 |
| D2 | 0.357 × 0.964 = 0.344 | — | **0.344** | 3 (tie) |
| D3 | 0.344 | — | **0.344** | 3 (tie) |

D1 contains both terms and wins. D4, with only the *rare* term, beats D2 and D3, which have only the common one. That's IDF at work.

## 6. Strengths and weaknesses

| Strengths | Weaknesses |
| --- | --- |
| Exact matching of names, codes, numbers | No synonyms: "money back" ≠ "refund" |
| No model, no GPU; fast to build and update | No word order: "dog bites man" = "man bites dog" |
| Explainable: you can see which words matched | Sensitive to spelling, tokenisation and language |
| Strong out of the box on many benchmarks (see BEIR) | Stemming and stop words are language-specific |

The weaknesses of BM25 are the strengths of embeddings, and vice versa. That is the whole case for hybrid search ([08](08-hybrid-search.md)).

## 7. Learned sparse retrieval

Models such as **SPLADE** keep the inverted-index format but use a transformer to choose and weight the terms, including terms *not in the text*: a passage about refunds might get a weight for "money" and "reimburse" too. That gives keyword-index speed with some of the vocabulary bridging of embeddings. `bge-m3` produces dense and sparse vectors in one pass.

## 8. Common mistakes

- **Different analysis at index and query time** (stemmed index, unstemmed query), so nothing matches.
- **Tokenising codes badly**: `BW-BT-5000` split into `bw`, `bt`, `5000` matches every BW part and every 5000.
- **Assuming embeddings make keyword search obsolete.** On names, codes and rare terms, BM25 often still wins.

## 9. In raggy

- [Week 6](../lessons/week06-hybrid-search.md) and the [hybrid search guide](../docs/hybrid-search.md): a BM25 full-text index on the same LanceDB table as the vectors, combined with vector search.

## 10. Further reading

| Read | Why |
| --- | --- |
| [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (Wikipedia) | The formula and its variants |
| [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), Manning, Raghavan & Schütze, chapters 1, 2 and 6 | Free textbook: inverted indexes, tokenisation, TF-IDF |
| [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), Robertson & Zaragoza (2009) | Where BM25 comes from, by its authors |
| [Practical BM25: How Shards Affect Relevance Scoring](https://www.elastic.co/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch) (Elastic) | BM25 as a real search engine computes it |
| [BEIR benchmark](https://arxiv.org/abs/2104.08663), Thakur et al. (2021) | Shows BM25 is a strong baseline across many domains |
| [SPLADE](https://arxiv.org/abs/2107.05720), Formal et al. (2021) | Learned sparse retrieval |
