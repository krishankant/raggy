# Vector search at scale: exact, IVF, HNSW, and when you need any of them

**[← Docs hub](README.md)** · **[Week 5 lesson](../lessons/week05-scale.md)**

This guide sits beside week 5. It explains how a vector store finds the nearest chunks, what approximate (ANN) indexes trade away, and how to decide whether you need one. Numbers marked *measured* come from this machine (Apple M2, 8 GB) with raggy's code. The week 5 lesson has the ones from real embeddings.

---

## 1. Exact search is a matrix multiply

Every stored chunk is a unit-length vector, and so is the query. Their dot product is their cosine similarity. Exact search computes it for every chunk and keeps the k largest:

```python
scores = vectors @ query                  # (n,) — one multiply-add per dimension per chunk
top = np.argpartition(-scores, k)[:k]     # the k best, unordered: O(n)
top = top[np.argsort(-scores[top])]       # sort only those k
```

The cost is n × d multiply-adds: 100,000 chunks × 384 dimensions is 38 million, which a laptop CPU does in a few milliseconds. *Measured* on random 384-dimensional vectors, p50 per query:

| vectors | numpy exact | LanceDB exact scan |
| --- | --- | --- |
| 10,000 | 0.20 ms | 4.1 ms |
| 100,000 | 3.5 ms | 22 ms |
| 500,000 | 16 ms | 93 ms |

Exact search grows linearly, and it's *exact*: it can't miss the true nearest chunk. Answering the question afterwards takes 1–2 seconds of LLM time, so retrieval below about 50 ms is invisible to the user.

LanceDB's scan is slower than an in-memory numpy matrix because it reads columnar data from disk and runs a query engine. You pay that for the things a matrix can't do: incremental writes, filters inside the search, and an index when you need one.

## 2. Approximate nearest neighbours (ANN)

ANN indexes avoid comparing the query with every vector. They are *approximate*: some true neighbours are skipped. The two families raggy can use, often combined:

### IVF: inverted file (cluster, then search a few clusters)

1. **Train:** run k-means on the vectors to find P centroids (*partitions*).
2. **Index:** assign every vector to its nearest centroid.
3. **Search:** compare the query with the P centroids, pick the `nprobes` nearest, and scan only their vectors.

With P = 256 and `nprobes` = 20, a search scans about 8% of the vectors. A true neighbour sitting just across a partition boundary is missed if its partition isn't probed. More `nprobes` means more recall and more time.

### HNSW: hierarchical navigable small world (walk a graph)

Each vector is linked to its nearest neighbours in a layered graph: sparse long-distance links at the top, dense short ones at the bottom. A search starts at the top and greedily walks towards the query, dropping down a layer whenever it can't get closer. It visits a few hundred vectors instead of all of them. It's fast and accurate, at the cost of memory for the links and slow inserts.

### SQ and PQ: compressing the vectors

**Scalar quantisation (SQ)** stores each dimension in 8 bits instead of 32: 4× smaller, with slightly fuzzy distances. **Product quantisation (PQ)** goes much further by splitting a vector into sub-vectors and storing each as a codebook index. It's fuzzier still. **`refine_factor`** fixes most of the fuzz: fetch `refine × k` candidates with the compressed distances, then re-score them with the full vectors and keep the best k.

raggy's ANN index is LanceDB's **IVF_HNSW_SQ**: partitions found by k-means, an HNSW graph inside each partition, and 8-bit vectors, re-scored with `refine_factor`.

## 3. Recall is measured against exact search, not against the labels

An ANN index can only lose what exact search would have found. So its quality has a clean measurement that needs no human labels: **overlap@k**, the fraction of exact search's top k that the ANN search also returned. raggy's eval records it on every question when `RAGGY_ANN=on`.

Then check the end metric (nDCG@10 against the benchmark's labels) as well. A lost neighbour only matters if it was relevant.

*Measured* on random vectors with default settings, overlap@10 was 0.46 at 10,000 vectors, 0.10 at 100,000 and 0.04 at 500,000. Random high-dimensional vectors have no clusters, which is the worst case for IVF: every partition boundary cuts through dense neighbourhoods. Real embeddings cluster by topic, and the lesson measures them. Never accept an ANN default without measuring overlap on *your* vectors.

## 4. When do you need ANN?

Work it out from your latency budget instead of guessing:

- Exact search costs about **n × d** multiply-adds per query. On this laptop, numpy does about 10 billion a second, and LanceDB's scan about a sixth of that.
- With d = 384 and a 50 ms budget, numpy exact search stays under budget up to roughly a million vectors, and LanceDB's scan up to roughly 200,000.
- Below those sizes, ANN buys milliseconds nobody notices, and costs recall, index build time and tuning.

That's why week 5 sets its rule before any run: ANN is on by default only if, on the scale benchmark, exact search's p95 latency is over 100 ms **and** ANN keeps nDCG@10 within 0.005 of exact.

## 5. What a vector database is for, below that size

At "thousands to tens of thousands of documents", LanceDB earns its place for reasons other than search speed:

- **Incremental, crash-safe writes.** Adding or deleting one document's rows writes a new version. The previous version stays readable until the write commits. NumpyStore rewrites every vector on every save.
- **Filters inside the search.** `where doc_type = 'pdf'` is applied before the top k is chosen (`prefilter=True`), so "the 4 best PDFs" returns 4 PDFs. Filtering after retrieval returns fewer than 4, or none.
- **One table for everything.** The rows hold the text, the metadata and the vector, and week 6 adds a full-text (BM25) index to the same table.

## 6. The other half of scale: embedding throughput

At ingest, the slow part isn't the store but the embedding model. *Measured* for bge-small on this machine: about 10 texts a second for 1,500-character texts. Three things move that number:

- **Length-sorted batches.** A batch is padded to its longest text. Sorting by length first means less padding.
- **Batch size.** 16 was faster than 64 here (11.7 against 9.2 texts/s), because a smaller batch wastes less padding.
- **Not embedding at all.** A cache keyed by `sha256(model + text)` makes every text the model has seen before free. Re-ingests, eval rebuilds and settings changes mostly re-embed nothing.

One that *didn't* help: four worker processes ran at 1.5 texts/s. Each loads its own copy of the model, and on an 8 GB machine the copies compete for memory and cores.

## Further reading

- Y. Malkov, D. Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016): HNSW.
- H. Jégou, M. Douze, C. Schmid, "Product quantization for nearest neighbor search" (2011): PQ and IVF.
- The LanceDB documentation on vector indexes: IVF_PQ, IVF_HNSW_SQ, `nprobes` and `refine_factor`.
- ann-benchmarks.com: recall against queries per second for many libraries, on standard datasets.
