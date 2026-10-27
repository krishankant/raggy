# Images: making pictures searchable

**[← Docs hub](README.md)** · **[Week 10 lesson](../lessons/week10-images.md)**

> **New to the ideas?** The general concepts behind this page, with diagrams, worked examples and reading links: [13 Multimodal RAG](../concepts/13-multimodal-rag.md).

This guide sits beside week 10. A retriever searches text, so a picture is only findable through text about it, or through a model that puts pictures and text in one space. This guide compares the three readers raggy uses, explains the block format and the cache, and lists what still slips through.

---

## 1. Three readers, three blind spots

Take one picture: a bar chart titled "Bikes serviced per month", with bars for Jan 42, Feb 57, Mar 81, Apr 64.

**OCR** finds text regions and transcribes them:

```text
Bikes serviced per month, Merrow workshop 2026
42
57
81
64
Jan Feb Mar Apr
```

Every string is exact. But "81" and "Mar" are on different lines, and nothing says they belong together; that link was the bar's *position*. Ask "how many in March?" and the answer depends on the answer model guessing the column order. And a picture with no words (a product photo) gives OCR nothing at all.

**A vision model** (a language model that takes images as input) looks at the whole picture and writes about it:

> A vertical bar chart titled "Bikes serviced per month, Merrow workshop 2026". Jan: 42, Feb: 57, Mar: 81, Apr: 64. Blue bars on a white background.

It keeps values with their labels and sees colours, shapes, arrows and counts. But it paraphrases, and it can misread: on the course laptop, the local 4B model's first answer called the workshop "Marrow". It's also slow: 10–20 s a picture on this laptop, or a paid call.

**CLIP** doesn't produce text. It maps the picture to a vector in the same space as text vectors, so "a bar chart" or "a blue cargo bike" lands near matching pictures. It's good for *finding* a picture by its look, and for finding pictures similar to another one. It's poor at reading text or counting, and it can't answer a question; it can only point at a picture.

| | OCR | vision model | CLIP |
| --- | --- | --- | --- |
| Exact strings (part numbers, values) | ✓ | usually | ✗ |
| Which value belongs to which label | ✗ | ✓ | ✗ |
| No-text content (colour, count, shape) | ✗ | ✓ | roughly |
| Find a picture by its look | via its words | via its description | ✓ |
| Cost per picture on this laptop | ~0.5 s | 10–20 s local / a paid call | ~0.1 s |

## 2. Why raggy keeps both texts

In `vision` mode, a block holds the description *and* the OCR text:

```text
[Image: torque-settings.png]
A screenshot of a two-column table titled "Ebb torque settings" ...
Text in image:
Ebb torque settings
Part
Torque
Stem bolts
5 Nm
...
```

The description carries structure and visual facts. The OCR text carries exact strings, which keyword search (week 6) and lexical citation checks (week 8) need, and which a paraphrase can quietly change. If the description fails, the block still has the OCR text, and the failure is counted.

## 3. The header is the citation and the gold

Every block starts with `[Image: <label>]`, where the label says where the picture is:

| Where | Label |
| --- | --- |
| A picture file | `chart.png` |
| A PDF figure | `spec.pdf p. 2, figure 1` |
| A slide picture | `deck.pptx slide 3, picture 2` |
| A Word picture | `report.docx figure 4` |

The header serves two purposes. In a citation, it sends the reader to the exact picture. In the eval, it's the gold: the header is written identically whatever reads the picture, so retrieval is scored the same way under every mode and model, with nothing depending on a model's wording.

## 4. Caching

A description is the slow, paid part of ingest. raggy caches each picture's block body by:

```text
sha256(image bytes) + mode + model name + prompt version (vis-v1)
```

- **Renaming** a file or reusing the picture elsewhere: a cache hit (the header is rebuilt with the new label).
- **Changing the model or the prompt:** a miss, so every picture is described again by the new reader.
- **A failure:** never cached, so the next ingest retries.

## 5. What gets skipped

- **Decorative pictures:** under 100 px on the short side, like icons, bullets and small logos.
- **Repeats:** a picture whose bytes already appeared in the same document (a logo on every slide) is read once, at its first appearance.
- **Scanned pages:** OCR'd whole by week 9, and not read again as a figure.
- **Vector-drawn charts in PDFs:** drawn with lines and text, not stored as an image. Their labels are text, so week 9 already reads them, but the bars' geometry isn't read.
- **Pictures in web pages:** each would need its own fetch; out of scope.

## 6. Choosing a mode

| Your collection | Mode |
| --- | --- |
| Few pictures, or decorative ones | `off` |
| Screenshots, labels, scanned forms: text in pictures | `ocr` |
| Charts, diagrams, slide decks, product photos: facts in the layout or the look | `vision` |
| You need to find pictures by appearance ("photos of damaged frames") | add `RAGGY_IMAGE_SEARCH=on` |

A local vision model (`RAGGY_VISION=ollama`, `qwen3.5:4b` by default) keeps everything on your machine, at 10–20 s a picture on an 8 GB laptop. That's paid once per picture and cached. A hosted model is faster and more accurate, and each picture is a paid call. Use the `images` suite to check on your own machine.

## Further reading

- [CLIP](https://arxiv.org/abs/2103.00020), Radford et al. (2021).
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): the detection + recognition pipeline behind RapidOCR.
- [ColPali](https://arxiv.org/abs/2407.01449), Faysse et al. (2024): retrieving document *pages* by embedding their images directly, skipping text extraction altogether. A different design from this course's text-first one, and worth knowing.
