"""How much of the document prefix does the provider cache during contextual retrieval (week 8)?

Runs contextual retrieval on the manual with an empty context cache, so every chunk is a real call:
    uv run python evals/cache_probe.py openai gpt-5.5
"""
import sys, tempfile, json
from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))
from raggy.config import Config
from raggy.chunking import chunk_document
from raggy.chunking.contextual import ContextCache, add_llm_context, _prefix
from raggy.embed.fastembed_embedder import FastEmbedEmbedder
from raggy.evals.suites import SUITES
from raggy.generate.llm import make_llm
c = Config.from_env()
doc = SUITES["manual"]().documents(c.load_options())[0]
emb = FastEmbedEmbedder(c.embed_model)
chunks = chunk_document(doc, c.chunk_settings(), emb.count_tokens)
llm = make_llm(sys.argv[1], sys.argv[2])
st = add_llm_context(doc, chunks, llm, ContextCache(tempfile.mktemp(suffix=".sqlite")), emb.count_tokens, emb.max_tokens)
u = llm.usage
print(json.dumps({"model": llm.name, "chunks": len(chunks), "prefix_tokens_approx": emb.count_tokens(_prefix(doc)),
                  "calls": st.calls, "failed": st.failed, **u,
                  "cached_share_of_input": round(u["cache_read"] / u["input"], 3) if u["input"] else None}))
