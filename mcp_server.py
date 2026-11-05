"""
mcp_server.py

Exposes the literature index as an MCP (Model Context Protocol) server, so
any MCP client -- a local agent, an orchestrator, or a desktop AI assistant --
can search the corpus and get grounded answers as tools.

The default corpus is public, published research, so connecting this server
to a cloud-backed client is a reasonable choice. One caveat: if you've added
anything private to data/notes/ or data/books/, whatever these tools return
is sent to the model behind the calling client. Keep private material out of
the index, or connect only local (Ollama-backed) clients.

Usage:
  python mcp_server.py
  (then point an MCP client at this script via stdio)
"""

import sys

import chromadb
from mcp.server.fastmcp import FastMCP

from build_index import CHROMA_DIR, COLLECTION_NAME, OllamaEmbeddingFunction
from query import DEFAULT_MODEL, get_answer, retrieve

mcp = FastMCP("clinical-lit-rag")

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_embed_fn = OllamaEmbeddingFunction()
_collection = _client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embed_fn)


@mcp.tool()
def search_literature(query: str, top_k: int = 5) -> str:
    """Search the indexed research literature for passages relevant to a query.

    Returns raw matching text with source labels. Use this when you need the
    underlying passages rather than a synthesized answer.
    """
    retrieved = retrieve(_collection, query, top_k=top_k)
    lines = []
    for i, (chunk, meta) in enumerate(retrieved, 1):
        label = meta.get("title") or meta.get("source_id", "")
        lines.append(f"[{i}] {label} ({meta.get('date', '')})\n{chunk}")
    return "\n\n---\n\n".join(lines) if lines else "No matches found."


@mcp.tool()
def ask_literature(question: str, model: str = DEFAULT_MODEL, top_k: int = 5) -> str:
    """Ask a question and get an answer synthesized by the local Ollama model,
    grounded only in retrieved passages, with the sources it drew on.

    Summarizes research only; it does not give individual medical advice.
    """
    result = get_answer(_collection, model, question, top_k=top_k)
    if result["error"]:
        return result["error"]
    sources = "\n".join(
        f"[{i}] {s['title']} ({s['date']}) [{s['type']}]"
        for i, s in enumerate(result["sources"], 1)
    )
    return f"{result['answer']}\n\nSources:\n{sources}"


if __name__ == "__main__":
    print("Starting clinical-lit-rag MCP server (stdio).", file=sys.stderr)
    mcp.run()
