"""
query.py

Ask questions of the indexed literature. Retrieval and generation both
happen through your own Ollama server -- nothing here calls a cloud API.

Usage:
  python query.py --ask "How do ICD-11 criteria distinguish complex PTSD from PTSD?"
  python query.py                              # interactive loop
  python query.py --model gemma3:4b            # override the generation model for one run
  CLINICAL_LIT_MODEL=gemma3:4b python query.py # same, as an environment variable
"""

import argparse
import os

import chromadb
import requests

from build_index import CHROMA_DIR, COLLECTION_NAME, OllamaEmbeddingFunction

OLLAMA_URL = "http://localhost:11434"

# "clinical-lit-rag" is a custom Ollama model built from ./Modelfile
# (run `ollama create clinical-lit-rag -f Modelfile` once). It bakes in
# SYSTEM_PROMPT, NUM_CTX and TEMPERATURE below, so a raw
# `ollama run clinical-lit-rag` gets the same settings as these scripts.
DEFAULT_MODEL = os.environ.get("CLINICAL_LIT_MODEL", "clinical-lit-rag")

TOP_K = 12  # With a corpus of over a thousand papers, fewer slots let one or two
            # papers dominate an answer. 12 chunks (~2,600-3,000 words at current
            # chunk sizes) still fits inside the 8192-token context with the prompt.
NUM_CTX = 8192  # Ollama silently defaults much lower (historically 2048) if this isn't set,
                # which can truncate retrieved context before the model ever sees it.
TEMPERATURE = 0.3  # a little above 0 for natural synthesis, still grounded / low-hallucination
THINK = False  # Some Gemma 4 builds run an internal reasoning pass by default, adding
               # several hundred tokens (and tens of seconds) per answer. Set True to
               # trade speed for the model reasoning before it answers.

# Also baked into the Modelfile's SYSTEM directive. Kept here as a fallback
# (sent via generate()'s "system" field) so a raw base-model tag passed with
# --model still behaves the same way.
SYSTEM_PROMPT = (
    "You are a research assistant for published clinical literature. Using only the "
    "context provided below, synthesize a clear, well-reasoned answer in your own "
    "words -- don't just string together quotes or list facts source by source. "
    "Actively connect and compare information across the passages: note where "
    "sources agree, where they disagree or add nuance, and what the fuller picture "
    "looks like when you combine them. Be precise about terminology: if sources use "
    "related but distinct terms or diagnostic frameworks, say which source uses which. "
    "If the answer isn't in the context, say so plainly instead of guessing -- never "
    "fill gaps with outside knowledge. You summarize research; you do not diagnose or "
    "recommend treatment for any individual. If asked to, say so and suggest a "
    "qualified professional. After synthesizing, note which source(s) (paper or book "
    "title) each key claim draws from."
)


def retrieve(collection, question: str, top_k: int = TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(chunks, metadatas))


def build_prompt(question: str, retrieved) -> str:
    # The system prompt lives in the Modelfile (or generate()'s "system" field),
    # so the prompt itself is just the retrieved context and the question.
    context_blocks = []
    for i, (chunk, meta) in enumerate(retrieved, 1):
        label = meta.get("title") or meta.get("source_id", f"source {i}")
        date = meta.get("date", "")
        context_blocks.append(f"[{i}] {label} ({date})\n{chunk}")
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


def generate(
    model: str, prompt: str,
    num_ctx: int = NUM_CTX, temperature: float = TEMPERATURE, think: bool = THINK,
    system: str = SYSTEM_PROMPT,
) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system,  # redundant with the Modelfile's SYSTEM for "clinical-lit-rag",
                                # but keeps raw base-model tags passed via --model consistent
            "stream": False,
            "think": think,  # harmless no-op for non-reasoning models
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        },
        # generous timeout: the first request includes cold-loading the model into
        # VRAM, and spilling into shared system memory slows generation further
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def get_answer(collection, model: str, question: str, top_k: int = TOP_K) -> dict:
    """Retrieve + generate, returning structured data. Shared by the CLI
    below, web_app.py and mcp_server.py, so every front end uses identical logic."""
    retrieved = retrieve(collection, question, top_k=top_k)
    if not retrieved:
        return {"answer": None, "sources": [], "error": "No indexed content yet -- run build_index.py first."}

    prompt = build_prompt(question, retrieved)
    response_text = generate(model, prompt).strip()
    sources = [
        {
            "title": meta.get("title") or meta.get("source_id", ""),
            "date": meta.get("date", ""),
            "type": meta.get("source_type", ""),
        }
        for _, meta in retrieved
    ]
    return {"answer": response_text, "sources": sources, "error": None}


def answer(collection, model: str, question: str, top_k: int = TOP_K):
    result = get_answer(collection, model, question, top_k=top_k)
    if result["error"]:
        print(result["error"])
        return

    print("\n--- Sources ---")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['title']} ({s['date']}) [{s['type']}]")

    print("\n--- Answer ---")
    print(result["answer"])
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", help="Ask a single question and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to use for generation.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = OllamaEmbeddingFunction()
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    if args.ask:
        answer(collection, args.model, args.ask, top_k=args.top_k)
        return

    print(f"Ready. Model: {args.model}. Type a question (or 'quit').")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        answer(collection, args.model, question, top_k=args.top_k)


if __name__ == "__main__":
    main()
