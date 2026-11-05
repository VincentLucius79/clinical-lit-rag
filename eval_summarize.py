"""
eval_summarize.py

A small eval harness for one narrow question: when the generation model
summarizes a full-text paper, how close does that summary land to the
paper's own (human-written) abstract?

This deliberately bypasses the retrieval path -- it feeds the model one full
paper directly and asks for a summary, rather than going through
build_index.py/query.py's chunk-and-retrieve pipeline. That isolates the
model's summarization ability from retrieval quality, so you get a number
that moves (or doesn't) when you change temperature or swap models, instead
of eyeballing a handful of answers and guessing whether they got better.

Scoring: embed both the model's summary and the paper's real abstract with
the same embedding model used for the index (nomic-embed-text), then compute
cosine similarity between the two vectors. Closer to 1.0 means the summary
covered similar ground to the abstract. It's a proxy for "did the model
capture what this paper is about," not a measure of writing quality or
factual accuracy -- a summary can score well and still misstate a detail, so
treat this as one signal, not a verdict.

Nothing here leaves the machine: papers are already-public PubMed/PMC text,
and both the summarizing and embedding models are local Ollama models.

Usage:
  python eval_summarize.py                          # eval 15 random full-text papers
  python eval_summarize.py --n 30                   # eval 30 papers instead
  python eval_summarize.py --seed 42                # reproducible sample
  python eval_summarize.py --model gemma3:4b        # compare a different model
  python eval_summarize.py --log                    # append results to data/eval_results.jsonl
                                                    # so runs can be compared after a tuning change
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone

import requests

from build_index import PAPERS_DIR, SCRIPT_DIR, OllamaEmbeddingFunction
from query import DEFAULT_MODEL, NUM_CTX, OLLAMA_URL, SYSTEM_PROMPT, TEMPERATURE, THINK

EMBED_MODEL = "nomic-embed-text"
EVAL_LOG_PATH = SCRIPT_DIR / "data" / "eval_results.jsonl"

SUMMARY_PROMPT_TEMPLATE = (
    "Summarize the following research paper in 3-5 sentences, focusing on "
    "its research question, method, and main finding. Write only the "
    "summary -- no preamble, no restating the title.\n\n"
    "PAPER:\n{full_text}\n\nSUMMARY:"
)

# Full papers can run long; truncate to keep this a fair single-shot
# summarization test rather than an accidental long-context stress test.
MAX_INPUT_WORDS = 3000


def load_full_text_papers():
    """Only papers with both a real abstract and confirmed full text -- you
    can't score against an abstract that doesn't exist, and summarizing an
    abstract-only record would test paraphrase, not summarization."""
    papers = []
    if not PAPERS_DIR.exists():
        return papers
    for path in PAPERS_DIR.glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("has_full_text") and record.get("full_text") and record.get("abstract"):
            papers.append(record)
    return papers


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def summarize(model: str, full_text: str) -> str:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(full_text=truncate_words(full_text, MAX_INPUT_WORDS))
    # Send every setting explicitly so any model is tested under identical
    # conditions. Otherwise only the custom "clinical-lit-rag" model would get
    # the Modelfile's 8192-token context and system prompt, and a raw model tag
    # would fall back to Ollama's smaller default context and silently see a
    # truncated paper. These values match the Modelfile, so results for
    # "clinical-lit-rag" are unchanged.
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "think": THINK,
            "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def run_eval(n: int, model: str, seed: int = None):
    papers = load_full_text_papers()
    if not papers:
        print(f"No full-text papers found in {PAPERS_DIR}. Run ingest_papers.py first.")
        return []

    rng = random.Random(seed)
    sample = rng.sample(papers, min(n, len(papers)))
    embed_fn = OllamaEmbeddingFunction(model=EMBED_MODEL)

    print(f"Evaluating {len(sample)} of {len(papers)} full-text papers with model '{model}'...\n")

    results = []
    for i, paper in enumerate(sample, 1):
        title = paper.get("title", "(untitled)")
        pmid = paper.get("pmid", "?")
        try:
            start = time.time()
            summary = summarize(model, paper["full_text"])
            elapsed = time.time() - start

            summary_vec = embed_fn.embed_single(summary)
            abstract_vec = embed_fn.embed_single(paper["abstract"])
            score = cosine_similarity(summary_vec, abstract_vec)

            results.append({"pmid": pmid, "title": title, "score": score, "seconds": round(elapsed, 1)})
            print(f"[{i}/{len(sample)}] {score:.3f}  ({elapsed:.1f}s)  {title[:70]}")
        except requests.exceptions.RequestException as e:
            print(f"[{i}/{len(sample)}] [error] {title[:70]}: {e}")

    return results


def print_summary_stats(results):
    if not results:
        print("\nNo results to summarize.")
        return
    scores = [r["score"] for r in results]
    scores_sorted = sorted(scores)
    mean = sum(scores) / len(scores)
    median = scores_sorted[len(scores_sorted) // 2]
    print(f"\n--- {len(scores)} papers scored ---")
    print(f"mean:   {mean:.3f}")
    print(f"median: {median:.3f}")
    print(f"min:    {min(scores):.3f}")
    print(f"max:    {max(scores):.3f}")
    worst = sorted(results, key=lambda r: r["score"])[:3]
    print("\nLowest-scoring (worth a manual look):")
    for r in worst:
        print(f"  {r['score']:.3f}  {r['title'][:70]}")


def log_results(results, model: str):
    if not results:
        return
    scores = [r["score"] for r in results]
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "n": len(results),
        "mean_score": round(sum(scores) / len(scores), 4),
        "results": results,
    }
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nAppended run to {EVAL_LOG_PATH} -- compare mean_score across lines after tuning changes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="Number of papers to sample and evaluate.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to use for summarization.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--log", action="store_true", help="Append this run's results to data/eval_results.jsonl.")
    args = parser.parse_args()

    results = run_eval(args.n, args.model, seed=args.seed)
    print_summary_stats(results)
    if args.log:
        log_results(results, args.model)


if __name__ == "__main__":
    main()
