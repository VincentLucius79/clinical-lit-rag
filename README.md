# clinical-lit-rag

A fully local retrieval-augmented generation (RAG) system for published
clinical research. It pulls papers from PubMed and PubMed Central, embeds them
into a local vector database, and answers questions with a quantized
open-weight model, citing the sources each claim draws from. Retrieval,
generation and evaluation all run on one consumer GPU. No cloud LLM is
involved at any point.

The demonstration corpus is the complex PTSD (CPTSD) literature:
**13,410 indexed chunks**, including **429 papers with open-access full text**.

> **This is a research tool, not a clinical one.** It summarizes published
> literature. It does not diagnose, and it does not recommend treatment for
> anyone. If you are struggling, please reach out to a qualified mental health
> professional. In the US, you can call or text 988 to reach the Suicide &
> Crisis Lifeline.

## Why this corpus

Complex PTSD is a hard test for retrieval because the field itself disagrees
about the terms. The WHO's ICD-11 recognizes CPTSD as a distinct diagnosis;
the DSM-5 does not list it separately. So the literature mixes closely related,
overlapping vocabulary: PTSD, complex PTSD, complex trauma, developmental
trauma, and comorbidity with borderline personality disorder.

A naive system blurs those together. This one is built to keep them apart:
the system prompt requires the model to say which source uses which framework,
to answer only from retrieved passages, and to say plainly when the answer
isn't in them.

## Architecture

```
PubMed / PMC ──(E-utilities, BioC, Europe PMC fallback)──▶ data/papers/*.json   ingest_papers.py
PDF books ─────(dropped in by you)──────────────────────▶ data/books/*.pdf
notes ─────────(dropped in by you)──────────────────────▶ data/notes/*.md|txt

data/*  ──(per-source chunking + nomic-embed-text via Ollama)──▶  data/chroma_db   build_index.py

data/chroma_db ──(top-k retrieval + Gemma 4 E4B via Ollama)──▶ answer + sources
                                                               ├─ query.py        CLI
                                                               ├─ web_app.py      local browser UI
                                                               └─ mcp_server.py   tools for AI agents

data/papers ──(summarize full text, compare to real abstract)──▶ eval_summarize.py
```

## What's worth a look

- **Resilient ingestion.** NCBI's full-text service is intermittently down,
  so `ingest_papers.py` falls back to Europe PMC (an independently run mirror)
  and then to the abstract alone, rather than failing the run. It stays under
  NCBI's rate limit and identifies itself as NCBI asks.
- **Chunking by source type.** Dense paper prose, long-form books and
  free-form notes each get their own chunk size and overlap. Books are chunked
  page by page so every citation carries a page number.
- **Fault tolerance in indexing.** Embedding calls retry transient failures;
  one bad chunk is logged and skipped instead of killing a run of thousands.
  Re-runs embed only what's new. Sources can be added or removed from the
  index without touching the files on disk.
- **One configuration, everywhere.** A custom Ollama `Modelfile` bakes in the
  system prompt, an 8,192-token context window and temperature, so the CLI,
  the web UI, the MCP server and a raw `ollama run` all behave identically.
  (Ollama otherwise defaults to a much smaller context window and can silently
  truncate retrieved passages.)
- **Three front ends, one code path.** The CLI, the Flask web UI and the MCP
  server all call the same `get_answer()`, so they can't drift apart.
- **An evaluation harness**, described below, so changes are measured
  instead of eyeballed.
- **Safe defaults.** The web UI listens on `127.0.0.1` only. Your NCBI
  contact email comes from an environment variable and never touches the repo.

## Evaluation

`eval_summarize.py` isolates one capability: summarization. It gives the
model a paper's full text (truncated to 3,000 words), asks for a 3–5 sentence
summary, then embeds both that summary and the paper's own human-written
abstract and measures their **cosine similarity** (1.0 means the two point in
exactly the same direction in embedding space). Retrieval is deliberately
bypassed so the score reflects the model, not the index. Every run can be
logged to `data/eval_results.jsonl` for before/after comparisons.

**Baseline run** — 15 randomly sampled full-text papers (seed 42), model
`clinical-lit-rag` (Gemma 4 E4B, 4-bit quantized), RTX 4060 8 GB:

| Metric | Score |
|---|---|
| Mean | 0.907 |
| Median | 0.929 |
| Min | 0.796 |
| Max | 0.964 |

Summaries took 3.9–8.5 seconds each.

**How to read this honestly.** Cosine similarity between two texts on the
same narrow topic starts high, so 0.907 does not mean "91% correct." It means
the summaries consistently cover the same ground as the authors' abstracts.
The score says nothing about factual accuracy: a summary can score well and
still misstate a detail. The three lowest-scoring papers (0.796–0.828) were
an opinion piece on treatment approaches, a longitudinal stability study and
a practitioner-training study — formats where a "research question, method,
main finding" summary fits less naturally than it does for an empirical
paper. They're the ones worth reading by hand.

### Does a bigger model help?

Same 15 papers, same seed, same settings. The harness sends the context
window, system prompt and temperature with every request, so any model is
tested under identical conditions.

| | Gemma 4 E4B (default) | Gemma 4 12B |
|---|---|---|
| Mean | 0.907 | 0.912 |
| Median | 0.929 | 0.930 |
| Min | 0.796 | 0.821 |
| Seconds per paper* | 5.5 | 16.0 |

*Excluding the first paper, which includes loading the model.

The 12B scored higher on 9 papers and lower on 6. A paired t-test on the
per-paper differences gives p ≈ 0.17 (a Wilcoxon signed-rank test agrees),
and one paper accounts for half the gap. On summarization, the 12B shows no
measurable gain at 2.9× the latency, partly because on an 8 GB card it doesn't
fully fit and 16–19% of it runs from system memory. **E4B stays the default.**

Spot checks of full RAG answers point the other way. In one 12B answer, every
claim cited a retrieved source; in one of two E4B answers to the same
question, the model repeatedly cited a paper that appeared only *inside* a
retrieved passage, not in its source list. Answers vary from run to run, so
that's a hypothesis rather than a result, and it's what the citation eval
below is for.

### Next steps for evaluation

1. **A floor for comparison:** score each summary against a *random other*
   abstract from the same corpus. The gap between matched and random pairs is
   the real signal.
2. **Citation faithfulness:** ask a fixed question set several times per
   model and count citations to anything outside the retrieved source list, or
   to source numbers that don't exist.
3. **Retrieval quality:** a small set of questions whose answers live in known
   papers, scored by recall@k (how often the right paper appears in the top k
   results), to tune chunk sizes and `TOP_K` against a number. A per-paper
   chunk cap belongs here too: in spot checks, 12 retrieval slots held only 10
   distinct papers.
4. **Terminology precision:** questions that hinge on the PTSD / CPTSD
   distinction, checking whether answers attribute each framework correctly.

## Setup

You need Python 3.10+, [Ollama](https://ollama.com/), and a GPU with about
8 GB of VRAM (a 4-bit 4B–8B model plus the small embedding model fit
comfortably; a 13B+ model alongside embeddings will not).

```bash
git clone https://github.com/VincentLucius79/clinical-lit-rag.git
cd clinical-lit-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama create clinical-lit-rag -f Modelfile
```

## Usage

```bash
# 1. Build a corpus (NCBI asks automated tools for a contact email)
export NCBI_CONTACT_EMAIL="you@example.com"
python ingest_papers.py --max-results 200            # default: complex PTSD literature
python ingest_papers.py --query '"ptsd"[tiab]'       # or any PubMed query

# 2. Index it (re-runs only embed what's new)
python build_index.py

# 3. Ask questions
python query.py --ask "How does ICD-11 distinguish complex PTSD from PTSD?"
python query.py                                      # interactive
python web_app.py                                    # browser UI at http://127.0.0.1:5050

# 4. Evaluate
python eval_summarize.py --n 15 --seed 42 --log
```

To use the corpus from an AI agent, run `python mcp_server.py` and register it
with any MCP client. It exposes `search_literature` (raw passages) and
`ask_literature` (a grounded answer with sources). The default corpus is public
research, but anything you add to `data/notes/` or `data/books/` will be sent
to whatever model sits behind the calling client.

### Tuning

The two biggest levers on answer quality are the chunk sizes in
`build_index.py` and `TOP_K` in `query.py`. Change one at a time, rebuild, and
re-run the eval with the same seed.

## Repository layout

```
ingest_papers.py    PubMed/PMC ingestion with full-text fallbacks
build_index.py      per-source chunking and local embedding into Chroma
query.py            retrieval + generation; the shared get_answer()
web_app.py          local Flask UI
mcp_server.py       MCP tools for AI agents
eval_summarize.py   summarization eval harness
Modelfile           the custom Ollama model: system prompt, context, temperature
```

`data/` is gitignored: the corpus is rebuildable, the index is large, and PDF
books may be copyrighted.

## Acknowledgements

- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) and the
  [BioC-PMC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/)
  for PubMed metadata, abstracts and open-access full text
- [Europe PMC](https://europepmc.org/RestfulWebService) (EMBL-EBI) for the
  full-text fallback
- [Ollama](https://ollama.com/), [Chroma](https://www.trychroma.com/),
  [nomic-embed-text](https://ollama.com/library/nomic-embed-text), and
  Google's Gemma 4 (via Unsloth's quantized GGUF build), used under its own
  terms and not redistributed here

## License

MIT, in `LICENSE`. Paper text retrieved by the scripts remains under its
publishers' and authors' licenses; none is included in this repository.
