"""
build_index.py

Chunks everything in data/papers/ (research papers from ingest_papers.py),
data/books/ (any .pdf files you drop in), and data/notes/ (any .txt/.md
files you drop in) and embeds it all into a local Chroma vector database
using an Ollama embedding model.

Nothing here calls out to any cloud API. The only network call this script
makes is to http://localhost:11434, which is your own Ollama server.

Note on PDFs: text extraction only works on born-digital PDFs (ones with
real embedded text, not just images of pages). A scanned book with no text
layer contributes nothing -- there's no OCR step here -- and the script
warns when that happens.

Prereqs:
  ollama pull nomic-embed-text

Usage:
  python build_index.py                         # embed anything new
  python build_index.py --rebuild               # wipe and rebuild from scratch
  python build_index.py --sources papers        # only index papers
  python build_index.py --remove-source books   # un-index books (local files untouched)
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import chromadb
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader

SCRIPT_DIR = Path(__file__).resolve().parent
NOTES_DIR = SCRIPT_DIR / "data" / "notes"
PAPERS_DIR = SCRIPT_DIR / "data" / "papers"
BOOKS_DIR = SCRIPT_DIR / "data" / "books"
CHROMA_DIR = SCRIPT_DIR / "data" / "chroma_db"
COLLECTION_NAME = "clinical_lit"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Each source type gets its own chunk size, because the corpus spans very
# differently-shaped text: dense technical paper prose, long-form book
# narrative, and free-form notes. Overlap is kept around 15-20% of size.
# These are reasoned starting points, not yet empirically tuned -- the eval
# harness exists so changes here can be measured rather than guessed.
PAPER_CHUNK_SIZE, PAPER_CHUNK_OVERLAP = 300, 55   # dense academic paragraphs; keep methods/results intact
BOOK_CHUNK_SIZE, BOOK_CHUNK_OVERLAP = 260, 45     # long-form narrative, already page-bounded separately
NOTE_CHUNK_SIZE, NOTE_CHUNK_OVERLAP = 220, 40     # notes vary too widely to specialize further

CHUNK_SIZE_WORDS = NOTE_CHUNK_SIZE      # chunk_text()'s own default, used if no explicit size/overlap given
CHUNK_OVERLAP_WORDS = NOTE_CHUNK_OVERLAP
MAX_TOKEN_LENGTH = 200  # "words" longer than this are almost always encoding junk, not language


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Embeds text using a locally running Ollama model. No data leaves the machine."""

    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_URL, max_retries: int = 3):
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries

    def embed_single(self, text: str) -> List[float]:
        """Embed one string, retrying transient failures. Raises after max_retries."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
            except requests.exceptions.RequestException as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        raise last_error

    def embed_single_safe(self, text: str) -> Optional[List[float]]:
        """Like embed_single, but returns None (and logs) instead of raising, so
        one bad chunk doesn't kill the whole indexing run."""
        try:
            return self.embed_single(text)
        except requests.exceptions.RequestException as e:
            preview = text[:120].replace("\n", " ")
            print(f"    [skip] embedding failed for chunk (preview: {preview!r}...): {e}")
            return None

    def __call__(self, input: Documents) -> Embeddings:
        # Used by Chroma for query-time embedding (single short strings) --
        # let failures raise here since there's nothing sensible to skip.
        return [self.embed_single(text) for text in input]


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    # Drop pathologically long "words" (no spaces) before chunking -- these are
    # almost always base64 blobs, URLs, or extraction junk that inflate a chunk
    # far past what the embedding model expects and can crash the request.
    words = [w if len(w) <= MAX_TOKEN_LENGTH else "[long-token-omitted]" for w in text.split()]
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def load_note_documents():
    docs = []
    if not NOTES_DIR.exists():
        return docs
    for path in list(NOTES_DIR.rglob("*.md")) + list(NOTES_DIR.rglob("*.txt")):
        text = path.read_text(errors="replace")
        for i, chunk in enumerate(chunk_text(text, size=NOTE_CHUNK_SIZE, overlap=NOTE_CHUNK_OVERLAP)):
            docs.append({
                "id": f"note-{path.stem}-{i}",
                "text": chunk,
                "metadata": {
                    "source_type": "note",
                    "source_id": str(path.relative_to(NOTES_DIR)),
                    "title": path.stem,
                    "date": "",
                    "from": "",
                },
            })
    return docs


def load_book_documents():
    """Chunked per-page (not across page boundaries) so every chunk carries
    an accurate page number for citation -- a book chunk that says 'p. 142'
    is far more useful than one that just says the book's title."""
    docs = []
    if not BOOKS_DIR.exists():
        return docs
    for path in BOOKS_DIR.glob("*.pdf"):
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            print(f"  [skip] couldn't open {path.name}: {type(e).__name__}: {e}")
            continue

        title = path.stem
        try:
            meta_title = reader.metadata.title if reader.metadata else None
            if meta_title:
                title = meta_title
        except Exception:
            pass

        pages_with_text = 0
        for page_num, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if not page_text.strip():
                continue
            pages_with_text += 1
            for i, chunk in enumerate(chunk_text(page_text, size=BOOK_CHUNK_SIZE, overlap=BOOK_CHUNK_OVERLAP)):
                docs.append({
                    "id": f"book-{path.stem}-p{page_num}-{i}",
                    "text": chunk,
                    "metadata": {
                        "source_type": "book",
                        "source_id": path.name,
                        "title": f"{title} (p. {page_num})",
                        "date": "",
                        "from": "",
                    },
                })

        if pages_with_text == 0:
            print(f"  [warn] {path.name}: no extractable text on any page -- "
                  f"likely a scanned PDF with no text layer (needs OCR, not handled here).")

    return docs


def load_paper_documents():
    docs = []
    if not PAPERS_DIR.exists():
        return docs
    for path in PAPERS_DIR.glob("*.json"):
        record = json.loads(path.read_text())
        authors = ", ".join(record.get("authors", [])[:5])  # cap at 5 to keep the header short
        header = (
            f"Title: {record.get('title', '')}\n"
            f"Authors: {authors}\n"
            f"Journal: {record.get('journal', '')}\n"
            f"Date: {record.get('pub_date', '')}\n\n"
        )
        body = record.get("full_text") or record.get("abstract", "")
        full_text = header + body
        for i, chunk in enumerate(chunk_text(full_text, size=PAPER_CHUNK_SIZE, overlap=PAPER_CHUNK_OVERLAP)):
            docs.append({
                "id": f"paper-{record['pmid']}-{i}",
                "text": chunk,
                "metadata": {
                    "source_type": "paper" if record.get("has_full_text") else "paper-abstract",
                    "source_id": record["pmid"],
                    "title": record.get("title", "(untitled)"),
                    "date": record.get("pub_date", ""),
                    "from": authors,
                },
            })
    return docs


# Friendly CLI names -> loader function, and -> the metadata "source_type"
# value(s) that loader's chunks get tagged with (papers has two, depending on
# whether full text was available).
SOURCE_LOADERS = {
    "papers": load_paper_documents,
    "books": load_book_documents,
    "notes": load_note_documents,
}
SOURCE_METADATA_TYPES = {
    "papers": ["paper", "paper-abstract"],
    "books": ["book"],
    "notes": ["note"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the collection from scratch.")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument(
        "--sources", default="papers,books,notes",
        help="Comma-separated source types to include when adding new content. "
             "Options: papers,books,notes.",
    )
    parser.add_argument(
        "--remove-source",
        help="Comma-separated source types to remove from the existing index before proceeding "
             "(only affects the vector index, not your local files). Options: papers,books,notes.",
    )
    args = parser.parse_args()

    active_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(active_sources) - set(SOURCE_LOADERS)
    if unknown:
        parser.error(f"Unknown source(s) in --sources: {', '.join(sorted(unknown))}. "
                     f"Valid options: {', '.join(SOURCE_LOADERS)}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = OllamaEmbeddingFunction(model=args.embed_model)

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    if args.remove_source:
        remove_list = [s.strip() for s in args.remove_source.split(",") if s.strip()]
        unknown = set(remove_list) - set(SOURCE_METADATA_TYPES)
        if unknown:
            parser.error(f"Unknown source(s) in --remove-source: {', '.join(sorted(unknown))}. "
                         f"Valid options: {', '.join(SOURCE_METADATA_TYPES)}")
        for name in remove_list:
            removed_total = 0
            for source_type in SOURCE_METADATA_TYPES[name]:
                existing = collection.get(where={"source_type": source_type})
                ids_to_remove = existing.get("ids", [])
                if ids_to_remove:
                    collection.delete(ids=ids_to_remove)
                    removed_total += len(ids_to_remove)
            print(f"Removed {removed_total} existing '{name}' chunk(s) from the index "
                  f"(local files untouched).")

    documents = []
    for name in active_sources:
        documents.extend(SOURCE_LOADERS[name]())

    if not documents:
        print(f"Nothing to index for sources {active_sources}. Put papers in {PAPERS_DIR} "
              f"(via ingest_papers.py), PDF books in {BOOKS_DIR}, or notes in {NOTES_DIR}.")
        return

    existing_ids = set(collection.get()["ids"]) if not args.rebuild else set()
    new_docs = [d for d in documents if d["id"] not in existing_ids]

    print(f"{len(documents)} chunks total, {len(new_docs)} new. Embedding via Ollama ({args.embed_model})...")

    batch_size = 64
    skipped = 0
    for start in range(0, len(new_docs), batch_size):
        batch = new_docs[start:start + batch_size]
        ids, docs_text, metas, embeddings = [], [], [], []
        for d in batch:
            vec = embed_fn.embed_single_safe(d["text"])
            if vec is None:
                skipped += 1
                continue
            ids.append(d["id"])
            docs_text.append(d["text"])
            metas.append(d["metadata"])
            embeddings.append(vec)

        if ids:
            # Pass embeddings we already computed directly -- this avoids Chroma
            # calling the embedding function again internally.
            collection.add(ids=ids, documents=docs_text, metadatas=metas, embeddings=embeddings)

        print(f"  ...{min(start + batch_size, len(new_docs))}/{len(new_docs)} "
              f"({len(ids)} added this batch{f', {skipped} skipped so far' if skipped else ''})")

    print(f"Index ready at {CHROMA_DIR} ({collection.count()} chunks total, {skipped} chunk(s) skipped).")


if __name__ == "__main__":
    main()
