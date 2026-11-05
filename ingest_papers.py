"""
ingest_papers.py

Pulls research literature from PubMed/PMC into local JSON files that
build_index.py can chunk and embed.

Three public, free, no-account-needed services:
  - E-utilities (ESearch/ESummary/EFetch, run by NCBI) -- search PubMed,
    get metadata and abstracts for essentially any indexed article.
  - BioC-PMC API (also NCBI) -- full text for articles in the PMC Open
    Access Subset. This service is intermittently unavailable on NCBI's
    end, so when it fails we fall through to the next source below.
  - Europe PMC RESTful API (run independently by EMBL-EBI, not NCBI) --
    mirrors much of the same open-access full text and acts as a fallback
    when NCBI's own service is unavailable. If both full-text sources come
    up empty, the article is still indexed using its abstract alone.

This only ever calls public research-literature APIs. No cloud LLM is
involved at any point.

NCBI asks that automated tools identify themselves with a contact email and
stay under ~3 requests/second without an API key -- this script does both.
Set your contact email once in your shell (it is never written to this repo):

  export NCBI_CONTACT_EMAIL="you@example.com"

Usage:
  python ingest_papers.py --max-results 200            # default trauma-literature query
  python ingest_papers.py --query '"ptsd"[tiab]' --max-results 500
  python ingest_papers.py --full-text-only             # skip abstract-only papers
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "papers"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

TOOL_NAME = "clinical-lit-rag"
CONTACT_EMAIL = os.environ.get("NCBI_CONTACT_EMAIL", "").strip()

REQUEST_DELAY_SECONDS = 0.34  # keeps us under 3 req/sec, NCBI's no-API-key limit

# The demonstration corpus: complex PTSD literature. CPTSD is recognized as a
# distinct diagnosis in ICD-11 but not in DSM-5, so the literature mixes
# closely related and contested terminology -- a useful stress test for
# retrieval. Pass --query to build a corpus on any other topic.
DEFAULT_QUERY = (
    '"complex ptsd"[tiab] OR "cptsd"[tiab] OR '
    '"complex post-traumatic stress disorder"[tiab] OR '
    '"complex posttraumatic stress disorder"[tiab]'
)


def require_contact_email() -> None:
    if not CONTACT_EMAIL:
        sys.exit(
            "NCBI_CONTACT_EMAIL is not set.\n"
            "NCBI asks automated tools to identify themselves with a contact email.\n"
            'Set it in your shell first, e.g.:  export NCBI_CONTACT_EMAIL="you@example.com"'
        )


def _get(url: str, params: dict, retries: int = 3) -> requests.Response:
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            time.sleep(REQUEST_DELAY_SECONDS)
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def search_pubmed(query: str, max_results: int) -> list[str]:
    """Return a list of PubMed IDs matching the query."""
    pmids = []
    retstart = 0
    page_size = min(200, max_results)
    while len(pmids) < max_results:
        resp = _get(EUTILS_BASE + "esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retstart": retstart,
            "retmax": page_size,
            "tool": TOOL_NAME,
            "email": CONTACT_EMAIL,
        })
        resp.raise_for_status()
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            break
        pmids.extend(id_list)
        retstart += len(id_list)
        if len(id_list) < page_size:
            break
    return pmids[:max_results]


def fetch_metadata(pmid: str) -> dict:
    """Title, journal, pub date, authors via ESummary."""
    resp = _get(EUTILS_BASE + "esummary.fcgi", {
        "db": "pubmed", "id": pmid, "retmode": "json",
        "tool": TOOL_NAME, "email": CONTACT_EMAIL,
    })
    resp.raise_for_status()
    result = resp.json().get("result", {}).get(pmid, {})
    authors = [a.get("name", "") for a in result.get("authors", [])]
    return {
        "title": result.get("title", ""),
        "journal": result.get("fulljournalname", "") or result.get("source", ""),
        "pub_date": result.get("pubdate", ""),
        "authors": authors,
    }


def fetch_abstract(pmid: str) -> str:
    """Plain abstract text via EFetch, parsed out of PubMed XML."""
    resp = _get(EUTILS_BASE + "efetch.fcgi", {
        "db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml",
        "tool": TOOL_NAME, "email": CONTACT_EMAIL,
    })
    resp.raise_for_status()
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return ""
    parts = [el.text for el in root.iter("AbstractText") if el.text]
    return "\n".join(parts).strip()


_full_text_diag_shown = False  # print one diagnostic line on the first failure, not 200 of them


def fetch_full_text(pmid: str) -> Optional[str]:
    """Full text via the BioC-PMC API, only works if the article is in the
    PMC Open Access Subset. Returns None (not an error) if it isn't --
    that's the normal case for most PubMed articles."""
    global _full_text_diag_shown

    url = f"{BIOC_BASE}/BioC_json/{pmid}/unicode"
    headers = {
        # www.ncbi.nlm.nih.gov (unlike eutils.ncbi.nlm.nih.gov) sits behind
        # bot filtering that can reject the default python-requests
        # User-Agent outright -- a descriptive one avoids that.
        "User-Agent": f"{TOOL_NAME}/1.0 (contact: {CONTACT_EMAIL})",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        time.sleep(REQUEST_DELAY_SECONDS)
    except requests.exceptions.RequestException:
        return None

    if resp.status_code != 200:
        if not _full_text_diag_shown:
            print(f"    [diag] BioC request for PMID {pmid} returned HTTP {resp.status_code} "
                  f"(content-type: {resp.headers.get('content-type', '?')}). "
                  f"First 200 chars: {resp.text[:200]!r}")
            _full_text_diag_shown = True
        return None
    try:
        data = resp.json()
    except ValueError:
        if not _full_text_diag_shown:
            print(f"    [diag] BioC request for PMID {pmid} returned HTTP 200 but non-JSON body "
                  f"(content-type: {resp.headers.get('content-type', '?')}). "
                  f"First 200 chars: {resp.text[:200]!r}")
            _full_text_diag_shown = True
        return None

    documents = data if isinstance(data, list) else data.get("documents", [])
    texts = []
    for doc in documents:
        for passage in doc.get("passages", []):
            text = passage.get("text", "")
            if text:
                texts.append(text)
    full_text = "\n\n".join(texts).strip()
    return full_text if full_text else None


_europepmc_diag_shown = False


def fetch_full_text_europepmc(pmid: str) -> Optional[str]:
    """Fallback full-text source, used when NCBI's own BioC service is down
    or doesn't have the article. Europe PMC is a separately-run mirror of
    much of the same open-access content (EMBL-EBI, not NCBI infrastructure).

    Written defensively: every step is wrapped so an unexpected response
    shape falls back to None (-> abstract-only for that paper) rather than
    crashing the whole ingestion run. Prints one diagnostic on first
    unexpected failure so a real problem is still visible.
    """
    global _europepmc_diag_shown

    def _diag(msg: str):
        global _europepmc_diag_shown
        if not _europepmc_diag_shown:
            print(f"    [diag] Europe PMC / PMID {pmid}: {msg}")
            _europepmc_diag_shown = True

    headers = {"User-Agent": f"{TOOL_NAME}/1.0 (contact: {CONTACT_EMAIL})"}

    try:
        search_resp = requests.get(f"{EUROPEPMC_BASE}/search", params={
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "resultType": "core",
            "pageSize": 1,
        }, headers=headers, timeout=30)
        time.sleep(REQUEST_DELAY_SECONDS)
        if search_resp.status_code != 200:
            _diag(f"search returned HTTP {search_resp.status_code}. "
                  f"First 200 chars: {search_resp.text[:200]!r}")
            return None

        results = search_resp.json().get("resultList", {}).get("result", [])
        if not results:
            _diag("search succeeded but returned zero results for this PMID.")
            return None

        record = results[0]
        pmcid = record.get("pmcid")
        is_oa = record.get("isOpenAccess")
        if not pmcid or is_oa != "Y":
            _diag(f"record found but not usable -- pmcid={pmcid!r}, isOpenAccess={is_oa!r} "
                  f"(record keys: {sorted(record.keys())})")
            return None

        text_resp = requests.get(
            f"{EUROPEPMC_BASE}/{pmcid}/fullTextXML", headers=headers, timeout=30
        )
        time.sleep(REQUEST_DELAY_SECONDS)
        if text_resp.status_code != 200:
            _diag(f"fullTextXML for {pmcid} returned HTTP {text_resp.status_code}. "
                  f"First 200 chars: {text_resp.text[:200]!r}")
            return None

        root = ET.fromstring(text_resp.text)
        # JATS full text: every <p> anywhere in the document body, namespace-agnostic.
        paragraphs = [
            "".join(el.itertext()) for el in root.iter()
            if el.tag.split("}")[-1] == "p"
        ]
        full_text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        if not full_text:
            _diag(f"fullTextXML for {pmcid} parsed fine but no <p> text found "
                  f"(root tag: {root.tag}).")
        return full_text if full_text else None

    except requests.exceptions.RequestException as e:
        _diag(f"request failed: {type(e).__name__}: {e}")
        return None
    except ET.ParseError as e:
        _diag(f"fullTextXML failed to parse as XML: {e}")
        return None
    except Exception as e:  # noqa: BLE001 -- deliberately broad: this is a best-effort fallback
        _diag(f"unexpected error: {type(e).__name__}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pull PubMed/PMC research papers to local JSON files.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query (same syntax as the PubMed search bar).")
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--full-text-only", action="store_true",
        help="Skip papers with no confirmed open-access full text entirely, instead of "
             "falling back to indexing just the abstract. Most PubMed articles are NOT in "
             "the PMC Open Access Subset, so this flag can shrink results a lot.",
    )
    args = parser.parse_args()

    require_contact_email()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching PubMed for: {args.query!r} (up to {args.max_results} results)")
    pmids = search_pubmed(args.query, args.max_results)
    print(f"Found {len(pmids)} matching articles. Fetching metadata + text...")

    saved = 0
    full_text_count = 0
    full_text_sources = {"ncbi": 0, "europepmc": 0}
    skipped_no_full_text = 0
    for i, pmid in enumerate(pmids, 1):
        out_path = args.output_dir / f"{pmid}.json"
        if out_path.exists():
            continue  # already ingested

        meta = fetch_metadata(pmid)
        abstract = fetch_abstract(pmid)

        full_text = fetch_full_text(pmid)
        full_text_source = "ncbi" if full_text else None
        if full_text is None:
            full_text = fetch_full_text_europepmc(pmid)
            full_text_source = "europepmc" if full_text else None

        has_full_text = full_text is not None

        if args.full_text_only and not has_full_text:
            skipped_no_full_text += 1
            if i % 25 == 0:
                print(f"  ...{i}/{len(pmids)} ({full_text_count} with full text so far, "
                      f"{skipped_no_full_text} skipped -- abstract-only)")
            continue

        record = {
            "pmid": pmid,
            "title": meta["title"],
            "authors": meta["authors"],
            "journal": meta["journal"],
            "pub_date": meta["pub_date"],
            "abstract": abstract,
            "full_text": full_text or abstract,  # fall back to abstract-only
            "has_full_text": has_full_text,
            "full_text_source": full_text_source,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        saved += 1
        if has_full_text:
            full_text_count += 1
            full_text_sources[full_text_source] += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(pmids)} ({full_text_count} with full text so far)")

    skip_note = (
        f" ({skipped_no_full_text} skipped -- no confirmed full text, per --full-text-only)."
        if args.full_text_only else " (rest are abstract-only)."
    )
    print(f"Done. Wrote {saved} new paper(s) to {args.output_dir} "
          f"({full_text_count} had full open-access text -- "
          f"{full_text_sources['ncbi']} via NCBI, {full_text_sources['europepmc']} via Europe PMC)"
          f"{skip_note}")


if __name__ == "__main__":
    main()
