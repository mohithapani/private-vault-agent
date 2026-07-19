"""
Debug utility for inspecting what's actually stored in the local Chroma
vector store. Run this directly (not through Streamlit) to see the raw
chunks + metadata that got indexed, and optionally test a query against
them the same way retrieve_node does.

Usage examples:

    # Dump everything in the DB (source, belongs_to_person, chunk preview)
    python debug_chroma.py

    # Only show chunks from a specific file
    python debug_chroma.py --source "Pranali"

    # Only show chunks tagged to a specific person
    python debug_chroma.py --person pranali

    # Run an actual similarity search (same call retrieve_node makes) and
    # see what comes back, with an optional person filter
    python debug_chroma.py --query "what is the address" --person pranali

    # Show full, untruncated chunk text instead of a preview
    python debug_chroma.py --source "Pranali" --full
"""

import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_DIR = "my_local_chroma_db"


def load_store():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name="secure_identity_vault",
        embedding_function=embedding_model,
        persist_directory=DB_DIR,
    )


def preview(text: str, full: bool, width: int = 300) -> str:
    text = text.replace("\n", " ⏎ ")
    if full or len(text) <= width:
        return text
    return text[:width] + f"... [{len(text) - width} more chars]"


def dump_all(store, source_filter: str, person_filter: str, full: bool):
    # .get() pulls raw records straight out of the collection -- no
    # embedding / similarity search involved, just "what's in there".
    raw = store.get(include=["documents", "metadatas"])
    ids = raw["ids"]
    docs = raw["documents"]
    metas = raw["metadatas"]

    print(f"Total chunks in collection: {len(ids)}\n")

    shown = 0
    for chunk_id, doc_text, meta in zip(ids, docs, metas):
        source = meta.get("source", "")
        person = meta.get("belongs_to_person", "<untagged>")

        if source_filter and source_filter.lower() not in source.lower():
            continue
        if person_filter and person_filter.lower() != str(person).lower():
            continue

        shown += 1
        print("-" * 80)
        print(f"id:       {chunk_id}")
        print(f"source:   {source}")
        print(f"person:   {person}")
        print(f"content:  {preview(doc_text, full)}")

    print("-" * 80)
    print(f"\nShown {shown} / {len(ids)} chunks matching your filters.")
    if shown == 0 and (source_filter or person_filter):
        print("No chunks matched -- check the filter text, or rerun with no "
              "filters to see everything that's actually stored.")


def run_query(store, query: str, person: str | None, full: bool):
    filt = {"belongs_to_person": {"$eq": person.lower()}} if person else None
    print(f"Running similarity_search(query={query!r}, k=4, filter={filt})\n")

    results = store.similarity_search_with_score(query, k=4, filter=filt)
    if not results:
        print("No results returned. If a person filter was set, this is "
              "the exact condition that triggers retrieve_node's fallback "
              "to a blanket (unfiltered) search.")
        return

    for doc, score in results:
        print("-" * 80)
        print(f"score (lower = more similar): {score:.4f}")
        print(f"source:   {doc.metadata.get('source')}")
        print(f"person:   {doc.metadata.get('belongs_to_person', '<untagged>')}")
        print(f"content:  {preview(doc.page_content, full)}")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="Substring filter on the source file path/name")
    parser.add_argument("--person", help="Exact filter on belongs_to_person (e.g. 'pranali')")
    parser.add_argument("--query", help="If given, runs a similarity search instead of dumping raw chunks")
    parser.add_argument("--full", action="store_true", help="Print full chunk text instead of a truncated preview")
    args = parser.parse_args()

    store = load_store()

    if args.query:
        run_query(store, args.query, args.person, args.full)
    else:
        dump_all(store, args.source or "", args.person or "", args.full)