# Secure Identity Vault — Local RAG Pipeline

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system
for ingesting personal identity documents (PDFs, images, text files) and
answering questions about them — scoped to a specific person when asked.

Everything runs on your machine: document parsing, OCR extraction,
embeddings, vector storage, and the LLM itself, all via [Ollama](https://ollama.com)
and a local [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) engine.
No document content or query ever leaves your computer.

---

## What this app does

You give it two kinds of input:

1. **A file path** (`.pdf`, `.png`/`.jpg`/`.jpeg`, `.txt`, `.md`) → it
   extracts the text and stores it in a local vector database, tagged with
   whichever person you explicitly assign it to (or left untagged if it
   doesn't belong to anyone in particular).
2. **A query** (a question in plain English) → it figures out *who you're
   asking about* (if anyone) by matching the name in your question against
   the people you've actually registered, searches the vector database —
   filtered to that person's documents when a match is found — and returns
   the most relevant chunks.

This makes it useful for households, families, or teams who scan in mixed
batches of documents (I-94s, passports, bills, forms) belonging to different
people, and want to later ask "what's *John's* admission number?" without
digging through every file by hand.

The whole thing is built as a **LangGraph state machine** — a small graph of
nodes that routes automatically between an *ingestion* pipeline and a
*retrieval* pipeline depending on what input you give it.

> **Note:** who a document belongs to is no longer auto-detected from its
> content. You assign it explicitly in the Streamlit sidebar at upload
> time — see [Section 5](#5-running-the-app-streamlit-ui). This is more
> reliable than the earlier LLM-classification approach and keeps a proper
> record of who's been added to your vault.

---

## Prerequisites

- macOS, Linux, or WSL2 on Windows
- [pyenv](https://github.com/pyenv/pyenv) (Python version management)
- [Poetry](https://python-poetry.org/) (dependency management)
- [Ollama](https://ollama.com) (local LLM runtime)
- [Poppler](https://poppler.freedesktop.org/) (system dependency for
  rendering PDF pages to images — required by `pdf2image`)
- ~8 GB free disk space for models

### Install Poppler

`pdf2image` shells out to Poppler's `pdftoppm` binary, so it needs to be on
your `PATH` separately from the Python dependencies.

**macOS:**
```bash
brew install poppler
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install poppler-utils
```

**Windows:** see the [pdf2image installation notes](https://github.com/Belval/pdf2image#windows).

---

## 1. Install Ollama and pull the required models

Ollama runs the LLMs locally. This app uses two:

| Purpose | Model | Used by |
|---|---|---|
| Embeddings | `nomic-embed-text` | `embedding_model` — turns text chunks into vectors for storage/search |
| Text routing & persona matching | `llama3.2` | `classifier_llm` — pulls a person's name out of a query, then matches it against your registered people |

> Text/OCR extraction (images **and** scanned PDFs) is handled locally by
> **PaddleOCR**, a Python library rather than an Ollama model — there's
> nothing to `ollama pull` for it, see [Section 3](#3-set-up-dependencies-with-poetry).
> A prior version of this app used the `llava` vision model for image
> OCR; that's no longer needed, since images and scanned PDFs now go
> through the same PaddleOCR path (see [Architecture](#7-architecture--the-langgraph-nodes)).

### Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** download the installer from [ollama.com/download](https://ollama.com/download).

### Start the Ollama server

Ollama usually starts automatically as a background service after install.
If it's not running, start it manually in its own terminal:

```bash
ollama serve
```

Leave this running — the app talks to it over `localhost` (default port `11434`).

### Pull the models

In a separate terminal:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify they're installed:

```bash
ollama list
```

You should see both models listed. You can sanity-check either one
directly:

```bash
ollama run llama3.2 "Say hello in one sentence."
```

---

## 2. Set up Python with pyenv

Pick a Python version compatible with your LangChain/LangGraph versions
(3.12 is a safe default at the time of writing).

```bash
# Install pyenv if you don't have it
curl https://pyenv.run | bash

# Restart your shell, or reload your shell config, then:
pyenv install 3.12.3
pyenv local 3.12.3      # pins this project to 3.12.3 via a .python-version file
python --version         # confirm it picked up 3.12.3
```

---

## 3. Set up dependencies with Poetry

If you don't already have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Tell Poetry to use the pyenv-managed interpreter, then install dependencies:

```bash
poetry env use $(pyenv which python)
poetry install
```

If you don't yet have a `pyproject.toml`, the minimal one for this app looks
like this — save it at the project root before running `poetry install`:

```toml
[tool.poetry]
name = "secure-identity-vault"
version = "0.1.0"
description = "Local RAG pipeline for identity document ingestion and retrieval"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.12"
langchain-core = "^0.3"
langchain-ollama = "^0.2"
langchain-chroma = "^0.2"
langchain-text-splitters = "^0.3"
langgraph = "^0.2"
pydantic = "^2.9"
pypdf = "^5.0"
pdf2image = "^1.17"
paddleocr = "^2.9"
opencv-python = "^4.10"
numpy = "^1.26"
pillow = "^10.4"
streamlit = "^1.38"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

> `paddleocr` downloads its detection/recognition model weights on first
> use (cached locally afterward), so the first OCR call — on an image or a
> scanned PDF — will take longer than subsequent ones.

Once installed, run any script through Poetry's virtual environment:

```bash
poetry run python your_script.py
```

or drop into a shell with the environment already active:

```bash
poetry shell
```

---

## 4. Project structure

```
.
├── main.py               # LangGraph pipeline: ingestion + retrieval nodes (see below)
├── app.py                 # Streamlit chat UI — the primary way to run this app
├── person_registry.py     # JSON-backed store of known people + their uploaded files
├── person_registry.json   # Created automatically on first use — do not commit real data
├── debug_chroma.py        # Standalone script for inspecting stored chunks/metadata
├── debug_ocr.py            # Standalone script for isolating OCR quality issues on a specific page/image
└── pyproject.toml
```

`app.py` imports the compiled graph (`local_rag_app`) from `main.py`, and
both `main.py` and `app.py` import `person_registry.py`, so all three files
need to live in the same directory.

`person_registry.json` is created the first time you register a person (via
the sidebar or `person_registry.add_person(...)`). It's a flat JSON file —
not a database — storing each person's display name, when they were added,
and the list of files indexed under them. It's separate from, and unrelated
to, the Chroma vector store's own internal SQLite database (see
[Troubleshooting](#troubleshooting) if you hit a Chroma database error).

---

## 5. Running the app (Streamlit UI)

This is the normal way to use the project day-to-day — a browser-based chat
interface with a sidebar for uploading and indexing documents.

Make sure `ollama serve` is running (Section 1) and your models are pulled,
then from the project root:

```bash
poetry run streamlit run app.py
```

This opens the app in your browser (default `http://localhost:8501`).

**Sidebar — Document Control Panel:**

1. **Assign to person** — choose who this upload batch belongs to:
   - Pick an existing person from the dropdown (populated from
     `person_registry.json`).
   - Choose **"+ Create new person"** and type a name to register someone
     new on the spot.
   - Choose **"No specific person (leave untagged)"** if the file(s) don't
     belong to anyone in particular — these are only ever returned by
     unfiltered searches, never by a person-scoped one.

   This selection applies to the *entire batch* you're about to upload —
   there's no per-file assignment.

2. Drag and drop one or more files (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`,
   `.md`), then click **🚀 Index Uploaded Files**. Each file is saved
   locally to a `streamlit_workspace/` folder (created automatically on
   first run, and kept there — files are not deleted after indexing) and
   run through the ingestion pipeline — extract → index — with a
   success/error status shown per file. Note that OCR on images and scanned
   PDFs runs on CPU by default and can take a few seconds per page/image.

   A **📋 Known people** expander at the bottom of the sidebar lists every
   registered person so you can double check who's in the vault.

**Main canvas — chat:**
Type a question in the chat box (e.g. *"Give me John's I-94 admission
number"*). Under the hood this:
1. Runs the retrieval pipeline (`router` → `persona_matcher` → `retriever`)
   to fetch the most relevant chunks — filtered to a specific registered
   person if the name in your query resolves to one.
2. Passes those chunks as context into a strict prompt sent to `llama3.2`
   (`chat_llm`), instructed to answer only from the provided snippets and
   say "I don't know" if the answer isn't in them.
3. Renders the answer along with a **📄 Sources Referenced** list showing
   which uploaded files the answer was drawn from.

If no matching chunks are found at all, the UI reports that directly instead
of calling the LLM.

> **Note:** `app.py` opens its own connection to the same Chroma collection
> (`secure_identity_vault`) and Ollama models that `main.py` uses internally
> — there's no separate setup step for this, it just needs the same
> `DB_DIR` and Ollama server to be reachable.

---

## 6. Advanced: calling the pipeline directly in Python

If you want to script ingestion or retrieval without the UI (e.g. batch-
loading a folder of documents), you can invoke the compiled graph directly:

### Ingest a document

```python
from main import local_rag_app

local_rag_app.invoke({
    "file_path": "streamlit_workspace/I-94 Official Website - John_Doe.pdf",
    "query": "",
    "assigned_person": "john_doe",   # or None to leave it untagged
})
```

This routes into the **ingestion pipeline**: extract → index. Passing
`assigned_person` registers that person (if new) and records this file
against them in `person_registry.json`, the same way the sidebar does.

### Ask a question

```python
from main import local_rag_app

result = local_rag_app.invoke({
    "file_path": "",
    "query": "Give me John's I-94 admission number"
})

for doc in result["search_results"]:
    print(doc.page_content)
    print(doc.metadata)
```

This routes into the **retrieval pipeline**: parse the query for a target
name → resolve it against registered people → search, filtered by that
person if a match was found.

`file_path` and `query` always need to be present in the state dict (empty
string if unused) — the graph's entry router (`routing_decision_gate`) uses
their presence to decide which pipeline to run. The other keys
(`assigned_person`, `filter_person`, `matched_person`, `extracted_text`,
`search_results`) are populated as the graph runs and can be safely omitted
or set to `None`/`[]` on the way in.

### Inspecting what's actually stored

See [`debug_chroma.py`](#debugging-whats-in-the-vector-store) below.

---

## 7. Architecture — the LangGraph nodes

```
                         START
                           │
                 routing_decision_gate
                 /                    \
      file_path present?        query present, no file?
              │                              │
              ▼                              ▼
         [extractor]                     [router]
              │                              │
              ▼                              ▼
          [indexer]                  [persona_matcher]
              │                              │
              ▼                              ▼
             END                       [retriever]
                                              │
                                              ▼
                                             END
```

### `routing_decision_gate`
The single entry point. Inspects the incoming state and decides which
pipeline to run:
- `file_path` set → **ingestion pipeline** (a new document to store)
- no `file_path` but `query` set → **retrieval pipeline** (a question to answer)
- neither set → exits immediately to `END` (prevents an empty/invalid run)

### Ingestion pipeline

**`extract_node`**
Looks at the file extension and routes to the right extraction method:
- `.png` / `.jpg` / `.jpeg` → `extract_image_text_via_ocr` (runs the image
  through the local **PaddleOCR** engine)
- `.pdf` → `extract_pdf_text_hybrid`, which checks **each page
  individually**: pages with a usable embedded text layer are kept as-is,
  and any page with little/no embedded text (a purely scanned page — e.g.
  an ID document's photo or address page) is rendered to an image via
  `pdf2image` and OCR'd on its own. This replaces an earlier all-or-nothing
  check that only ran OCR when the *entire* PDF came back empty — which
  meant a document with any embedded text on any page (even a stray line)
  would silently skip OCR for every other page, dropping content from any
  purely-scanned pages. If per-page extraction fails outright (e.g. pypdf
  can't parse the file), it falls back to full-document OCR
  (`extract_pdf_text_via_ocr`) as a last resort.
- `.txt` / `.md` → read directly from disk

Images and scanned-PDF pages both go through the **same shared OCR
routine** (`run_ocr_on_image`): crop out surrounding whitespace/background,
downscale to a max dimension PaddleOCR can handle, then run text detection
+ recognition.

Output: raw extracted text, stored in `state["extracted_text"]`.

**`index_node`**
Splits the extracted text into overlapping chunks (`RecursiveCharacterTextSplitter`,
1000 chars, 200 overlap) and writes them into the Chroma vector store.

- If `state["assigned_person"]` was set (from the sidebar, or passed
  directly), it's expanded into a set of aliases via
  `generate_person_aliases` — e.g. `"john_doe"` becomes
  `{"john_doe", "john doe", "john"}` — and **one tagged copy of each
  chunk is stored per alias**, under a single scalar metadata field
  `belongs_to_person`. This is what lets a later first-name-only query
  (like "John") find documents assigned to a full name ("john_doe")
  without any fuzzy/substring matching at query time — Chroma doesn't
  support that on metadata, so the aliasing has to happen up front, at
  write time. The person is also registered (if new) and the file is
  recorded against them in `person_registry.json`.
- If no person was assigned, chunks are stored as-is with no
  `belongs_to_person` field — they'll only ever surface via an unfiltered
  semantic search, never a person-scoped one.

### Retrieval pipeline

**`query_routing_node`**
Sends the user's query to `llama3.2`, constrained via a Pydantic schema
(`QueryRouter`), asking it to pull out any person's name mentioned in the
question and normalize it (`lowercase_with_underscores`). This is just raw
text extraction from the query — it doesn't know yet whether that name
corresponds to anyone actually in the vault. If no person is mentioned,
`filter_person` stays `None`.

**`persona_match_node`**
Takes the raw name from `filter_person` and resolves it against the list of
*actually registered* people (`person_registry.list_people()`). The LLM is
given the full known-people list and asked to pick the single best match
(e.g. `"john"` → `"john_doe"`), or `None` if nothing reasonably matches. Any
LLM response that isn't literally in the registry is discarded as a safety
check. If `filter_person` was `None`, or no one is registered yet, this
step is skipped and `matched_person` stays `None`.

**`retrieve_node`**
Runs a similarity search (`k=2`) against the vector store.
- If `matched_person` was resolved, it first tries a metadata-filtered
  search (`belongs_to_person == matched_person`), scoped to that person's
  tagged chunks.
- If that filtered search comes back empty, it **falls back** to an
  unfiltered semantic search across the whole vault, so the user still gets
  an answer rather than nothing.
- If no person was matched at all, it runs a plain unfiltered semantic
  search from the start.

---

## Debugging what's in the vector store

If a query isn't finding something you know you uploaded (a missing
address, a field that "should" be there), the fastest way to check is to
look at the raw chunks directly rather than guessing. `debug_chroma.py`
connects to the same Chroma collection as the app and lets you inspect it
without going through Streamlit:

```bash
# Dump every chunk from a specific file, full text (not truncated)
poetry run python debug_chroma.py --source "passport" --full

# Only show chunks tagged to a specific registered person
poetry run python debug_chroma.py --person john_doe

# Run the exact same kind of similarity search retrieve_node does, and see
# the scores + which chunks actually come back
poetry run python debug_chroma.py --query "what is the address" --person john_doe --full
```

This helps narrow down whether a "missing" answer is because:
- OCR never captured that text in the first place (check with `--source`)
- the text is there but got split awkwardly across chunk boundaries
- the chunk exists but doesn't rank in the top `k` results for that query
  phrasing (check with `--query`)

See the script's own `--help` / docstring for more detail.

### Debugging OCR extraction quality

If `debug_chroma.py` shows a chunk exists but the *text itself* is
partial, garbled, or missing a specific field (e.g. an address that's
there in the document but not in the extracted text), the problem is
upstream in OCR, not in chunking or retrieval. `debug_ocr.py` isolates the
OCR step entirely — no indexing, no chunking — so you can inspect exactly
what PaddleOCR sees for one specific page or image:

```bash
# OCR a specific PDF page directly, and save the cropped image so you can
# eyeball whether the crop step is clipping anything important
python debug_ocr.py --pdf "streamlit_workspace/Some Document.pdf" --page 2 --save-crop cropped_page2.png

# Compare against OCR on the full-resolution image with no resize cap
# (run_ocr_on_image normally caps every image to max 2500x2500 before OCR)
python debug_ocr.py --pdf "streamlit_workspace/Some Document.pdf" --page 2 --raw

# Try alternate PaddleOCR configs (orientation/unwarping enabled, and a
# Devanagari-aware model) against the same page, useful for documents that
# mix scripts (e.g. Hindi + English on Indian ID documents)
python debug_ocr.py --pdf "streamlit_workspace/Some Document.pdf" --page 2 --experiment

# OCR a standalone image file instead of a PDF page
python debug_ocr.py --image path/to/photo.jpg
```

This walks through, in order: the rendered page size, the cropped size,
the size actually sent to OCR after the pipeline's resize cap, OCR output
with vs. without cropping, and (optionally) full-resolution and
alternate-config OCR results for comparison. It'll also flag automatically
if cropping or the resize cap appears to be losing content.

**Known limitation:** some source material is just genuinely hard for a
general-purpose local OCR model — small, dense, bilingual text (e.g. Hindi
+ English) printed over a security background pattern, as found on
official ID document pages, can come out garbled even at full resolution
with orientation/unwarping enabled and no meaningful improvement from a
higher DPI render. If `debug_ocr.py` shows the same degraded output across
crop/no-crop, full resolution, and alternate configs, that's a genuine
PaddleOCR recognition limit on that specific content rather than a bug in
the pipeline — accept the partial extraction, or consider a different OCR
engine/model for that document type.

---

## Known limitations

- **First-name collisions aren't disambiguated.** If two different people
  in your vault share a first name (e.g. "John Doe" and "John Snow"),
  a first-name-only query will match chunks from both, silently blended
  into one result set. Fine for a single-person or small trusted-family
  vault; worth revisiting if the vault grows to cover people who might share
  a first name.
- **Persona matching quality depends on the LLM.** `llama3.2` running
  locally is fast but not infallible when resolving a name in your query
  against the registered people list — always spot-check `matched_person`
  on sensitive queries (the retrieval pipeline prints this to the console).
- **`person_registry.json` doesn't dedupe near-duplicate names.**
  Registering "John Doe" and later "john doe" both normalize to the same
  `john_doe` id and correctly merge, but something like "Jon Doe" would
  register as a *separate* person. Double-check the "Known people" list in
  the sidebar before creating a new person if you're not sure they're
  already registered.
- **OCR quality depends on image/scan quality.** PaddleOCR does well on
  clean, reasonably high-resolution scans and photos, but low-light,
  blurry, or heavily skewed images can produce garbled or missing text.
  If extraction on a specific file looks off, use `debug_chroma.py` (see
  above) to check the actual stored chunk text before assuming retrieval is
  at fault.
- **Dense, small, bilingual/security-pattern text has a real OCR
  ceiling.** Official ID document pages (e.g. a passport's address page,
  mixing Hindi + English over a printed security pattern) can come out
  garbled even at full resolution, with orientation/unwarping enabled, and
  across different PaddleOCR model configs — see
  [Debugging OCR extraction quality](#debugging-ocr-extraction-quality).
  This is a genuine recognition limit of the local OCR model on that kind
  of content, not a pipeline bug.
- **No deletion/update flow.** Re-ingesting a corrected version of a
  document adds new chunks rather than replacing the old ones. If you need
  to correct a misassigned document, you'll need to remove its chunks from
  the Chroma collection manually (or via a custom script using the same
  `vector_store` connection as `debug_chroma.py`).

---

## Troubleshooting

**`ConnectionError` / model calls hang or fail**
Ollama isn't running, or isn't reachable on `localhost:11434`. Run
`ollama serve` in a separate terminal and confirm `ollama list` shows your
models (`nomic-embed-text`, `llama3.2`).

**Chroma error: "attempt to write a readonly database"**
This is Chroma's own internal SQLite file (inside `my_local_chroma_db/`) —
unrelated to `person_registry.json`. It usually means the *directory*, not
just the `.sqlite3` file, isn't writable by the user running Streamlit.
Fix ownership/permissions on the whole folder and clear any stale journal
files:
```bash
sudo chown -R $USER:$USER my_local_chroma_db
chmod -R u+rwX my_local_chroma_db
rm -f my_local_chroma_db/chroma.sqlite3-journal my_local_chroma_db/chroma.sqlite3-wal my_local_chroma_db/chroma.sqlite3-shm
```
Also make sure no other process (a second Streamlit session, a leftover
script) still has the same `persist_directory` open.

**`Poppler conversion failed` error**
`pdf2image` couldn't find the Poppler binaries. Confirm Poppler is
installed and on your `PATH` (see [Prerequisites](#prerequisites)) —
`pdftoppm -v` should print a version number if it's set up correctly.

**OCR extraction is slow or produces garbled text**
PaddleOCR runs on CPU by default unless you've configured a GPU build, so
large batches of high-DPI pages/images can take a while — this is expected.
If text comes out garbled or a specific field seems missing, use
`debug_ocr.py` (see [Debugging OCR extraction quality](#debugging-ocr-extraction-quality))
to isolate exactly what PaddleOCR extracted from that page, independent of
chunking/indexing. Note that dense, small, bilingual text over a security
background pattern (common on official ID document pages) is a known hard
case that may not improve much even with cropping/resolution/config
changes — see the limitation noted in that section.

**A query filtered to a person always falls back to global search**
Run `poetry run python debug_chroma.py --person <id>` to confirm chunks
were actually tagged with that exact id, and compare it against what the
console printed for `matched_person` when you ran the query — they need to
match exactly (both are lowercase, underscore-normalized strings). If
`matched_person` came back `None`, the issue is upstream in
`persona_match_node` not recognizing the name — check that the person is
actually in the "Known people" list in the sidebar.

**Streamlit app can't import `local_rag_app`**
`app.py` must be run from the same directory as `main.py` and
`person_registry.py` (`poetry run streamlit run app.py` from the project
root), since it does `from main import local_rag_app` and
`import person_registry`.

**Uploaded file shows "Successfully Indexed" but chat can't find something in it**
First, check the node-level logs (extraction, indexing) that print to the
terminal running `streamlit run app.py`, not the browser. Then use
`debug_chroma.py --source <filename> --full` to see exactly what text was
extracted and stored for that file — see
[Debugging what's in the vector store](#debugging-whats-in-the-vector-store).