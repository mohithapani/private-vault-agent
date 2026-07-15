# Secure Identity Vault — Local RAG Pipeline

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system
for ingesting personal identity documents (PDFs, images, text files) and
answering questions about them — scoped to a specific person when asked.

Everything runs on your machine: document parsing, OCR/vision extraction,
embeddings, vector storage, and the LLM itself, all via [Ollama](https://ollama.com).
No document content or query ever leaves your computer.

---

## What this app does

You give it two kinds of input:

1. **A file path** (`.pdf`, `.png`/`.jpg`/`.jpeg`, `.txt`, `.md`) → it
   extracts the text, figures out *who the document belongs to*, and stores
   it in a local vector database tagged with that person's identity.
2. **A query** (a question in plain English) → it figures out *who you're
   asking about* (if anyone), searches the vector database — filtered to
   that person's documents when possible — and returns the most relevant
   chunks.

This makes it useful for households, families, or teams who scan in mixed
batches of documents (I-94s, passports, bills, forms) belonging to different
people, and want to later ask "what's *John's* admission number?" without
digging through every file by hand.

The whole thing is built as a **LangGraph state machine** — a small graph of
nodes that routes automatically between an *ingestion* pipeline and a
*retrieval* pipeline depending on what input you give it.

---

## Prerequisites

- macOS, Linux, or WSL2 on Windows
- [pyenv](https://github.com/pyenv/pyenv) (Python version management)
- [Poetry](https://python-poetry.org/) (dependency management)
- [Ollama](https://ollama.com) (local LLM runtime)
- ~8 GB free disk space for models (more if you use a larger vision model)

---

## 1. Install Ollama and pull the required models

Ollama runs the LLMs locally. This app uses three:

| Purpose | Model | Used by |
|---|---|---|
| Embeddings | `nomic-embed-text` | `embedding_model` — turns text chunks into vectors for storage/search |
| Vision / OCR | `llava` | `vision_llm` — reads text out of image uploads (e.g. photographed IDs) |
| Text classification & routing | `llama3.2` | `classifier_llm` — figures out who a document belongs to, and who a query is about |

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
ollama pull llava
ollama pull llama3.2
```

Verify they're installed:

```bash
ollama list
```

You should see all three models listed. You can sanity-check any one of them
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
streamlit = "^1.38"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

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
├── main.py     # LangGraph pipeline: ingestion + retrieval nodes (see below)
├── app.py      # Streamlit chat UI — the primary way to run this app
└── pyproject.toml
```

`app.py` imports the compiled graph (`local_rag_app`) from `main.py`, so both
files need to live in the same directory.

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
Drag and drop one or more files (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`,
`.md`), then click **🚀 Index Uploaded Files**. Each file is saved locally to
a `streamlit_workspace/` folder (created automatically on first run) and run
through the full ingestion pipeline — extract → classify → index — with a
success/error status shown per file.

**Main canvas — chat:**
Type a question in the chat box (e.g. *"Give me John's I-94 admission
number"*). Under the hood this:
1. Runs the retrieval pipeline (`router` → `retriever`) to fetch the most
   relevant chunks, filtered to a specific person if one was mentioned.
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
    "query": ""
})
```

This routes into the **ingestion pipeline**: extract → classify → index.

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

This routes into the **retrieval pipeline**: parse query for a target person
→ search, filtered by that person if one was detected.

Both keys (`file_path`, `query`) always need to be present in the state dict
(empty string if unused) — the graph's entry router (`routing_decision_gate`)
uses their presence to decide which pipeline to run.

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
        [classifier]                    [retriever]
              │                              │
              ▼                              ▼
          [indexer]                         END
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
- `.png` / `.jpg` / `.jpeg` → `extract_image_text_via_vlm` (sends the image
  to the `llava` vision model with an OCR-style prompt, asking it to
  transcribe names, dates, ID numbers, and employer info verbatim)
- `.pdf` → `extract_pdf_text` (uses `pypdf` to pull raw text page by page)
- `.txt` / `.md` → read directly from disk

Output: raw extracted text, stored in `state["extracted_text"]`.

**`classify_identity_node`**
Sends the extracted text to `llama3.2`, constrained via a Pydantic schema
(`IdentityClassification`) to return a clean list of person identifiers. The
prompt tells the model to normalize names to `lowercase_with_underscores`
form and to strip email addresses down to just the local part (e.g.
`john_doe@gmail.com` → `john_doe`). Falls back to `["unknown"]` if
extraction fails or the model errors out.

**`index_node`**
Splits the extracted text into overlapping chunks (`RecursiveCharacterTextSplitter`,
1000 chars, 200 overlap) and writes them into the Chroma vector store.

Each person identified by the classifier is expanded into a set of aliases
via `generate_person_aliases` — e.g. `"john_doe"` becomes
`{"john_doe", "john doe", "john"}` — and **one tagged copy of each
chunk is stored per alias**, under a single scalar metadata field
`belongs_to_person`. This is what lets a later first-name-only query (like
"John") find documents that were classified under a full name
("john_doe") without any fuzzy/substring matching at query time — Chroma
doesn't support that on metadata, so the aliasing has to happen up front, at
write time.

### Retrieval pipeline

**`query_routing_node`**
Sends the user's query to `llama3.2`, constrained via a Pydantic schema
(`QueryRouter`), asking it to pull out any person's name mentioned in the
question and normalize it the same way (`lowercase_with_underscores`). If no
person is mentioned, `filter_person` stays `None`.

**`retrieve_node`**
Runs a similarity search (`k=2`) against the vector store.
- If a `filter_person` was detected, it first tries a metadata-filtered
  search (`belongs_to_person == <normalized query term>`), scoped to that
  person's tagged chunks.
- If that filtered search comes back empty, it **falls back** to an
  unfiltered semantic search across the whole vault, so the user still gets
  an answer rather than nothing.
- If no person was detected in the query at all, it runs a plain unfiltered
  semantic search from the start.

---

## Known limitations

- **First-name collisions aren't disambiguated.** If two different people
  in your vault share a first name (e.g. "John Doe" and "John Snow"),
  a first-name-only query will match chunks from both, silently blended
  into one result set. Fine for a single-person or small trusted-family
  vault; worth revisiting if the vault grows to cover people who might share
  a first name.
- **Classification quality depends on the LLM.** `llama3.2` running locally
  is fast but not infallible — always spot-check `identified_people` on
  sensitive documents (the ingestion pipeline prints this to the console).
- **No deletion/update flow.** Re-ingesting a corrected version of a
  document adds new chunks rather than replacing the old ones. If you need
  to correct a misclassified document, you'll need to remove its chunks
  from the Chroma collection manually.

---

## Troubleshooting

**`ConnectionError` / model calls hang or fail**
Ollama isn't running, or isn't reachable on `localhost:11434`. Run
`ollama serve` in a separate terminal and confirm `ollama list` shows your
models.

**Classification always returns `["unknown"]`**
Usually means `extracted_text` was empty — check that the file path is
correct and that PDF/image extraction actually produced text (some scanned
PDFs with no embedded text layer will need OCR instead of `pypdf`).

**Filtered search always falls back to global search**
Check the console output under `---Metadata before indexing---` to confirm
chunks were actually tagged with the alias you expect, and compare it
against what `query_routing_node` printed for `filter_person` — they need to
match exactly (both are lowercase, underscore-normalized strings).

**Streamlit app can't import `local_rag_app`**
`app.py` must be run from the same directory as `main.py` (`poetry run
streamlit run app.py` from the project root), since it does `from main
import local_rag_app`.

**Uploaded file shows "Successfully Indexed" but chat can't find it**
The node-level logs (extraction, classification, indexing) print to the
terminal running `streamlit run app.py`, not to the browser — check there
first to confirm what `identified_people` came back as for that file.