import os
import cv2
import numpy as np
from PIL import Image
from typing import TypedDict, List, Optional
import pypdf
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# LangChain / Community Modules
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph Modules
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Local person registry (JSON-backed; see person_registry.py)
import person_registry

# ==========================================
# 1. SETUP CONFIGURATION & INFRASTRUCTURE
# ==========================================
DB_DIR = "my_local_chroma_db"

# Embedding Model (Optimized for documents/forms)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Configure the Text Splitter (Optimized for structured forms/documents)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Small size to keep identity facts isolated
    chunk_overlap=200,     # Small overlap to preserve split-line contexts
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Preserves line breaks in forms
)

# Persistence Store
vector_store = Chroma(
    collection_name="secure_identity_vault",
    embedding_function=embedding_model,
    persist_directory=DB_DIR
)

# Fast Text LLM for query routing + persona matching
classifier_llm = ChatOllama(model="llama3.2", temperature=0.1)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en"
)

def crop_scanned_document(pil_image):

    img = np.array(pil_image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Detect non-white pixels
    mask = gray < 200

    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        return pil_image

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    padding = 50

    x1 = max(0, x1-padding)
    y1 = max(0, y1-padding)
    x2 = min(img.shape[1], x2+padding)
    y2 = min(img.shape[0], y2+padding)

    cropped = img[y1:y2, x1:x2]

    return Image.fromarray(cropped)


def run_ocr_on_image(pil_image, crop: bool = True) -> str:
    """Shared OCR routine used for BOTH standalone images and PDF page
    images, so cropping/resizing/text-joining behavior is identical for
    both input types instead of images going through a separate VLM path.
    """
    if crop:
        pil_image = crop_scanned_document(pil_image)

    pil_image = pil_image.convert("RGB")
    pil_image.thumbnail(
        (2500, 2500),
        Image.Resampling.LANCZOS
    )

    img_array = np.array(pil_image)
    # PaddleOCR accepts a numpy array
    result = ocr.predict(img_array)

    page_text = []
    for block in result:
        for line in block["rec_texts"]:
            page_text.append(line)

    return "\n".join(page_text)


# ==========================================
# 2. FILE EXTRACTION UTILITIES
# ==========================================
def extract_pdf_text(file_path: str) -> str:
    """Safely extracts clean string values from a PDF."""
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text.strip()


def extract_pdf_text_hybrid(file_path: str, min_chars_before_ocr: int = 20) -> str:
    """
    Extracts text from a PDF page-by-page rather than all-or-nothing.

    Multi-page ID documents (e.g. a passport's photo/biodata page vs. its
    address page) often mix an embedded text layer on some pages with pages
    that are purely a scanned image. The previous approach only ran OCR
    when the ENTIRE document came back empty -- so if even one page had a
    little embedded text (a stray MRZ line, a header), OCR was skipped
    entirely and any purely-scanned pages silently lost all their content.

    Here, each page is checked individually: if its embedded text is empty
    or suspiciously short, that specific page (and only that page) is
    rendered to an image and OCR'd, then stitched back in at the right
    position. Pages with a real text layer are left as-is.
    """
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        num_pages = len(reader.pages)
        page_texts = [(page.extract_text() or "").strip() for page in reader.pages]

    combined = []
    for i, page_text in enumerate(page_texts):
        if len(page_text) >= min_chars_before_ocr:
            combined.append(page_text)
            continue

        print(f"-> Page {i + 1}/{num_pages} has little/no embedded text "
              f"({len(page_text)} chars) -- running OCR on this page.")
        try:
            rendered = convert_from_path(
                file_path, dpi=300, first_page=i + 1, last_page=i + 1, thread_count=1
            )
            ocr_text = run_ocr_on_image(rendered[0]) if rendered else ""
            # Keep whichever is longer/non-empty; OCR is expected to win here,
            # but this guards against an OCR call that returns nothing.
            combined.append(ocr_text or page_text)
        except Exception as e:
            print(f"-> OCR failed on page {i + 1}: {e}")
            combined.append(page_text)

    return "\n\n".join(t for t in combined if t.strip())


def extract_image_text_via_ocr(file_path: str) -> str:
    """OCRs a standalone image file (.png/.jpg/.jpeg) using the same
    PaddleOCR engine/pipeline as scanned PDF pages (see run_ocr_on_image).
    """
    with Image.open(file_path) as img:
        img.load()
        return run_ocr_on_image(img)


def extract_pdf_text_via_ocr(file_path: str) -> str:
    """Converts PDF pages to images and extracts text using PaddleOCR."""

    print(f"[1/4] Starting PDF conversion for: {os.path.basename(file_path)}")

    try:
        pages = convert_from_path(
            file_path,
            dpi=300,
            thread_count=2,
        )
        print(f"[2/4] Successfully rendered {len(pages)} page(s).")
    except Exception as e:
        return f"Poppler conversion failed: {e}"

    combined_text = []

    for i, page in enumerate(pages):

        print(f"[3/4] Running OCR on page {i + 1}...")

        try:
            combined_text.append(run_ocr_on_image(page))
            print(f"-> Page {i + 1} complete.")

        except Exception as e:
            print(f"-> OCR failed on page {i + 1}: {e}")
            combined_text.append(f"[OCR Error Page {i + 1}]")

    return "\n\n".join(combined_text)

# ==========================================
# 3. LANGGRAPH ARCHITECTURE (State Machine)
# ==========================================
class AgentState(TypedDict):
    file_path: str
    query: str
    extracted_text: str
    assigned_person: Optional[str]   # person explicitly chosen at upload time (None/"" = untagged)
    filter_person: Optional[str]     # raw name text extracted from the query
    matched_person: Optional[str]    # canonical registered person resolved from filter_person
    search_results: List[Document]


class QueryRouter(BaseModel):
    extracted_person: Optional[str] = Field(
        default=None,
        description="The name or identifier of the person explicitly mentioned in the query text. If they ask about 'john', output 'john'. If no person is specified, output None."
    )


class PersonMatch(BaseModel):
    matched_person: Optional[str] = Field(
        default=None,
        description="The single best-matching canonical person id from the KNOWN PEOPLE list, or None if no reasonable match exists."
    )


def query_routing_node(state: AgentState) -> dict:
    """Node: Automatically populates filter_person by parsing the query text."""
    query = state.get("query", "")
    print(f"-> Query: {query}")

    if not query.strip():
        return {"filter_person": None}

    # Bind structured output to your Llama/text model
    router_llm = classifier_llm.with_structured_output(QueryRouter)

    system_prompt = (
        "You are a search query routing engine. "
        "Identify the person name mentioned in the query. "
        "Normalize names by:"
        "\n- lowercase"
        "\n- replace spaces with underscores"
        "\nExamples:"
        "\nJohn Doe -> john_doe"
        "\njohn doe -> john_doe"
        "\nJOHN -> john"
    )

    try:
        decision = router_llm.invoke(f"{system_prompt}\n\nUser Query: {query}")
        extracted = decision.extracted_person
        print(f"-> Extracted person: {extracted}")
        if extracted:
            extracted = extracted.strip().lower()
    except Exception:
        extracted = None  # Fallback if model fails

    print(f"-> Router deduced target filter_person: '{extracted}'")
    return {"filter_person": extracted}


def persona_match_node(state: AgentState) -> dict:
    """Node: Resolves the raw name text extracted from the query against the
    registry of known, explicitly-tagged people. This replaces the old
    auto-classification step -- instead of guessing who a document belongs
    to, we ask the LLM to pick the best matching *registered* person for
    whatever name the user typed (e.g. 'john' -> 'john_doe'). If nothing
    matches (or no people are registered yet), matched_person is None and
    retrieval falls through to plain semantic search.
    """
    raw_name = state.get("filter_person")
    if not raw_name:
        return {"matched_person": None}

    known_people = person_registry.list_people()
    if not known_people:
        print("-> No registered people yet; skipping persona match.")
        return {"matched_person": None}

    matcher_llm = classifier_llm.with_structured_output(PersonMatch)

    system_prompt = (
        "You are matching a name mentioned in a search query to a canonical list of "
        "known person identifiers. Person identifiers are lowercase, words joined by "
        "underscores (e.g. 'john_doe'). Pick the single best matching identifier from "
        "the KNOWN PEOPLE list for the QUERY NAME below. A partial match, such as "
        "'john' matching 'john_doe', is valid and expected. If nothing in the list "
        "reasonably matches the query name, output None.\n\n"
        f"KNOWN PEOPLE: {known_people}\n"
        f"QUERY NAME: {raw_name}"
    )

    try:
        decision = matcher_llm.invoke(system_prompt)
        matched = decision.matched_person
        if matched:
            matched = matched.strip().lower()
            if matched not in known_people:
                # LLM returned something outside the registry -- don't trust it.
                print(f"-> Persona matcher returned unknown id '{matched}', discarding.")
                matched = None
    except Exception:
        matched = None

    print(f"-> Persona match resolved '{raw_name}' -> '{matched}'")
    return {"matched_person": matched}


def extract_node(state: AgentState) -> dict:
    """Node: Identifies format routing and pulls text raw states."""
    path = state["file_path"]

    ext = os.path.splitext(path)[1].lower()
    extracted = ""

    if ext in [".png", ".jpg", ".jpeg"]:
        extracted = extract_image_text_via_ocr(path)
    elif ext == ".pdf":
        try:
            # Per-page hybrid: OCRs only the specific pages that lack a
            # usable embedded text layer, instead of an all-or-nothing
            # check on the whole document (see extract_pdf_text_hybrid).
            extracted = extract_pdf_text_hybrid(path)
        except Exception as e:
            print(f"-> Per-page hybrid extraction failed ({e}); "
                  f"falling back to full-document OCR.")
            extracted = extract_pdf_text_via_ocr(path)
        # Last-resort fallback if the hybrid path still came back empty
        # (e.g. pypdf couldn't read the file structure at all).
        if not extracted:
            extracted = extract_pdf_text_via_ocr(path)
    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            extracted = f.read()

    return {"extracted_text": extracted}


def index_node(state: AgentState) -> dict:
    """Node: Packs clean string structures into Vector representations with
    targeted metadata. The owning person is now whatever was explicitly
    chosen at upload time (state['assigned_person']) -- there is no more
    LLM-based auto-identification. If no person was chosen, chunks are
    stored untagged so they only ever surface via unfiltered semantic
    search, never via a person filter.
    """
    text_content = state.get("extracted_text", "")
    assigned_person = (state.get("assigned_person") or "").strip().lower()

    if text_content and text_content.strip():
        doc = Document(
            page_content=text_content,
            metadata={
                "source": state["file_path"],
            }
        )

        # Split the document into small chunks
        split_chunks = text_splitter.split_documents([doc])

        if assigned_person:
            # Chroma metadata values must be scalars and there's no
            # substring operator on metadata, so -- same as before -- we
            # expand the assigned person into every alias (full name,
            # underscored form, first name) and store one tagged copy of
            # each chunk per alias. This is what lets a later query for
            # "john" match a document explicitly assigned to "john_doe".
            tagged_chunks = []
            for chunk in split_chunks:
                for alias in generate_person_aliases(assigned_person):
                    tagged_chunk = chunk.model_copy(deep=True)
                    tagged_chunk.metadata["belongs_to_person"] = alias
                    tagged_chunks.append(tagged_chunk)

            vector_store.add_documents(tagged_chunks)

            # Register the person (idempotent) and record this file against them.
            person_registry.add_person(assigned_person)
            person_registry.record_file(assigned_person, state["file_path"])

            print(f"-> Indexed {len(tagged_chunks)} chunks (from {len(split_chunks)} splits) "
                  f"tagged for: {assigned_person}")
        else:
            # No person chosen -- store untagged. These chunks are simply
            # absent from the 'belongs_to_person' field, so a person-filtered
            # search will never match them, but a blanket semantic search will.
            vector_store.add_documents(split_chunks)
            print(f"-> Indexed {len(split_chunks)} chunks (unassigned, no person tag)")

    return {"extracted_text": text_content}  # Return text to keep state pipeline valid


def retrieve_node(state: AgentState) -> dict:
    """Node: Runs similarity indexing based on vectorized queries."""
    query = state.get("query", "")
    target_person = state.get("matched_person", None)
    results = []
    if query:
        if target_person:
            print(f"Looking for -> {target_person}")
            # Chunks were tagged with every alias at index time, and
            # matched_person is guaranteed to be a real registered id, so a
            # plain equality match is sufficient.
            chroma_metadata_filter = {
                "belongs_to_person": {
                    "$eq": target_person
                }
            }

            print("--searching with metadata filter--")
            results = vector_store.similarity_search(
                query=query,
                k=2,
                filter=chroma_metadata_filter,
            )

            # Fallback if filtered search returns nothing
            if not results:
                print(
                    f"--> [FALLBACK ACTIVE] No metadata matches found for '{target_person}'. "
                    "Executing blanket semantic search."
                )
                results = vector_store.similarity_search(
                    query=query,
                    k=2,
                    filter=None,
                )
        else:
            print("--No matched person. Executing global semantic search.--")
            results = vector_store.similarity_search(
                query=query,
                k=2,
                filter=None,
            )

    return {"search_results": results}


def routing_decision_gate(state: AgentState) -> str:
    """Evaluates the state to determine whether to Index files or Search text."""
    # If a file path is provided, prioritize the Ingestion Pipeline
    if state.get("file_path") and state["file_path"].strip():
        print("--> Dynamic Route Selected: [INGESTION / INDEXING]")
        return "extractor"

    # If no file is provided but a query exists, go straight to Search
    if state.get("query") and state["query"].strip():
        print("--> Dynamic Route Selected: [RETRIEVAL / SEARCH]")
        return "router"

    # Fallback to prevent infinite loops if payload is completely empty
    return END


def generate_person_aliases(name: str) -> list[str]:
    """
    Generates searchable aliases for a person's identifier.

    Examples:
        John Doe -> ["john doe", "john_doe", "john"]
        john_doe -> ["john_doe", "john doe", "john"]
        John -> ["john"]
    """
    normalized = name.strip().lower()

    # Handle underscore names
    clean_name = normalized.replace("_", " ")

    aliases = {
        normalized,
        clean_name,
        clean_name.replace(" ", "_"),
    }

    # Add first name only
    first_name = clean_name.split()[0]
    aliases.add(first_name)

    return list(aliases)


# Initialize and connect our production graph
workflow = StateGraph(AgentState)

workflow.add_node("extractor", extract_node)
workflow.add_node("indexer", index_node)
workflow.add_node("router", query_routing_node)
workflow.add_node("persona_matcher", persona_match_node)
workflow.add_node("retriever", retrieve_node)

# Build execution routing pipelines
# Based on the file_path or query, jump to the correct starting node
workflow.add_conditional_edges(
    START,
    routing_decision_gate,
    {
        "extractor": "extractor",  # Branch A: Document Ingestion
        "router": "router",        # Branch B: Question Search
        END: END                   # Branch C: Empty State Exit
    }
)
# RUNTIME INGESTION PATHWAY
# (No more auto-classification step -- the person is provided explicitly
# in state['assigned_person'] from the UI, so text goes straight to indexing.)
workflow.add_edge("extractor", "indexer")
workflow.add_edge("indexer", END)

# RUNTIME QUERY PATHWAY
workflow.add_edge("router", "persona_matcher")
workflow.add_edge("persona_matcher", "retriever")
workflow.add_edge("retriever", END)

local_rag_app = workflow.compile()