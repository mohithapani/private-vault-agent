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

# Fast Text LLM for Structured Person Classification
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


# Pydantic Schema to force the LLM to return clean person lists
class IdentityClassification(BaseModel):
    detected_people: List[str] = Field(
        description="List of full names or clean identity strings found in the document. Returns empty list if none."
    )


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


def extract_image_text_via_ocr(file_path: str) -> str:
    """OCRs a standalone image file (.png/.jpg/.jpeg) using the same
    PaddleOCR engine/pipeline as scanned PDF pages (see run_ocr_on_image).
    Replaces the previous llava/vision-LLM based extraction.
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
    filter_person: Optional[str]
    identified_people: List[str]
    search_results: List[Document]

class QueryRouter(BaseModel):
    extracted_person: Optional[str] = Field(
        default=None,
        description="The name or identifier of the person explicitly mentioned in the query text. If they ask about 'john', output 'john'. If no person is specified, output None."
    )


def query_routing_node(state: AgentState) -> dict:
    """Node: Automatically populates filter_person by parsing the query."""
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


def extract_node(state: AgentState) -> dict:
    """Node: Identifies format routing and pulls text raw states."""
    path = state["file_path"]

    # Fix applied here: index [1] grabs the extension string directly
    ext = os.path.splitext(path)[1].lower()
    extracted = ""

    if ext in [".png", ".jpg", ".jpeg"]:
        extracted = extract_image_text_via_ocr(path)
    elif ext == ".pdf":
        extracted = extract_pdf_text(path)
        # Fallback for reading pdf's which are scanned copies
        if not extracted:
            extracted = extract_pdf_text_via_ocr(path)
    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            extracted = f.read()

    return {"extracted_text": extracted}


def classify_identity_node(state: AgentState) -> dict:
    """Node: Analyzes extracted text to parse exactly who this document belongs to."""
    text_content = state.get("extracted_text", "")

    if not text_content.strip():
        return {"identified_people": ["unknown"]}

    # Force the text model to strictly output the structured JSON array
    structured_llm = classifier_llm.with_structured_output(IdentityClassification)

    system_prompt = (
        "You are an identity classification engine. Analyze the provided text and "
        "extract the names of all unique individuals whom this bill, form, or document "
        "belongs to. Normalize the output text names to lowercase words joined by underscores "
        "(e.g., 'john_doe', 'person_x'). If you find an email address like 'john_doe@gmail.com', "
        "strip the domain and clean it to a standardized identifier like 'john_doe'"
    )

    try:
        result = structured_llm.invoke(f"{system_prompt}\n\nDocument Text:\n{text_content}")
        people = result.detected_people if result.detected_people else ["unknown"]

        # Normalize names
        people = [
            person.strip().lower().replace(" ", "_")
            for person in people
        ]
    except Exception:
        # Fallback if structured output fails locally
        people = ["unknown"]

    print(f"-> Classified Document Ownership: {people}")
    return {"identified_people": people}


def index_node(state: AgentState) -> dict:
    """Node: Packs clean string structures into Vector representations with targeted metadata."""
    text_content = state.get("extracted_text", "")
    people_tags = state.get("identified_people", ["unknown"])

    if text_content and text_content.strip():
        # Chroma metadata values must be scalars (str/int/float/bool) -- it
        # cannot store a Python list, and it has NO substring / "$contains"
        # operator for metadata (that operator only applies to page_content
        # via where_document). Storing "john_doe john doe john" as one big
        # string and filtering with $contains silently matches nothing.
        #
        # Fix: expand each person into all of their aliases HERE (at index
        # time) and store one canonical scalar id per tagged chunk. Then a
        # simple $eq at query time is enough (see retrieve_node).
        primary_people = [person.strip().lower() for person in people_tags]

        doc = Document(
            page_content=text_content,
            metadata={
                "source": state["file_path"],
            }
        )

        # Split the document into small chunks
        split_chunks = text_splitter.split_documents([doc])

        # Index one tagged copy of each chunk per ALIAS of each person, so a
        # partial-name query ("mohit") still finds a document classified
        # under the full name ("mohit_hapani"). Aliases must be expanded
        # here at index time -- expanding at query time can't reconstruct
        # "mohit_hapani" from just "mohit".
        tagged_chunks = []
        for chunk in split_chunks:
            for person in primary_people:
                for alias in generate_person_aliases(person):
                    tagged_chunk = chunk.model_copy(deep=True)
                    tagged_chunk.metadata["belongs_to_person"] = alias
                    tagged_chunks.append(tagged_chunk)

        vector_store.add_documents(tagged_chunks)
        print(f"-> Indexed {len(tagged_chunks)} chunks (from {len(split_chunks)} splits) "
              f"tagged for: {primary_people}")
    return {"extracted_text": text_content}  # Return text to keep state pipeline valid


def retrieve_node(state: AgentState) -> dict:
    """Node: Runs similarity indexing based on vectorized queries."""
    query = state.get("query", "")
    target_person = state.get("filter_person", None)
    results = []
    if query:
        # Build standard metadata filter dict dynamically
        if target_person:
            normalized_target = target_person.strip().lower()
            print(f"Looking for -> {normalized_target}")
            # Chunks were tagged with every alias at index time, so a plain
            # equality match on the normalized query term is sufficient.
            chroma_metadata_filter = {
                "belongs_to_person": {
                    "$eq": normalized_target
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
            print("--No target person provided. Executing global semantic search.--")
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
workflow.add_node("classifier", classify_identity_node)
workflow.add_node("indexer", index_node)
workflow.add_node("router", query_routing_node)
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
workflow.add_edge("extractor", "classifier")  # Route text to classifier first
workflow.add_edge("classifier", "indexer")    # Feed classification data into storage indexing
workflow.add_edge("indexer", END)

# RUNTIME QUERY PATHWAY
workflow.add_edge("router", "retriever")
workflow.add_edge("retriever", END)

local_rag_app = workflow.compile()