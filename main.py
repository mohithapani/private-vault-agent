import os
import base64
from typing import TypedDict, List
from io import BytesIO
#from PIL import Image
import pypdf

# LangChain / Community Modules
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# LangGraph Modules
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. SETUP CONFIGURATION & INFRASTRUCTURE
# ==========================================
DB_DIR = "my_local_chroma_db"

# Embedding Model (Optimized for documents/forms)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Persistence Store
vector_store = Chroma(
    collection_name="secure_identity_vault",
    embedding_function=embedding_model,
    persist_directory=DB_DIR
)

# Vision LLM for Image Extraction
vision_llm = ChatOllama(model="llava", temperature=0)


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


def extract_image_text_via_vlm(file_path: str) -> str:
    """Encodes images to Base64 and hands off to Llava via LangChain."""
    with open(file_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Structure prompt mimicking native LangChain ChatMessage schemas
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Act as an OCR and identity document data extractor. Transcribe all visible "
                    "text exactly. Extract full names, specific travel/employment dates, passport "
                    "or legal ID numbers, and employer names. Avoid generalized summaries."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    response = vision_llm.invoke(prompt)
    return response.content


# ==========================================
# 3. LANGGRAPH ARCHITECTURE (State Machine)
# ==========================================
class AgentState(TypedDict):
    file_path: str
    query: str
    extracted_text: str
    search_results: List[Document]


def extract_node(state: AgentState) -> dict:
    """Node: Identifies format routing and pulls text raw states."""
    path = state["file_path"]

    # Fix applied here: index [1] grabs the extension string directly
    ext = os.path.splitext(path)[1].lower()
    extracted = ""

    if ext in [".png", ".jpg", ".jpeg"]:
        extracted = extract_image_text_via_vlm(path)
    elif ext == ".pdf":
        extracted = extract_pdf_text(path)
    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            extracted = f.read()

    return {"extracted_text": extracted}


def index_node(state: AgentState) -> dict:
    """Node: Packs clean string structures into Vector representations."""
    text_content = state.get("extracted_text", "")
    if text_content and text_content.strip():
        doc = Document(
            page_content=text_content,
            metadata={"source": state["file_path"]}
        )
        vector_store.add_documents([doc])
        print(f"-> Indexed data into local database from: {state['file_path']}")
    return {"extracted_text": text_content}  # Return text to keep state pipeline valid


def retrieve_node(state: AgentState) -> dict:
    """Node: Runs similarity indexing based on vectorized queries."""
    query = state.get("query", "")
    results = []
    if query:
        results = vector_store.similarity_search(query, k=2)
    return {"search_results": results}


# Initialize and connect our production graph
workflow = StateGraph(AgentState)

workflow.add_node("extractor", extract_node)
workflow.add_node("indexer", index_node)
workflow.add_node("retriever", retrieve_node)

# Build execution routing pipelines
workflow.add_edge(START, "extractor")
workflow.add_edge("extractor", "indexer")
workflow.add_edge("indexer", "retriever")
workflow.add_edge("retriever", END)

local_rag_app = workflow.compile()

# ==========================================
# 4. LIVE RUN EXAMPLES
# ==========================================
if __name__ == "__main__":
    print("Hello Private Vault")