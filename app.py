import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


# Import your compiled LangGraph workflow from main.py
from main import local_rag_app
import person_registry

# ==========================================
# 1. APPLICATION & INFRASTRUCTURE SETUP
# ==========================================
DB_DIR = "my_local_chroma_db"
st.set_page_config(page_title="Local Doc AI", page_icon="🗂️", layout="wide")

# Connect to the persistent Nomic vector store instance
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="secure_identity_vault",
    embedding_function=embedding_model,
    persist_directory= DB_DIR
)

# Fast, high-quality local answering engine
chat_llm = ChatOllama(model="llama3.2", temperature=0.1)


# Establish a temporary workspace directory for saving the uploaded files
UPLOAD_DIR = "streamlit_workspace"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Sentinel values for the person-selection dropdown
NO_PERSON_OPTION = "-- No specific person (leave untagged) --"
NEW_PERSON_OPTION = "+ Create new person"

# ==========================================
# 2. SIDEBAR FILE UPLOADER CONTROL PANEL
# ==========================================
with st.sidebar:
    st.header("🎛️ Document Control Panel")
    st.write(
        "Upload new identity forms, travel receipts, or employment history files directly into your local database.")

    # Pull the current known-people list fresh on every render so a
    # newly-created person shows up in the dropdown right away.
    known_people = person_registry.list_people()
    person_options = [NO_PERSON_OPTION, NEW_PERSON_OPTION] + known_people

    st.subheader("👤 Assign to person")
    st.caption("This selection applies to every file in this upload batch.")
    selected_option = st.selectbox(
        "Who do these files belong to?",
        options=person_options,
        index=0,
    )

    new_person_name = ""
    if selected_option == NEW_PERSON_OPTION:
        new_person_name = st.text_input("New person's name (e.g. 'John Doe')")

    # Core drag-and-drop widget mapping your allowed file extensions
    uploaded_files = st.file_uploader(
        "Choose local documents:",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md"],
        accept_multiple_files=True
    )

    # Execution process trigger
    if st.button("🚀 Index Uploaded Files", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file first.")
        else:
            # Resolve the batch-level person assignment once, up front.
            assigned_person = None
            valid_selection = True

            if selected_option == NO_PERSON_OPTION:
                assigned_person = None
            elif selected_option == NEW_PERSON_OPTION:
                if not new_person_name.strip():
                    st.warning("Please enter a name for the new person.")
                    valid_selection = False
                else:
                    assigned_person = person_registry.add_person(new_person_name)
            else:
                # An existing registered person was picked from the dropdown.
                assigned_person = selected_option

            if valid_selection:
                for uploaded_file in uploaded_files:
                    # Save the in-memory uploaded file bytes to your temporary workspace path
                    temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Show status tracker to the user
                    with st.spinner(f"Ingesting: {uploaded_file.name}..."):
                        try:
                            # Feed the local file path into your LangGraph architecture
                            inputs = {
                                "file_path": temp_file_path,
                                "query": "",
                                "extracted_text": "",
                                "assigned_person": assigned_person,
                                "filter_person": None,
                                "matched_person": None,
                                "search_results": []
                            }
                            local_rag_app.invoke(inputs)
                            label = assigned_person if assigned_person else "no specific person"
                            st.success(f"Successfully Indexed: {uploaded_file.name} (assigned to: {label})")
                        except Exception as e:
                            st.error(f"Failed parsing {uploaded_file.name}: {str(e)}")

                # Refresh the sidebar so a newly-created person appears in the dropdown.
                st.rerun()

    if known_people:
        with st.expander(f"📋 Known people ({len(known_people)})"):
            for person_id in known_people:
                st.write(f"- `{person_id}`")

# ==========================================
# 3. INTERACTIVE MAIN CHAT CANVAS
# ==========================================
st.title("🗂️ Local Document Chatbot")
st.caption("Query your local secure vector vault in real time using private AI models.")

# Initialize the chat message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to search your uploaded local documents. Ask me anything!"}
    ]

# Render conversational history across state changes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User interactive messaging query loop
if user_query := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Response generation pipeline
    with st.chat_message("assistant"):
        with st.spinner("Scanning database and generating answer..."):
            try:
                inputs = {
                    "file_path": "",
                    "query": user_query,
                    "extracted_text": "",
                    "assigned_person": None,
                    "filter_person": None,
                    "matched_person": None,
                    "search_results": []
                }

                result = local_rag_app.invoke(inputs)

                search_results = result["search_results"]

                if not search_results:
                    response_text = "No records matching that request were located inside your secure document database."
                else:
                    # 2. CONSTRUCT context text from references
                    context_str = "\n\n".join([doc.page_content for doc in search_results])

                    # 3. BUILD a strict prompt forcing the model to stick to the facts
                    system_prompt = (
                        "You are a helpful assistant. Use only the provided document snippets to answer the user's question. "
                        "If the answer cannot be found in the context, say that you don't know.\n\n"
                        f"--- DOCUMENT SNIPPETS ---\n{context_str}\n\n"
                        f"--- USER QUESTION ---\n{user_query}"
                    )

                    # 4. GENERATE the direct answer using your local model
                    ai_reply = chat_llm.invoke(system_prompt)

                    # 5. FORMAT output cleanly showing the source files at the bottom
                    response_text = ai_reply.content + "\n\n---\n### 📄 Sources Referenced:\n"
                    seen_sources = set()
                    for doc in search_results:
                        src_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        if src_name not in seen_sources:
                            response_text += f"- `{src_name}`\n"
                            seen_sources.add(src_name)

            except Exception as e:
                response_text = f"An infrastructure retrieval failure occurred: {str(e)}"

            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})