# ui.py
import streamlit as st
import requests
import time
import random

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8001"
ASK_URL = f"{API_BASE_URL}/ask"
INGEST_URL = f"{API_BASE_URL}/ingest"
PAGE_TITLE = "EvoRAG 🧠"
PAGE_ICON = "🧠"

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(PAGE_TITLE)


# --- Animation for Ingestion ---
def ingestion_animation(placeholder):
    """Displays a fun, dynamic animation while a document is being processed."""
    emojis = ["📄", "➡️", "🧠", "🔍", "⚡️", "✨"]
    messages = [
        "Parsing document structure...",
        "Chunking text for optimal retrieval...",
        "Generating embeddings...",
        "Storing vectors in the database...",
        "Optimizing for search...",
        "Almost there..."
    ]

    progress_bar = placeholder.progress(0)

    for i in range(100):
        # Update progress bar
        progress_bar.progress(i + 1)

        # Update text with a fun emoji effect
        message_index = (i // 17) % len(messages)
        emoji_index = (i // 5) % len(emojis)
        placeholder.text(f"{emojis[emoji_index]} {messages[message_index]}")

        # Simulate work being done
        time.sleep(random.uniform(0.05, 0.2))  # Random sleep for a more "real" feel

    placeholder.success("✅ Document processed and ready for questions!")
    time.sleep(2)
    placeholder.empty()


# --- Sidebar for Actions ---
with st.sidebar:
    st.header("Actions")

    # New Chat Button
    if st.button("✨ New Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
        st.rerun()

    st.divider()

    # File Uploader
    st.header("Upload a Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, or other document",
        type=['pdf', 'txt', 'md', 'docx']
    )

    if uploaded_file is not None:
        # Create a placeholder for the animation
        animation_placeholder = st.empty()

        # Show animation while processing
        ingestion_animation(animation_placeholder)

        # Send the file to the backend for ingestion
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            with st.spinner(f"Sending '{uploaded_file.name}' to the backend..."):
                response = requests.post(INGEST_URL, files=files)
                response.raise_for_status()
                st.success(f"Successfully uploaded and ingested '{uploaded_file.name}'!")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to upload document. Error: {e}")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]


# --- Helper Functions (from previous version, slightly modified) ---
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "details" in message:
                with st.expander("Show sources and debug info"):
                    st.json(message["details"])


def handle_user_query(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(ASK_URL, json={"query": prompt})
                response.raise_for_status()
                api_response = response.json()

                full_response = api_response.get("answer", "Sorry, I encountered an error.")
                details = {
                    "Cited Documents": api_response.get("cited_docs"),
                    "All Referenced Documents": api_response.get("referenced_docs"),
                    "Rewritten Query for Retrieval": api_response.get("rewritten_query_for_retrieval")
                }
                st.markdown(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "details": details
                })
                # We need to rerun to properly handle the expander in the new message
                st.rerun()

            except requests.exceptions.RequestException as e:
                error_message = f"Could not connect to the backend API. Error: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


# --- Main Application Logic ---
display_chat_history()

if prompt := st.chat_input("Ask a question about your documents..."):
    handle_user_query(prompt)