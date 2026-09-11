import sys
import os

# Ensure the project root is in the Python path (fixes ImportError for `core`)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from core import config
from core.rag_engine import initialize_rag_from_upload, format_docs_with_sources

st.set_page_config(
    page_title="PDF Intelligence Studio",
    page_icon="📚",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #6c757d; font-size: 1rem; margin-bottom: 1.5rem; }
    .stChatMessage { margin-bottom: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_pipeline" not in st.session_state:
    st.session_state["rag_pipeline"] = None
if "current_file_id" not in st.session_state:
    st.session_state["current_file_id"] = None
if "file_stats" not in st.session_state:
    st.session_state["file_stats"] = {}

# Sidebar UI
with st.sidebar:
    st.subheader("🔑 Credentials")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=config.GEMINI_API_KEY,
        type="password",
        help="Paste your Gemini API key from https://aistudio.google.com/"
    )

    st.markdown("---")
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Drop a PDF document here",
        type=["pdf"],
        help="Upload any PDF to parse, index, and query on the fly."
    )

    if uploaded_file and api_key:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state["current_file_id"] != file_signature:
            with st.spinner("Analyzing document and creating vector index..."):
                try:
                    pipeline_data = initialize_rag_from_upload(uploaded_file, api_key)
                    st.session_state["rag_pipeline"] = pipeline_data
                    st.session_state["current_file_id"] = file_signature
                    st.session_state["file_stats"] = {
                        "name": uploaded_file.name,
                        "pages": pipeline_data["total_pages"],
                        "chunks": pipeline_data["total_chunks"]
                    }
                    st.session_state["messages"] = []
                    st.success("Indexing complete!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    if st.session_state["file_stats"]:
        st.markdown("---")
        st.subheader("📊 Document Info")
        stats = st.session_state["file_stats"]
        st.write(f"**File:** `{stats['name']}`")
        st.write(f"**Total Pages:** `{stats['pages']}`")
        st.write(f"**Vector Chunks:** `{stats['chunks']}`")

    st.markdown("---")
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# Main Header
st.markdown('<div class="main-header">📚 Dynamic PDF Question Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents dynamically, ask questions, and inspect verified source passages.</div>', unsafe_allow_html=True)

# Main Chat View
if not uploaded_file:
    st.info("👈 Please upload a PDF in the left sidebar to start chatting.")
elif not api_key:
    st.warning("👈 Please enter your Google Gemini API key in the sidebar.")
else:
    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for c in msg["citations"]:
                        st.markdown(f"**Excerpt {c['chunk_id']} (Page {c['page']})**")
                        st.caption(c["snippet"])
                        st.divider()

    # User Input
    if user_query := st.chat_input("Ask a question about the uploaded document..."):
        # Append and display user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant passages and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document context..."):
                retriever = st.session_state["rag_pipeline"]["retriever"]
                chain = st.session_state["rag_pipeline"]["chain"]

                retrieved_docs = retriever.invoke(user_query)
                context_str, citations = format_docs_with_sources(retrieved_docs)

                response_text = chain.invoke({
                    "context": context_str,
                    "question": user_query
                })

                st.markdown(response_text)
                if citations:
                    with st.expander("🔍 View Retrieved Sources"):
                        for c in citations:
                            st.markdown(f"**Excerpt {c['chunk_id']} (Page {c['page']})**")
                            st.caption(c["snippet"])
                            st.divider()

                # Save assistant response to session state
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": citations
                })