import os
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from core import config
from core.rag_engine import format_docs_with_sources, initialize_rag_from_upload

st.set_page_config(
    page_title="DocuSense AI | Document Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar visual tweaks */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1;
    }

    /* App Header Styling */
    .brand-container {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .brand-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }

    /* Metric card */
    .stat-badge {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        margin-bottom: 0.4rem;
    }
    .stat-row:last-child {
        margin-bottom: 0;
    }
    .stat-label { color: #94a3b8; }
    .stat-val { color: #f8fafc; font-weight: 600; }

    /* Empty state canvas */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: #f8fafc;
        border: 1.5px dashed #cbd5e1;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .empty-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1rem;
    }
    .empty-desc {
        color: #64748b;
        font-size: 0.9rem;
        max-width: 480px;
        margin: 0.5rem auto 0 auto;
        line-height: 1.5;
    }

    /* Source Citation Card */
    .citation-box {
        background-color: #f1f5f9;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #334155;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Secure API Key resolution: Environment Variable or Streamlit Secrets
api_key = os.getenv("GOOGLE_API_KEY") or getattr(config, "GEMINI_API_KEY", "")

# Initialize State
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_pipeline" not in st.session_state:
    st.session_state["rag_pipeline"] = None
if "current_file_id" not in st.session_state:
    st.session_state["current_file_id"] = None
if "file_stats" not in st.session_state:
    st.session_state["file_stats"] = {}
if "sample_prompt" not in st.session_state:
    st.session_state["sample_prompt"] = None

# Sidebar
with st.sidebar:
    st.markdown("### 📑 **DocSense By Zia**")
    st.caption("Enterprise Document Intelligence")
    st.markdown("---")

    st.markdown("#### **Upload Document**")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload standard PDF reports, contracts, manuals, or documents.",
        label_visibility="collapsed",
    )

    # Missing API Key Warning for Administrators
    if not api_key:
        st.error(
            "⚠️ **API Key missing:** Please define `GOOGLE_API_KEY` in your `.env` file or Streamlit secrets."
        )

    # Process Document Indexing
    if uploaded_file and api_key:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state["current_file_id"] != file_signature:
            with st.spinner("Indexing vector representations..."):
                try:
                    pipeline_data = initialize_rag_from_upload(
                        uploaded_file, api_key
                    )
                    st.session_state["rag_pipeline"] = pipeline_data
                    st.session_state["current_file_id"] = file_signature
                    st.session_state["file_stats"] = {
                        "name": uploaded_file.name,
                        "pages": pipeline_data["total_pages"],
                        "chunks": pipeline_data["total_chunks"],
                        "size_mb": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                    }
                    st.session_state["messages"] = []
                    st.toast("Document indexed successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    # Document Status Card
    if st.session_state["file_stats"]:
        stats = st.session_state["file_stats"]
        st.markdown("---")
        st.markdown("#### **Active Index**")
        st.markdown(
            f"""
            <div class="stat-badge">
                <div class="stat-row">
                    <span class="stat-label">File</span>
                    <span class="stat-val" style="max-width:140px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">{stats['name']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Size</span>
                    <span class="stat-val">{stats['size_mb']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Pages</span>
                    <span class="stat-val">{stats['pages']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Index Chunks</span>
                    <span class="stat-val">{stats['chunks']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🧹 Reset Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("🔄 Reload Index", use_container_width=True):
            st.session_state["current_file_id"] = None
            st.rerun()

# Main Interface Header
st.markdown(
    """
    <div class="brand-container">
        <div class="brand-title">📚 Document Intelligence Studio</div>
        <div class="brand-subtitle">Ask direct questions, review context references, and synthesize verified insights from your documents.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty State / Landing View
if not uploaded_file:
    st.markdown(
        """
        <div class="empty-state">
            <span style="font-size: 3rem;">📂</span>
            <div class="empty-title">No document currently loaded</div>
            <div class="empty-desc">
                Upload a PDF file using the left sidebar to generate vector embeddings and begin querying the document.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not api_key:
    st.warning(
        "Service credentials are not configured. Please supply a valid `GOOGLE_API_KEY` in the environment configuration."
    )
else:
    # Quick Starter Questions (Show if chat is empty)
    if len(st.session_state["messages"]) == 0:
        st.markdown("##### 💡 Suggested Prompts")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button(
                "📌 Summarize Key Points",
                key="btn_s1",
                use_container_width=True,
            ):
                st.session_state["sample_prompt"] = (
                    "Can you provide a structured summary of the key takeaways from this document?"
                )
        with p_col2:
            if st.button(
                "📋 List Action Items / Findings",
                key="btn_s2",
                use_container_width=True,
            ):
                st.session_state["sample_prompt"] = (
                    "What are the main findings, obligations, or actionable items mentioned?"
                )
        with p_col3:
            if st.button(
                "🔍 Extract Core Figures & Metrics",
                key="btn_s3",
                use_container_width=True,
            ):
                st.session_state["sample_prompt"] = (
                    "Extract any significant statistics, metrics, dates, or financial figures present in the text."
                )

    # Render Conversation History
    for msg in st.session_state["messages"]:
        with st.chat_message(
            msg["role"],
            avatar="🧑‍💻" if msg["role"] == "user" else "🤖",
        ):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("🔍 Verified Source Passages"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <strong>Excerpt {c['chunk_id']} (Page {c['page']})</strong><br>
                                {c['snippet']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Determine prompt from chat input or button click
    active_prompt = st.chat_input("Ask a question about the document...")
    if st.session_state.get("sample_prompt"):
        active_prompt = st.session_state.pop("sample_prompt")

    # Processing Queries
    if active_prompt:
        st.session_state["messages"].append(
            {"role": "user", "content": active_prompt}
        )
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(active_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing document context..."):
                retriever = st.session_state["rag_pipeline"]["retriever"]
                chain = st.session_state["rag_pipeline"]["chain"]

                retrieved_docs = retriever.invoke(active_prompt)
                context_str, citations = format_docs_with_sources(
                    retrieved_docs
                )

                response_text = chain.invoke(
                    {"context": context_str, "question": active_prompt}
                )

                st.markdown(response_text)

                if citations:
                    with st.expander("🔍 Verified Source Passages"):
                        for c in citations:
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <strong>Excerpt {c['chunk_id']} (Page {c['page']})</strong><br>
                                    {c['snippet']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "citations": citations,
                    }
                )