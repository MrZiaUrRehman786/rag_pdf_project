import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Try Streamlit secrets first (for Cloud), fall back to env (for local)
try:
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
