import os
from dotenv import load_dotenv

# Load variables from .env file if available
load_dotenv()

# Read API key from environment. NEVER hardcode it here.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model & chunking configuration
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4