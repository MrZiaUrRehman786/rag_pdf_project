import os
from dotenv import load_dotenv

# Load variables from .env file if available
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "REDACTED")
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
