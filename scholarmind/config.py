"""
ScholarMind — Configuration
Central configuration for all project constants.
"""

import os

# ── Endee Vector Database ──────────────────────────────────────────
ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost")
ENDEE_PORT = int(os.getenv("ENDEE_PORT", "8080"))
ENDEE_BASE_URL = f"{ENDEE_HOST}:{ENDEE_PORT}/api/v1"
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")

# ── Index Configuration ────────────────────────────────────────────
PAPERS_INDEX_NAME = "scholarmind_papers"
PAPERS_HYBRID_INDEX_NAME = "scholarmind_papers_hybrid"

# Embedding dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
EMBEDDING_DIMENSION = 384

# Sparse vector dimension for TF-IDF (hybrid search)
SPARSE_DIMENSION = 10000

# HNSW search parameters
SPACE_TYPE = "cosine"
DEFAULT_TOP_K = 10
DEFAULT_EF = 128

# ── Embedding Model ───────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Gemini (RAG) ──────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ── Data Paths ────────────────────────────────────────────────────
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "papers.json")

# ── Categories ────────────────────────────────────────────────────
VALID_CATEGORIES = [
    "machine_learning",
    "natural_language_processing",
    "computer_vision",
    "reinforcement_learning",
    "ai_safety",
    "generative_ai",
    "robotics",
    "graph_neural_networks",
]

VALID_AREAS = [
    "nlp", "cv", "rl", "ml", "ai", "dl",
    "transformers", "diffusion", "gan",
    "optimization", "representation_learning",
]
