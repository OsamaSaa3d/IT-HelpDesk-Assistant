import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
OLD_TICKETS_DIR = DATA_DIR / "old_tickets"
PROCESSED_DIR = DATA_DIR / "old_tickets_processed"

UNIFIED_CSV_PATH = PROCESSED_DIR / "old_tickets_unified.csv"
DOCUMENTS_JSONL_PATH = PROCESSED_DIR / "old_tickets_documents.jsonl"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_index.flat"
FAISS_IDS_PATH = PROCESSED_DIR / "faiss_ids.npy"
FAISS_META_PATH = PROCESSED_DIR / "faiss_meta.jsonl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 128

DEFAULT_TOP_K = 5

GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

REQUIRED_COLUMNS = [
    "ticket_id",
    "issue",
    "description",
    "resolution",
    "category",
    "resolved",
]

OPTIONAL_COLUMNS = [
    "date",
    "agent_name",
    "source_file",
]


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

