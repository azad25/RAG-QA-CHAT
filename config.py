import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# HuggingFace
LLAMA_MODEL = "moonshotai/Kimi-K2-Thinking:novita"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
COLLECTION_NAME = "xyz_real_estate"
