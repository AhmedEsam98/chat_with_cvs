import os
from dotenv import load_dotenv

load_dotenv(override=True)

# App settings
MIN_CVS = 8
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_DIM = 1536          # text-embedding-3-small; change for other models
EMBED_BATCH_SIZE = 16

# Azure OpenAI
AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
CHAT_MODEL = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
EMBED_MODEL = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# Azure AI Search
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX"]

# Azure Blob Storage
STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER = os.environ["AZURE_STORAGE_CONTAINER"]

# Redis & Semantic Cache
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # Default 1 hour
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() in ("true", "1", "yes")
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))  # Cosine similarity cutoff

# Re-ranker (FlashRank Cross-Encoder)
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))  # Retrieve top N candidates from search before re-ranking to k