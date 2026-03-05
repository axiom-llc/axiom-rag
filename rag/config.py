from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    gemini_api_key: str       # "" acceptable for store-only operations
    chroma_path: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    embedding_model: str
    generation_model: str

    def requires_api_key(self) -> None:
        """Raise ValueError if API key is unset. Call at entry to embed/generate."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for this operation. "
                "Set it in the environment or pass gemini_api_key= to load_config()."
            )


def load_config(**overrides) -> Config:
    """Resolve Config from environment variables with optional programmatic overrides.

    GEMINI_API_KEY is not validated here; validation is deferred to the point of use
    so that store-only CLI commands (list, delete, stats) work without an API key.

    Override precedence: kwargs > environment > hard-coded default.
    """
    def _get(field: str, env_var: str, default, cast=str):
        return cast(overrides.get(field, os.environ.get(env_var, default)))

    return Config(
        gemini_api_key=(
            overrides.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
        ),
        chroma_path=_get("chroma_path",       "RAG_CHROMA_PATH",       "~/.rag/chroma"),
        collection_name=_get("collection_name","RAG_COLLECTION",        "documents"),
        chunk_size=_get("chunk_size",          "RAG_CHUNK_SIZE",        512,  int),
        chunk_overlap=_get("chunk_overlap",    "RAG_CHUNK_OVERLAP",     64,   int),
        top_k=_get("top_k",                    "RAG_TOP_K",             5,    int),
        score_threshold=_get(
            "score_threshold", "RAG_SCORE_THRESHOLD", 0.4, float
        ),
        embedding_model=_get(
            "embedding_model", "RAG_EMBEDDING_MODEL", "models/text-embedding-004"
        ),
        generation_model=_get(
            "generation_model", "RAG_GENERATION_MODEL", "gemini-2.5-flash"
        ),
    )
