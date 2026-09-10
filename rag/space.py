"""Identity of the canonical Gemini retrieval adapter, not inferred vector provenance."""
from dataclasses import dataclass, asdict
from rag.config import Config


@dataclass(frozen=True)
class EmbeddingSpace:
    provider: str
    model: str
    dimension: int
    schema: int = 1  # Includes document/query preprocessing in rag.embedder.

    @classmethod
    def configured(cls, config: Config):
        if type(config.embedding_dimension) is not int or config.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")
        model = config.embedding_model.removeprefix("models/")
        if not model or model != model.strip() or "/" in model:
            raise ValueError("Embedding model must be a model ID or models/<ID>")
        return cls("google-gemini", model, config.embedding_dimension)

    def metadata(self):
        return {f"rag:embedding:{key}": value for key, value in asdict(self).items()}

    def validate(self, metadata, name):
        expected = self.metadata()
        actual = {key: (metadata or {}).get(key) for key in expected}
        if any(value is None for value in actual.values()):
            raise ValueError(f"Collection {name!r} has unknown embedding provenance; preserve it and use a fresh RAG_COLLECTION")
        if any(type(actual[key]) is not type(value) or actual[key] != value
               for key, value in expected.items()):
            raise ValueError(f"Collection {name!r} has incompatible embedding provenance: stored={actual}, configured={expected}; use a fresh RAG_COLLECTION")

    def validate_vectors(self, vectors):
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError(f"Embedding dimension must equal configured dimension {self.dimension}")
