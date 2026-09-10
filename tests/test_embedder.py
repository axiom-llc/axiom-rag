"""Retired model configuration fails before credentials reach the provider."""
from unittest.mock import patch

import pytest

from rag import embedder
from rag.config import load_config


@pytest.mark.parametrize("model", ["text-embedding-004", "models/text-embedding-004"])
@pytest.mark.parametrize("operation,value", [(embedder.embed_texts, ["document"]), (embedder.embed_query, "question")])
def test_retired_model_requires_explicit_migration(model, operation, value):
    config = load_config(gemini_api_key="test-key", embedding_model=model)
    with patch("rag.embedder.genai.Client") as client:
        with pytest.raises(ValueError, match="fresh RAG_COLLECTION"):
            operation(value, config)
    client.assert_not_called()
