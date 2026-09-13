"""Run with python -I in a clean venv, outside the source checkout."""
from importlib import metadata
from pathlib import Path
import sys
from unittest.mock import patch

import cli
import rag
from rag import http_client, remote
from rag.config import load_config

assert metadata.version('axiom-rag') == '1.5.0'
for module in (cli, rag, http_client, remote):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
with patch.dict('os.environ', {'RAG_BASE_URL': 'http://127.0.0.1:8000'}):
    with patch.object(http_client.Client, 'create', return_value={'created': True}) as call:
        assert remote.create_collection(load_config(gemini_api_key='')) == {'created': True}
        call.assert_called_once_with()
print('PASS: installed RAG 1.5.0 CLI/client and HTTP adapter; no provider calls')
