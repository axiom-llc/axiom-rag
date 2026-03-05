#!/usr/bin/env python3
"""
rag — Retrieval-Augmented Generation CLI

Usage:
  rag ingest <file_or_directory> [--strategy fixed|sentences]
  rag query  "<question>"
  rag list
  rag delete <doc_id>
  rag stats

Environment:
  GEMINI_API_KEY   Required for ingest and query.
  (See .env.example for full configuration options.)
"""
import sys
import json
from pathlib import Path

from rag.config import load_config
from rag import pipeline, store


def _usage(msg: str | None = None) -> None:
    if msg:
        print(f"Error: {msg}", file=sys.stderr)
    print(__doc__)
    sys.exit(1 if msg else 0)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _usage()

    cmd, *rest = args
    # Config is always the same call; API key validation is deferred to
    # embed/generate operations, so store-only commands work without a key.
    config = load_config()

    if cmd == "ingest":
        if not rest:
            _usage("ingest requires a file or directory path")
        target = rest[0]
        strategy = "fixed"
        if "--strategy" in rest:
            idx = rest.index("--strategy")
            try:
                strategy = rest[idx + 1]
            except IndexError:
                _usage("--strategy requires a value: fixed or sentences")
        p = Path(target).expanduser()
        if p.is_dir():
            results = pipeline.ingest_directory(str(p), config, strategy=strategy)
            for r in results:
                print(f"  ✓  {r['doc_id']}  —  {r['chunks_stored']} chunks")
            print(f"\nIngested {len(results)} document(s).")
        else:
            r = pipeline.ingest_file(str(p), config, strategy=strategy)
            print(f"  ✓  {r['doc_id']}  —  {r['chunks_stored']} chunks")

    elif cmd == "query":
        if not rest:
            _usage("query requires a question string")
        question = " ".join(rest)
        result = pipeline.query(question, config)
        print(f"\n{result['answer']}\n")
        print(f"Sources ({result['chunk_count']} chunks):  {', '.join(result['sources']) or 'none'}")

    elif cmd == "list":
        docs = store.list_documents(config)
        if not docs:
            print("No documents ingested.")
        else:
            for d in docs:
                print(f"  {d}")

    elif cmd == "delete":
        if not rest:
            _usage("delete requires a doc_id")
        doc_id = rest[0]
        n = store.delete_document(doc_id, config)
        print(f"Deleted {n} chunk(s) for '{doc_id}'.")

    elif cmd == "stats":
        print(json.dumps(store.collection_stats(config), indent=2))

    else:
        _usage(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
