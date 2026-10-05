"""Local PersistentClient DiskANN smoke test for Linux.

Usage (from chroma_diskann repo root, after `pip install -e .`):

    PERSIST_DIR=./chroma-diskann-testdata python examples/diskann/local_persistent.py

The persist directory is kept so you can inspect native DiskANN files.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb


def _native_dirs(persist_dir: Path) -> list[Path]:
    return [path for path in persist_dir.rglob("native") if path.is_dir()]


def main() -> None:
    persist_dir = Path(os.environ.get("PERSIST_DIR", "./chroma-diskann-testdata")).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"persist dir: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    if "diskann_demo" in [c.name for c in client.list_collections()]:
        client.delete_collection("diskann_demo")

    col = client.create_collection(
        name="diskann_demo",
        configuration={"diskann": {"graph_degree": 32, "search_list_size": 64}},
        embedding_function=None,
    )
    print("created collection with diskann config")

    print("\n=== step 1: small N (<256), exact scan ===")
    col.add(ids=["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]])
    small = col.query(query_embeddings=[[0.1, 0.2]], n_results=1)
    print("small-n query ids:", small["ids"])
    print("native dirs after small add:", _native_dirs(persist_dir))

    print("\n=== step 2: upsert 256 vectors, native DiskANN build + search ===")
    ids = [f"id-{i}" for i in range(256)]
    embeddings = [[float(i), float(i + 1)] for i in range(256)]
    col.upsert(ids=ids, embeddings=embeddings)
    large = col.query(query_embeddings=[[0.0, 1.0]], n_results=3)
    print("large-n query ids:", large["ids"])

    native_dirs = _native_dirs(persist_dir)
    print("native index dirs:", native_dirs)
    for native in native_dirs:
        print("native files:", sorted(p.name for p in native.iterdir()))

    if not native_dirs:
        raise SystemExit(
            "FAIL: expected persist_dir/<segment_id>/native/ after 256 upserts"
        )
    print("\nOK")


if __name__ == "__main__":
    main()
