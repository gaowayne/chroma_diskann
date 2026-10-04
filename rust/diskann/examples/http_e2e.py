"""Exercise the normal Python Chroma HTTP API against a DiskANN-bound demo collection."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError


def expect_rejection(operation: Callable[[], object], message: str) -> None:
    try:
        operation()
    except ChromaError as error:
        if message.lower() not in str(error).lower():
            raise AssertionError(f"Unexpected server error: {error}") from error
    else:
        raise AssertionError(f"Expected server rejection containing: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seed", "binding", "verify"])
    parser.add_argument("--collection")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tenant", default="default_tenant")
    parser.add_argument("--database", default="default_database")
    arguments = parser.parse_args()
    if arguments.action == "binding":
        if arguments.snapshot is None:
            parser.error("binding requires --snapshot")
        directory = arguments.snapshot.resolve(strict=True)
        with (directory / "chroma_snapshot.json").open(encoding="utf-8") as source:
            manifest = json.load(source)
        print(json.dumps({manifest["info"]["source"]["collection_id"]: str(directory)}))
        return
    if not arguments.collection:
        parser.error("seed and verify require --collection")
    client = chromadb.HttpClient(
        host=arguments.host,
        port=arguments.port,
        tenant=arguments.tenant,
        database=arguments.database,
    )

    if arguments.action == "seed":
        collection = client.create_collection(arguments.collection)
        generator = random.Random(71)
        vectors = [[generator.uniform(-1.0, 1.0) for _ in range(4)] for _ in range(512)]
        collection.add(ids=[f"doc-{row:04}" for row in range(512)], embeddings=vectors)
        assert collection.count() == 512
        print(
            json.dumps(
                {
                    "status": "seeded",
                    "collection": collection.name,
                    "collection_id": str(collection.id),
                    "points": collection.count(),
                },
                indent=2,
            )
        )
        return

    collection = client.get_collection(arguments.collection)
    source = collection.get(limit=1, include=["embeddings"])
    embeddings = source["embeddings"]
    assert source["ids"] and embeddings is not None
    source_id = source["ids"][0]
    vector = [float(value) for value in embeddings[0]]
    result = collection.query(
        query_embeddings=[vector], n_results=5, include=["distances", "embeddings"]
    )
    assert len(result["ids"]) == 1 and len(result["ids"][0]) == 5
    assert result["ids"][0][0] == source_id
    distances = result["distances"]
    returned_embeddings = result["embeddings"]
    assert distances is not None and abs(distances[0][0]) < 1e-5
    assert returned_embeddings is not None
    assert len(returned_embeddings[0][0]) == len(vector)
    assert all(
        abs(float(actual) - expected) < 1e-6
        for actual, expected in zip(returned_embeddings[0][0], vector)
    )
    expect_rejection(
        lambda: collection.query(
            query_embeddings=[vector],
            n_results=5,
            include=["distances"],
            where={"probe_group": "demo"},
        ),
        "DiskANN snapshot queries do not support filters",
    )
    expect_rejection(
        lambda: collection.add(ids=[source_id], embeddings=[vector]),
        "read-only DiskANN snapshot",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "collection_id": str(collection.id),
                "ids": result["ids"],
                "distances": distances,
                "diskann_filter_guard": "verified",
                "read_only_guard": "verified",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()