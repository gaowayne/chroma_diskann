import argparse
from pprint import pprint

import numpy as np

from chromadb.config import Settings
from chromadb.experimental.diskann import PersistentClient, index_status, rebuild


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and query a real local DiskANN index through Chroma"
    )
    parser.add_argument("--path", default="./diskann-demo-data")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--space", choices=["l2", "cosine"], default="l2")
    arguments = parser.parse_args()
    if arguments.points < 256:
        parser.error("--points must be at least 256 to train the disk index")

    vectors = (
        np.random.default_rng(42).normal(size=(arguments.points, 16)).astype(np.float32)
    )
    settings = Settings(anonymized_telemetry=False)
    client = PersistentClient(arguments.path, settings)
    try:
        collection = client.get_or_create_collection(
            "diskann-demo",
            embedding_function=None,
            metadata={"diskann:space": arguments.space},
        )
        if (collection.metadata or {}).get("diskann:space") != arguments.space:
            raise ValueError(
                "Use a new demo directory when changing the distance function"
            )
        collection.upsert(
            ids=[str(ordinal) for ordinal in range(arguments.points)],
            embeddings=vectors,
            metadatas=[
                {"group": "even" if ordinal % 2 == 0 else "odd"}
                for ordinal in range(arguments.points)
            ],
        )
        print("Before explicit rebuild:")
        pprint(index_status(collection))
        rebuild(collection)
        print("After rebuild:")
        pprint(index_status(collection))
        print("DiskANN query:")
        pprint(collection.query(query_embeddings=vectors[:1], n_results=5))
        print("Metadata-filtered query (exact in this version):")
        pprint(
            collection.query(
                query_embeddings=vectors[:1], n_results=3, where={"group": "odd"}
            )
        )
        collection.delete(ids=["0"])
        collection.upsert(
            ids=["replacement"], embeddings=vectors[:1], metadatas=[{"group": "even"}]
        )
        print("After delete and upsert:")
        pprint(collection.query(query_embeddings=vectors[:1], n_results=1))
    finally:
        client.close()

    reopened = PersistentClient(arguments.path, settings)
    try:
        collection = reopened.get_collection("diskann-demo", embedding_function=None)
        print("After reopening:")
        pprint(index_status(collection))
        pprint(collection.query(query_embeddings=vectors[:1], n_results=1))
    finally:
        reopened.close()


if __name__ == "__main__":
    main()
