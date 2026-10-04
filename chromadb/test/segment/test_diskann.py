from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chromadb.segment.impl.vector.diskann_params import DiskAnnParams
from chromadb.segment.impl.vector.diskann_store import DiskAnnStore
from chromadb.types import Operation


class FakeDiskIndex:
    def __init__(self, prefix: str):
        self.values = np.load(prefix + ".test.npy")
        self.dimensions = self.values.shape[1]
        self.num_points = len(self.values)
        self.queries = 0

    def query(
        self, vector: Any, num_results: int, search_list_size: int, beam_width: int
    ):
        self.queries += 1
        distances = np.sum((self.values - vector) ** 2, axis=1)
        return [
            (int(ordinal), float(distances[ordinal]))
            for ordinal in np.argsort(distances)[:num_results]
        ]


class FakeNative:
    """Test double for snapshot lifecycle, not a DiskANN implementation."""

    DiskIndex = FakeDiskIndex
    fail = False

    def build_index(self, data_path: str, prefix: str, *args):
        if self.fail:
            raise RuntimeError("injected build failure")
        with open(data_path, "rb") as source:
            count, dimensions = np.fromfile(source, dtype="<u4", count=2)
            np.save(
                prefix + ".test.npy",
                np.fromfile(source, dtype="<f4").reshape((count, dimensions)),
            )


def record(sequence, record_id, vector=None, operation=Operation.ADD):
    return {
        "log_offset": sequence,
        "record": {"id": record_id, "embedding": vector, "operation": operation},
    }


@pytest.mark.parametrize("space", ["l2", "cosine"])
def test_diskann_durable_mutations(tmp_path: Path, space: str):
    params = DiskAnnParams({"diskann:space": space})
    store = DiskAnnStore(tmp_path, params, FakeNative())
    store.apply([record(1, "first", [1, 0]), record(2, "second", [0, 1])])
    assert store.query([1, 0], 10)[0] == ("first", 0.0)
    assert store.query([1, 0], 10, []) == []
    assert store.query([1, 0], 10, ["second", "missing"])[0][0] == "second"
    store.apply(
        [
            record(3, "first", [0, 2], Operation.UPDATE),
            record(4, "second", operation=Operation.DELETE),
        ]
    )
    store.close()
    reopened = DiskAnnStore(tmp_path, params, FakeNative())
    assert reopened.count() == 1
    assert reopened.max_seqid == 4
    np.testing.assert_array_equal(reopened.vectors()["first"], [0, 2])
    reopened.apply([record(1, "first", [1, 0])])
    np.testing.assert_array_equal(reopened.vectors()["first"], [0, 2])
    reopened.close()


def test_diskann_snapshot_delta_and_failed_rebuild(tmp_path: Path):
    native = FakeNative()
    store = DiskAnnStore(
        tmp_path, DiskAnnParams({"diskann:rebuild_threshold": 256}), native
    )
    store.apply(
        [record(ordinal + 1, str(ordinal), [ordinal, 1]) for ordinal in range(256)]
    )
    assert store.status()["mode"] == "diskann+delta"
    generation = store.status()["generation"]
    assert store.query([0, 1], 1)[0] == ("0", 0.0)
    assert store._index.queries == 1
    store.apply(
        [
            record(257, "0", operation=Operation.DELETE),
            record(258, "1", [1000, 1], Operation.UPDATE),
            record(259, "new", [0, 1], Operation.UPSERT),
        ]
    )
    assert store.query([0, 1], 2)[0] == ("new", 0.0)
    assert {entry[0] for entry in store.query([0, 1], 2)} == {"new", "2"}
    native.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        store.rebuild()
    assert store.status()["generation"] == generation
    store.close()
    reopened = DiskAnnStore(tmp_path, store.params, native)
    assert reopened.query([0, 1], 1)[0] == ("new", 0.0)
    reopened.close()


def test_diskann_batch_rollback(tmp_path: Path):
    store = DiskAnnStore(tmp_path, DiskAnnParams({}), FakeNative())
    with pytest.raises(ValueError, match="dimensionality"):
        store.apply([record(1, "first", [1, 2]), record(2, "invalid", [1, 2, 3])])
    assert store.count() == 0
    assert store.max_seqid == -1
    assert store.dimensionality is None
    store.close()


@pytest.mark.parametrize(
    "metadata",
    [
        {"diskann:beam_width": True},
        {"diskann:space": "ip"},
        {"diskann:alpha": float("nan")},
        {"diskann:graph_degree": 100},
    ],
)
def test_diskann_rejects_invalid_configuration(metadata):
    with pytest.raises(ValueError):
        DiskAnnParams(metadata)


def test_diskann_chroma_client_roundtrip(tmp_path, monkeypatch):
    from chromadb.config import Settings
    from chromadb.experimental.diskann import PersistentClient, index_status
    from chromadb.segment.impl.vector import diskann_store

    monkeypatch.setattr(diskann_store, "require_native", lambda: FakeNative())
    client = PersistentClient(tmp_path, Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        "diskann-roundtrip",
        embedding_function=None,
        metadata={"diskann:rebuild_threshold": 256},
    )
    collection.add(
        ids=[str(ordinal) for ordinal in range(256)],
        embeddings=[[float(ordinal), 1.0] for ordinal in range(256)],
        metadatas=[
            {"group": "even" if ordinal % 2 == 0 else "odd"} for ordinal in range(256)
        ],
        documents=[f"document {ordinal}" for ordinal in range(256)],
    )
    assert index_status(collection)["mode"] == "diskann+delta"
    assert collection.query(query_embeddings=[[0, 1]], n_results=1)["ids"] == [["0"]]
    assert collection.query(
        query_embeddings=[[0, 1]], n_results=1, where={"group": "odd"}
    )["ids"] == [["1"]]
    assert collection.query(
        query_embeddings=[[0, 1]], n_results=1, where={"group": "missing"}
    )["ids"] == [[]]
    collection.delete(ids=["0"])
    collection.update(ids=["1"], embeddings=[[1000, 1]])
    assert collection.query(query_embeddings=[[0, 1]], n_results=1)["ids"] == [["2"]]
    client.close()
    reopened = PersistentClient(tmp_path, Settings(anonymized_telemetry=False))
    collection = reopened.get_collection("diskann-roundtrip", embedding_function=None)
    assert collection.count() == 255
    assert collection.query(query_embeddings=[[0, 1]], n_results=1)["ids"] == [["2"]]
    reopened.delete_collection("diskann-roundtrip")
    assert not list(tmp_path.glob("*/diskann.sqlite3"))
    reopened.close()


def test_diskann_configuration_rejected_before_collection_creation(
    tmp_path, monkeypatch
):
    from chromadb.config import Settings
    from chromadb.experimental.diskann import PersistentClient
    from chromadb.segment.impl.vector import diskann_store

    monkeypatch.setattr(diskann_store, "require_native", lambda: FakeNative())
    client = PersistentClient(tmp_path, Settings(anonymized_telemetry=False))
    with pytest.raises(ValueError):
        client.create_collection("invalid-diskann", metadata={"diskann:space": "ip"})
    assert client.count_collections() == 0
    collection = client.create_collection("valid-diskann", embedding_function=None)
    with pytest.raises(ValueError, match="immutable"):
        collection.modify(metadata={"diskann:space": "cosine"})
    with pytest.raises(ValueError, match="HNSW"):
        collection.modify(metadata={"hnsw:search_ef": 8})
    collection.modify(metadata={"description": "updated"})
    assert (
        client.get_collection("valid-diskann").metadata["chroma:vector_index"]
        == "diskann"
    )
    client.close()


def test_diskann_cosine_original_embeddings_and_document_filter(tmp_path, monkeypatch):
    from chromadb.config import Settings
    from chromadb.experimental.diskann import PersistentClient
    from chromadb.segment.impl.vector import diskann_store

    monkeypatch.setattr(diskann_store, "require_native", lambda: FakeNative())
    client = PersistentClient(tmp_path, Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        "cosine-diskann", embedding_function=None, metadata={"diskann:space": "cosine"}
    )
    collection.add(
        ids=["z-last", "a-first"],
        embeddings=[[3, 0], [0, 4]],
        documents=["hello", "world"],
    )
    retrieved = collection.get(include=["embeddings"])
    expected = {"z-last": [3, 0], "a-first": [0, 4]}
    for record_id, vector in zip(retrieved["ids"], retrieved["embeddings"]):
        np.testing.assert_array_equal(vector, expected[record_id])
    result = collection.query(
        query_embeddings=[[2, 0]], n_results=2, include=["distances", "embeddings"]
    )
    assert result["ids"] == [["z-last", "a-first"]]
    np.testing.assert_allclose(result["distances"], [[0, 1]])
    np.testing.assert_array_equal(result["embeddings"][0][0], [3, 0])
    assert collection.query(
        query_embeddings=[[2, 0]], n_results=1, where_document={"$contains": "world"}
    )["ids"] == [["a-first"]]
    for vector in ([0, 0], [float("nan"), 1], [float("inf"), 0]):
        with pytest.raises(ValueError):
            collection.add(ids=["invalid"], embeddings=[vector])
    assert collection.count() == 2
    client.close()


def test_diskann_mutation_semantics_and_empty_collection(tmp_path):
    store = DiskAnnStore(tmp_path, DiskAnnParams({}), FakeNative())
    store.apply(
        [
            record(1, "first", [1, 0]),
            record(2, "first", [2, 0]),
            record(3, "absent", [3, 0], Operation.UPDATE),
            record(4, "first", operation=Operation.UPDATE),
        ]
    )
    assert store.count() == 1
    assert store.query([1, 0], 1) == [("first", 0.0)]
    store.apply([record(5, "first", [2, 0], Operation.UPSERT)])
    assert store.query([1, 0], 1) == [("first", 1.0)]
    store.apply([record(6, "first", operation=Operation.DELETE)])
    assert store.query([1, 0], 1) == []
    store.close()


def test_diskann_missing_snapshot_recovers_exactly(tmp_path):
    import shutil

    store = DiskAnnStore(
        tmp_path, DiskAnnParams({"diskann:rebuild_threshold": 256}), FakeNative()
    )
    store.apply(
        [record(ordinal + 1, str(ordinal), [ordinal, 1]) for ordinal in range(256)]
    )
    generation = store.status()["generation"]
    store.close()
    shutil.rmtree(tmp_path / generation)
    reopened = DiskAnnStore(tmp_path, store.params, FakeNative())
    assert reopened.status()["mode"] == "exact"
    assert reopened.status()["last_error"]
    assert reopened.query([0, 1], 1) == [("0", 0.0)]
    assert reopened.rebuild()
    assert reopened.status()["last_error"] is None
    reopened.close()


def test_diskann_close_on_another_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = DiskAnnStore(tmp_path, DiskAnnParams({}), FakeNative())
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(store.close).result()
    reopened = DiskAnnStore(tmp_path, store.params, FakeNative())
    reopened.close()


def test_diskann_rejects_second_writer(tmp_path):
    from filelock import Timeout

    store = DiskAnnStore(tmp_path, DiskAnnParams({}), FakeNative())
    with pytest.raises(Timeout):
        DiskAnnStore(tmp_path, store.params, FakeNative())
    store.close()


def test_diskann_native_extension_required(tmp_path, monkeypatch):
    from chromadb.experimental.diskann import PersistentClient
    from chromadb.segment.impl.vector import diskann_store

    def missing_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(diskann_store.importlib, "import_module", missing_module)
    with pytest.raises(ImportError, match="optional chroma-diskann-native"):
        PersistentClient(tmp_path)


def test_diskann_does_not_change_default_hnsw(tmp_path):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        str(tmp_path),
        Settings(
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
            anonymized_telemetry=False,
        ),
    )
    collection = client.create_collection("hnsw-regression", embedding_function=None)
    collection.add(ids=["first", "second"], embeddings=[[1, 0], [0, 1]])
    assert collection.query(query_embeddings=[[1, 0]], n_results=1)["ids"] == [
        ["first"]
    ]
    collection.delete(ids=["first"])
    assert collection.query(query_embeddings=[[1, 0]], n_results=1)["ids"] == [
        ["second"]
    ]
    client.close()


def test_diskann_delete_unloaded_collection(tmp_path, monkeypatch):
    from chromadb.config import Settings
    from chromadb.experimental.diskann import PersistentClient
    from chromadb.segment.impl.vector import diskann_store

    monkeypatch.setattr(diskann_store, "require_native", lambda: FakeNative())
    settings = Settings(anonymized_telemetry=False)
    client = PersistentClient(tmp_path, settings)
    collection = client.create_collection("cold-delete", embedding_function=None)
    collection.add(ids=["first"], embeddings=[[1, 0]])
    client.close()
    assert list(tmp_path.glob("*/diskann.sqlite3"))
    reopened = PersistentClient(tmp_path, settings)
    reopened.delete_collection("cold-delete")
    assert not list(tmp_path.glob("*/diskann.sqlite3"))
    assert reopened.count_collections() == 0
    reopened.close()


@pytest.mark.parametrize("space", ["l2", "cosine"])
def test_diskann_real_native_roundtrip(tmp_path, space):
    import os
    from chromadb.segment.impl.vector.diskann_store import require_native

    native = (
        require_native()
        if os.getenv("CHROMA_DISKANN_REQUIRE_NATIVE") == "1"
        else pytest.importorskip(
            "chroma_diskann_native",
            reason="Build the local native extension to run real DiskANN tests",
        )
    )
    from chromadb.config import Settings
    from chromadb.experimental.diskann import PersistentClient, index_status, rebuild

    assert native.BACKEND == "diskann-disk"
    vectors = np.random.default_rng(42).normal(size=(320, 8)).astype(np.float32)
    client = PersistentClient(tmp_path, Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        "native-diskann",
        embedding_function=None,
        metadata={
            "diskann:space": space,
            "diskann:graph_degree": 8,
            "diskann:build_search_list_size": 32,
            "diskann:search_list_size": 128,
            "diskann:num_threads": 1,
            "diskann:pq_bytes": 4,
        },
    )
    collection.add(
        ids=[str(ordinal) for ordinal in range(len(vectors))], embeddings=vectors
    )
    assert rebuild(collection)
    assert index_status(collection)["mode"] == "diskann+delta"
    assert list(tmp_path.glob("*/*/index_disk.index"))
    assert collection.query(query_embeddings=vectors[:1], n_results=1)["ids"] == [["0"]]
    collection.delete(ids=["0"])
    collection.upsert(ids=["replacement"], embeddings=vectors[:1])
    assert collection.query(query_embeddings=vectors[:1], n_results=1)["ids"] == [
        ["replacement"]
    ]
    client.close()
    reopened = PersistentClient(tmp_path, Settings(anonymized_telemetry=False))
    collection = reopened.get_collection("native-diskann", embedding_function=None)
    assert collection.query(query_embeddings=vectors[:1], n_results=1)["ids"] == [
        ["replacement"]
    ]
    reopened.close()
