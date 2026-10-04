import shutil
from pathlib import Path
from typing import Optional, Sequence

from overrides import override

from chromadb.config import Component, System
from chromadb.db.impl.sqlite import SqliteDB
from chromadb.ingest import Consumer
from chromadb.segment import VectorReader
from chromadb.segment.impl.vector.diskann_params import DiskAnnParams
from chromadb.segment.impl.vector.diskann_store import DiskAnnStore
from chromadb.types import (
    LogRecord,
    Metadata,
    RequestVersionContext,
    Segment,
    SeqId,
    VectorEmbeddingRecord,
    VectorQuery,
    VectorQueryResult,
)


class LocalDiskAnnSegment(VectorReader):
    def __init__(self, system: System, segment: Segment):
        Component.__init__(self, system)
        if not system.settings.is_persistent:
            raise ValueError("DiskANN requires a persistent Chroma client")
        self._id = segment["id"]
        self._collection = segment["collection"]
        self._subscription = None
        self._consumer = self.require(Consumer)
        self._db = self.require(SqliteDB)
        self._store = DiskAnnStore(
            Path(system.settings.persist_directory) / str(self._id),
            DiskAnnParams(segment["metadata"] or {}),
        )

    @staticmethod
    @override
    def propagate_collection_metadata(metadata: Metadata) -> Optional[Metadata]:
        return DiskAnnParams.extract(metadata)

    @override
    def start(self) -> None:
        if self._running:
            return
        self._store.open()
        super().start()
        if self._collection is not None:
            self._subscription = self._consumer.subscribe(
                self._collection, self._write_records, start=self.max_seqid()
            )

    @override
    def stop(self) -> None:
        if self._subscription is not None:
            self._consumer.unsubscribe(self._subscription)
            self._subscription = None
        self._store.close()
        super().stop()

    def _write_records(self, records: Sequence[LogRecord]) -> None:
        if not self._running:
            raise RuntimeError("Cannot write to a stopped DiskANN segment")
        self._store.apply(records)
        with self._db.tx() as cursor:
            cursor.execute(
                "INSERT OR REPLACE INTO max_seq_id (segment_id, seq_id) VALUES (?, ?)",
                (self._db.uuid_to_db(self._id), self._store.max_seqid),
            )

    @override
    def max_seqid(self) -> SeqId:
        return self._store.max_seqid

    @override
    def count(self, request_version_context: RequestVersionContext) -> int:
        return self._store.count()

    @override
    def get_vectors(
        self,
        request_version_context: RequestVersionContext,
        ids: Optional[Sequence[str]] = None,
    ) -> Sequence[VectorEmbeddingRecord]:
        vectors = self._store.vectors(ids)
        ordered_ids = list(vectors) if ids is None else ids
        return [
            VectorEmbeddingRecord(id=record_id, embedding=vectors[record_id])
            for record_id in ordered_ids
            if record_id in vectors
        ]

    @override
    def query_vectors(
        self, query: VectorQuery
    ) -> Sequence[Sequence[VectorQueryResult]]:
        results = []
        with self._store._lock:
            for vector in query["vectors"]:
                candidates = self._store.query(vector, query["k"], query["allowed_ids"])
                embeddings = (
                    self._store.vectors([record_id for record_id, _ in candidates])
                    if query["include_embeddings"]
                    else {}
                )
                results.append(
                    [
                        VectorQueryResult(
                            id=record_id,
                            distance=distance,
                            embedding=embeddings.get(record_id),
                        )
                        for record_id, distance in candidates
                    ]
                )
        return results

    def open_persistent_index(self) -> None:
        self._store.open()

    def close_persistent_index(self) -> None:
        self._store.close()

    @override
    def delete(self) -> None:
        self.stop()
        shutil.rmtree(self._store.directory, ignore_errors=False)

    @override
    def reset_state(self) -> None:
        if self._system.settings.allow_reset and self._store.directory.exists():
            self.delete()
