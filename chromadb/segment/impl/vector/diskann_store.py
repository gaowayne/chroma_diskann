import heapq
import importlib
import logging
import os
import shutil
import sqlite3
import struct
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

import numpy as np
from filelock import FileLock

from chromadb.segment.impl.vector.diskann_params import DiskAnnParams
from chromadb.types import LogRecord, Operation

logger = logging.getLogger(__name__)


def require_native() -> Any:
    try:
        return importlib.import_module("chroma_diskann_native")
    except ImportError as error:
        raise ImportError(
            "DiskANN requires the optional chroma-diskann-native extension. "
            "Build rust/diskann_bindings with maturin; see docs/diskann.md."
        ) from error


class DiskAnnStore:
    """A durable vector table with atomically published DiskANN snapshots."""

    def __init__(self, directory: Path, params: DiskAnnParams, native: Any = None):
        self.directory = directory
        self.params = params
        self._native = native if native is not None else require_native()
        self._lock = RLock()
        self._file_lock = FileLock(str(directory / "diskann.lock"), thread_local=False)
        self._connection: Optional[sqlite3.Connection] = None
        self._index: Any = None
        self._last_error: Optional[str] = None
        self.open()

    def open(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            self._file_lock.acquire(timeout=0)
            try:
                connection = sqlite3.connect(
                    str(self.directory / "diskann.sqlite3"), check_same_thread=False
                )
                self._connection = connection
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.executescript(
                    "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                    "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector BLOB NOT NULL, revision INTEGER NOT NULL);"
                    "CREATE TABLE IF NOT EXISTS delta (id TEXT PRIMARY KEY);"
                    "CREATE TABLE IF NOT EXISTS snapshot (ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, revision INTEGER NOT NULL);"
                )
                with connection:
                    for key, value in (("format", "1"), ("space", self.params.space)):
                        stored = self._get_state(key)
                        if stored is not None and stored != value:
                            raise ValueError(
                                f"Persisted DiskANN {key} mismatch: {stored} != {value}"
                            )
                        self._set_state(key, value)
                generation = self._get_state("generation")
                if generation is not None:
                    UUID(generation)
                    try:
                        self._index = self._native.DiskIndex(
                            str(self.directory / generation / "index")
                        )
                        if self._index.dimensions != self.dimensionality:
                            raise ValueError("DiskANN snapshot dimension mismatch")
                    except Exception as error:
                        self._index = None
                        self._last_error = str(error)
                        logger.warning(
                            "DiskANN snapshot unavailable; using exact recovery: %s",
                            error,
                        )
            except Exception:
                self.close()
                raise

    @property
    def connection(self) -> sqlite3.Connection:
        self.open()
        assert self._connection is not None
        return self._connection

    def _get_state(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row is not None else None

    def _set_state(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO state VALUES (?, ?)", (key, value)
        )

    @property
    def dimensionality(self) -> Optional[int]:
        value = self._get_state("dimensions")
        return int(value) if value is not None else None

    @property
    def max_seqid(self) -> int:
        with self._lock:
            return int(self._get_state("max_seqid") or "-1")

    def count(self) -> int:
        with self._lock:
            return int(
                self.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            )

    def _validate_vector(self, vector: Any) -> np.ndarray:
        values = np.asarray(vector, dtype="<f4")
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError(
                "DiskANN vectors must be nonempty, finite, one-dimensional float32 arrays"
            )
        dimensions = self.dimensionality
        if dimensions is not None and values.size != dimensions:
            raise ValueError(
                f"Vector dimensionality {values.size} does not match {dimensions}"
            )
        if (
            self.params.space == "cosine"
            and np.linalg.norm(values.astype(np.float64)) == 0
        ):
            raise ValueError("DiskANN cosine vectors must have nonzero norm")
        return values

    def _index_vector(self, vector: np.ndarray) -> np.ndarray:
        if self.params.space == "cosine":
            return np.asarray(
                vector / np.linalg.norm(vector.astype(np.float64)), dtype="<f4"
            )
        return vector

    def apply(self, records: Sequence[LogRecord]) -> None:
        with self._lock:
            connection = self.connection
            with connection:
                last_seqid = self.max_seqid
                for log_record in records:
                    sequence = log_record["log_offset"]
                    if sequence <= last_seqid:
                        continue
                    record = log_record["record"]
                    record_id = record["id"]
                    operation = record["operation"]
                    exists = (
                        connection.execute(
                            "SELECT 1 FROM vectors WHERE id = ?", (record_id,)
                        ).fetchone()
                        is not None
                    )
                    changed = False
                    if operation == Operation.DELETE:
                        if exists:
                            connection.execute(
                                "DELETE FROM vectors WHERE id = ?", (record_id,)
                            )
                            changed = True
                    elif operation in (
                        Operation.ADD,
                        Operation.UPDATE,
                        Operation.UPSERT,
                    ):
                        should_write = (
                            operation == Operation.UPSERT
                            or (operation == Operation.ADD and not exists)
                            or (operation == Operation.UPDATE and exists)
                        )
                        if should_write and record["embedding"] is not None:
                            vector = self._validate_vector(record["embedding"])
                            if self.dimensionality is None:
                                self._set_state("dimensions", str(vector.size))
                            connection.execute(
                                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                                (record_id, vector.tobytes(), sequence),
                            )
                            changed = True
                    else:
                        raise ValueError(f"Unsupported DiskANN operation: {operation}")
                    if changed:
                        connection.execute(
                            "INSERT OR IGNORE INTO delta VALUES (?)", (record_id,)
                        )
                    last_seqid = sequence
                self._set_state("max_seqid", str(last_seqid))
            dirty = connection.execute("SELECT COUNT(*) FROM delta").fetchone()[0]
            if dirty >= self.params.rebuild_threshold and self.count() >= max(
                256, self.params.graph_degree + 1
            ):
                try:
                    self.rebuild()
                except Exception as error:
                    self._last_error = str(error)
                    logger.warning(
                        "DiskANN rebuild failed; committed changes remain queryable: %s",
                        error,
                    )

    def rebuild(self) -> bool:
        with self._lock:
            count = self.count()
            if count < max(256, self.params.graph_degree + 1):
                return False
            generation = str(uuid4())
            destination = self.directory / generation
            destination.mkdir()
            data_path = destination / "vectors.bin"
            published = False
            replacement = None
            try:
                with data_path.open("wb") as output:
                    output.write(struct.pack("<II", count, self.dimensionality))
                    for row in self.connection.execute(
                        "SELECT vector FROM vectors ORDER BY id"
                    ):
                        vector = np.frombuffer(row[0], dtype="<f4")
                        output.write(self._index_vector(vector).tobytes())
                    output.flush()
                    os.fsync(output.fileno())
                prefix = str(destination / "index")
                self._native.build_index(
                    str(data_path),
                    prefix,
                    self.params.graph_degree,
                    self.params.build_search_list_size,
                    self.params.pq_bytes,
                    self.params.num_threads,
                    self.params.alpha,
                )
                for artifact in destination.iterdir():
                    if artifact.is_file():
                        with artifact.open("r+b") as handle:
                            os.fsync(handle.fileno())
                if os.name == "posix":
                    for directory in (destination, self.directory):
                        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                replacement = self._native.DiskIndex(prefix)
                if replacement.dimensions != self.dimensionality:
                    raise ValueError("New DiskANN snapshot dimension mismatch")
                with self.connection:
                    self.connection.execute("DELETE FROM snapshot")
                    self.connection.execute(
                        "INSERT INTO snapshot SELECT ROW_NUMBER() OVER (ORDER BY id) - 1, id, revision FROM vectors"
                    )
                    self.connection.execute("DELETE FROM delta")
                    self._set_state("generation", generation)
                published = True
                self._index = replacement
                self._last_error = None
                data_path.unlink(missing_ok=True)
                self._remove_old_generations(generation)
                return True
            except Exception as error:
                self._last_error = str(error)
                raise
            finally:
                if not published:
                    replacement = None
                    shutil.rmtree(destination, ignore_errors=True)

    def _remove_old_generations(self, current: str) -> None:
        for path in self.directory.iterdir():
            if not path.is_dir() or path.name == current:
                continue
            try:
                UUID(path.name)
            except ValueError:
                continue
            try:
                shutil.rmtree(path)
            except OSError as error:
                logger.warning(
                    "Could not remove obsolete DiskANN generation %s: %s",
                    path.name,
                    error,
                )

    def _rows_for_ids(self, ids: Sequence[str]) -> Iterable[Tuple[str, bytes]]:
        for offset in range(0, len(ids), 500):
            batch = ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            yield from self.connection.execute(
                f"SELECT id, vector FROM vectors WHERE id IN ({placeholders})", batch
            )

    def vectors(self, ids: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        with self._lock:
            rows = (
                self.connection.execute("SELECT id, vector FROM vectors")
                if ids is None
                else self._rows_for_ids(ids)
            )
            return {
                record_id: np.frombuffer(blob, dtype="<f4").copy()
                for record_id, blob in rows
            }

    def _rank(
        self, query: np.ndarray, rows: Iterable[Tuple[str, bytes]], num_results: int
    ) -> List[Tuple[str, float]]:
        query64 = query.astype(np.float64)

        def scores() -> Iterable[Tuple[str, float]]:
            for record_id, blob in rows:
                vector = np.frombuffer(blob, dtype="<f4").astype(np.float64)
                if self.params.space == "cosine":
                    distance = float(
                        np.clip(
                            1.0
                            - np.dot(query64, vector)
                            / (np.linalg.norm(query64) * np.linalg.norm(vector)),
                            0.0,
                            2.0,
                        )
                    )
                else:
                    difference = query64 - vector
                    distance = float(np.dot(difference, difference))
                yield record_id, distance

        return heapq.nsmallest(
            num_results, scores(), key=lambda item: (item[1], item[0])
        )

    def query(
        self, vector: Any, num_results: int, allowed_ids: Optional[Sequence[str]] = None
    ) -> List[Tuple[str, float]]:
        with self._lock:
            query = self._validate_vector(vector)
            if num_results <= 0 or allowed_ids is not None and len(allowed_ids) == 0:
                return []
            num_results = min(num_results, self.count())
            if not num_results:
                return []
            if allowed_ids is not None:
                return self._rank(
                    query,
                    self._rows_for_ids(list(dict.fromkeys(allowed_ids))),
                    num_results,
                )
            if self._index is None:
                return self._rank(
                    query,
                    self.connection.execute("SELECT id, vector FROM vectors"),
                    num_results,
                )
            snapshot_count = self.connection.execute(
                "SELECT COUNT(*) FROM snapshot"
            ).fetchone()[0]
            limit = min(
                snapshot_count, max(num_results * 4, self.params.search_list_size)
            )
            candidates = self._index.query(
                self._index_vector(query).tolist(),
                limit,
                max(limit, self.params.search_list_size),
                self.params.beam_width,
            )

            def rows() -> Iterable[Tuple[str, bytes]]:
                ordinals = list(
                    dict.fromkeys(int(candidate[0]) for candidate in candidates)
                )
                for offset in range(0, len(ordinals), 500):
                    batch = ordinals[offset : offset + 500]
                    placeholders = ",".join("?" for _ in batch)
                    yield from self.connection.execute(
                        "SELECT vectors.id, vectors.vector FROM snapshot JOIN vectors "
                        "ON snapshot.id = vectors.id AND snapshot.revision = vectors.revision "
                        f"WHERE snapshot.ordinal IN ({placeholders})",
                        batch,
                    )
                yield from self.connection.execute(
                    "SELECT vectors.id, vectors.vector FROM delta JOIN vectors ON delta.id = vectors.id"
                )

            result = self._rank(query, rows(), num_results)
            if len(result) < num_results:
                return self._rank(
                    query,
                    self.connection.execute("SELECT id, vector FROM vectors"),
                    num_results,
                )
            return result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self.open()
            return {
                "backend": "diskann-disk",
                "mode": "diskann+delta" if self._index is not None else "exact",
                "count": self.count(),
                "snapshot_count": self.connection.execute(
                    "SELECT COUNT(*) FROM snapshot"
                ).fetchone()[0],
                "delta_count": self.connection.execute(
                    "SELECT COUNT(*) FROM delta"
                ).fetchone()[0],
                "generation": self._get_state("generation"),
                "last_error": self._last_error,
            }

    def close(self) -> None:
        with self._lock:
            self._index = None
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._file_lock.release()
