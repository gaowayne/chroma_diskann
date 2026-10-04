from copy import deepcopy
from typing import List, Optional
from uuid import UUID

import numpy as np
from overrides import override

from chromadb.api.collection_configuration import (
    CreateCollectionConfiguration,
    UpdateCollectionConfiguration,
)
from chromadb.api.segment import SegmentAPI
from chromadb.api.types import CollectionMetadata, Schema
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, System
from chromadb.segment import VectorReader
from chromadb.segment.impl.vector.diskann_params import DiskAnnParams
from chromadb.segment.impl.vector.local_diskann import LocalDiskAnnSegment
from chromadb.types import Collection, OperationRecord


class DiskAnnAPI(SegmentAPI):
    def __init__(self, system: System):
        if (
            not system.settings.is_persistent
            or system.settings.chroma_segment_manager_impl
            != "chromadb.segment.impl.manager.diskann.DiskAnnSegmentManager"
        ):
            raise ValueError(
                "Use chromadb.experimental.diskann.PersistentClient to enable DiskANN"
            )
        super().__init__(system)

    @override
    def create_collection(
        self,
        name: str,
        schema: Optional[Schema] = None,
        configuration: Optional[CreateCollectionConfiguration] = None,
        metadata: Optional[CollectionMetadata] = None,
        get_or_create: bool = False,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
    ) -> Collection:
        settings = deepcopy(configuration or {})
        values = dict(metadata or {})
        if schema is not None or settings.get("spann") is not None:
            raise ValueError(
                "DiskANN currently supports local dense-vector collections without a custom schema"
            )
        hnsw = settings.get("hnsw") or {}
        if set(hnsw) - {"space"} or any(key.startswith("hnsw:") for key in values):
            raise ValueError(
                "Configure DiskANN with diskann: parameters, not HNSW tuning parameters"
            )
        if (
            "space" in hnsw
            and "diskann:space" in values
            and hnsw["space"] != values["diskann:space"]
        ):
            raise ValueError("Conflicting vector distance configurations")
        values.setdefault("diskann:space", hnsw.get("space", "l2"))
        params = DiskAnnParams(values)
        if values.get("chroma:vector_index", "diskann") != "diskann":
            raise ValueError("This client creates DiskANN collections only")
        values["chroma:vector_index"] = "diskann"
        settings["hnsw"] = {"space": params.space}
        return super().create_collection(
            name=name,
            configuration=settings,
            metadata=values,
            get_or_create=get_or_create,
            tenant=tenant,
            database=database,
        )

    @override
    def _modify(
        self,
        id: UUID,
        new_name: Optional[str] = None,
        new_metadata: Optional[CollectionMetadata] = None,
        new_configuration: Optional[UpdateCollectionConfiguration] = None,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
    ) -> None:
        collection = self._get_collection(id)
        original = collection.metadata or {}
        if original.get("chroma:vector_index") == "diskann":
            if new_configuration is not None:
                raise ValueError(
                    "DiskANN index configuration is immutable; create a new collection"
                )
            if new_metadata is not None:
                if any(key.startswith("hnsw:") for key in new_metadata):
                    raise ValueError(
                        "Configure DiskANN with diskann: parameters, not HNSW tuning parameters"
                    )
                reserved = {
                    key: value
                    for key, value in original.items()
                    if key.startswith("diskann:") or key == "chroma:vector_index"
                }
                merged = {**reserved, **new_metadata}
                if merged.get("chroma:vector_index") != "diskann" or vars(
                    DiskAnnParams(merged)
                ) != vars(DiskAnnParams(original)):
                    raise ValueError("DiskANN index configuration is immutable")
                new_metadata = merged
        return super()._modify(
            id, new_name, new_metadata, new_configuration, tenant, database
        )

    @override
    def _validate_embedding_record_set(
        self, collection: Collection, records: List[OperationRecord]
    ) -> None:
        metadata = collection.metadata or {}
        if metadata.get("chroma:vector_index") == "diskann":
            params = DiskAnnParams(metadata)
            for record in records:
                if record["embedding"] is None:
                    continue
                vector = np.asarray(record["embedding"], dtype=np.float32)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                    raise ValueError("DiskANN vectors must be nonempty and finite")
                if (
                    params.space == "cosine"
                    and np.linalg.norm(vector.astype(np.float64)) == 0
                ):
                    raise ValueError("DiskANN cosine vectors must have nonzero norm")
        super()._validate_embedding_record_set(collection, records)

    def diskann_segment(self, collection_id: UUID) -> LocalDiskAnnSegment:
        self._get_collection(collection_id)
        segment = self._manager.get_segment(collection_id, VectorReader)
        if not isinstance(segment, LocalDiskAnnSegment):
            raise ValueError("Collection does not use DiskANN")
        return segment
