from typing import Sequence
from uuid import UUID

from overrides import override

from chromadb.config import System
from chromadb.segment import SegmentType, VectorReader
from chromadb.segment.impl.manager.local import LocalSegmentManager


class DiskAnnSegmentManager(LocalSegmentManager):
    def __init__(self, system: System):
        if not system.settings.is_persistent:
            raise ValueError("DiskANN requires persistence")
        super().__init__(system)
        self._vector_segment_type = SegmentType.DISKANN_LOCAL_PERSISTED

    @override
    def delete_segments(self, collection_id: UUID) -> Sequence[UUID]:
        for segment in self._sysdb.get_segments(collection=collection_id):
            if (
                segment["type"] == SegmentType.DISKANN_LOCAL_PERSISTED.value
                and segment["id"] not in self._instances
            ):
                self.get_segment(collection_id, VectorReader)
        return super().delete_segments(collection_id)
