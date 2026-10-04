from pathlib import Path
from typing import Any, Dict, Optional, Union

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.diskann import DiskAnnAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from chromadb.segment.impl.vector import diskann_store


def PersistentClient(
    path: Union[str, Path] = "./chroma_diskann",
    settings: Optional[Settings] = None,
    tenant: str = DEFAULT_TENANT,
    database: str = DEFAULT_DATABASE,
) -> ClientAPI:
    """Create an experimental single-process Chroma client backed by DiskANN3."""
    diskann_store.require_native()
    configured = settings.model_copy(deep=True) if settings is not None else Settings()
    configured.chroma_api_impl = "chromadb.api.diskann.DiskAnnAPI"
    configured.chroma_segment_manager_impl = (
        "chromadb.segment.impl.manager.diskann.DiskAnnSegmentManager"
    )
    return chromadb.PersistentClient(
        path=str(Path(path).resolve()),
        settings=configured,
        tenant=tenant,
        database=database,
    )


def _api(collection: Collection) -> DiskAnnAPI:
    api = collection._client
    if not isinstance(api, DiskAnnAPI):
        raise ValueError("Use a collection from the local DiskANN PersistentClient")
    return api


def rebuild(collection: Collection) -> bool:
    """Synchronously publish a new snapshot; return False below the training minimum."""
    return _api(collection).diskann_segment(collection.id)._store.rebuild()


def index_status(collection: Collection) -> Dict[str, Any]:
    """Report disk-index, exact-fallback, and durable-delta state."""
    return _api(collection).diskann_segment(collection.id)._store.status()
