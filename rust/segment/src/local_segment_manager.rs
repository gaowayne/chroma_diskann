use chroma_cache::{Cache, CacheConfig, CacheError, FoyerCacheConfig};
use chroma_config::{
    registry::{Injectable, Registry},
    Configurable,
};
use chroma_error::{ChromaError, ErrorCodes};
use chroma_index::IndexUuid;
use chroma_sqlite::db::SqliteDb;
use chroma_types::{Collection, Segment};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;
use tokio::sync::Mutex;

use crate::local_diskann::{
    LocalDiskAnnIndex, LocalDiskAnnSegmentReader, LocalDiskAnnSegmentReaderError,
    LocalDiskAnnSegmentWriter, LocalDiskAnnSegmentWriterError,
};
use crate::local_hnsw::{
    LocalHnswIndex, LocalHnswSegmentReader, LocalHnswSegmentReaderError, LocalHnswSegmentWriter,
    LocalHnswSegmentWriterError,
};

fn default_hnsw_index_pool_cache_config() -> CacheConfig {
    CacheConfig::Memory(FoyerCacheConfig {
        capacity: 65536,
        ..Default::default()
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocalSegmentManagerConfig {
    // TODO(Sanket): Estimate the max number of FDs that can be kept open and
    // use that as a capacity in the cache.
    #[serde(default = "default_hnsw_index_pool_cache_config")]
    pub hnsw_index_pool_cache_config: CacheConfig,
    pub persist_path: Option<String>,
}

#[derive(Clone, Debug)]
pub struct LocalSegmentManager {
    hnsw_index_pool: Arc<dyn Cache<IndexUuid, LocalHnswIndex>>,
    diskann_index_pool: Arc<Mutex<HashMap<IndexUuid, LocalDiskAnnIndex>>>,
    #[allow(dead_code)]
    eviction_callback_task_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    sqlite: SqliteDb,
    persist_root: Option<String>,
}

impl Injectable for LocalSegmentManager {}

#[async_trait::async_trait]
impl Configurable<LocalSegmentManagerConfig> for LocalSegmentManager {
    async fn try_from_config(
        config: &LocalSegmentManagerConfig,
        registry: &Registry,
    ) -> Result<Self, Box<dyn chroma_error::ChromaError>> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let hnsw_index_pool: Box<dyn Cache<IndexUuid, LocalHnswIndex>> =
            chroma_cache::from_config_with_event_listener(&config.hnsw_index_pool_cache_config, tx)
                .await?;
        let sqldb = registry.get::<SqliteDb>().map_err(|e| e.boxed())?;
        // TODO(Sanket): Might need tokio runtime to be passed here to spawn the task.
        let handle = tokio::spawn(async move {
            while let Some((_, index)) = rx.recv().await {
                // Close the FD here.
                index.close().await;
            }
        });
        let res = Self {
            hnsw_index_pool: hnsw_index_pool.into(),
            diskann_index_pool: Arc::new(Mutex::new(HashMap::new())),
            eviction_callback_task_handle: Some(Arc::new(handle)),
            sqlite: sqldb,
            persist_root: config.persist_path.clone(),
        };
        registry.register(res.clone());
        Ok(res)
    }
}

#[derive(Error, Debug)]
pub enum LocalSegmentManagerError {
    #[error("Error creating hnsw segment reader: {0}")]
    LocalHnswSegmentReaderError(#[from] LocalHnswSegmentReaderError),
    #[error("Error reading hnsw pool cache: {0}")]
    PoolCacheError(#[from] CacheError),
    #[error("Error creating hnsw segment writer: {0}")]
    LocalHnswSegmentWriterError(#[from] LocalHnswSegmentWriterError),
    #[error("Error creating DiskANN segment reader: {0}")]
    LocalDiskAnnSegmentReaderError(#[from] LocalDiskAnnSegmentReaderError),
    #[error("Error creating DiskANN segment writer: {0}")]
    LocalDiskAnnSegmentWriterError(#[from] LocalDiskAnnSegmentWriterError),
}

impl ChromaError for LocalSegmentManagerError {
    fn code(&self) -> ErrorCodes {
        match self {
            LocalSegmentManagerError::LocalHnswSegmentReaderError(e) => e.code(),
            LocalSegmentManagerError::PoolCacheError(e) => e.code(),
            LocalSegmentManagerError::LocalHnswSegmentWriterError(e) => e.code(),
            LocalSegmentManagerError::LocalDiskAnnSegmentReaderError(e) => e.code(),
            LocalSegmentManagerError::LocalDiskAnnSegmentWriterError(e) => e.code(),
        }
    }
}

impl LocalSegmentManager {
    pub async fn get_hnsw_reader(
        &self,
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
    ) -> Result<LocalHnswSegmentReader, LocalSegmentManagerError> {
        let index_uuid = IndexUuid(segment.id.0);
        match self.hnsw_index_pool.get(&IndexUuid(segment.id.0)).await? {
            Some(hnsw_index) => Ok(LocalHnswSegmentReader::from_index(hnsw_index)),
            None => {
                let reader = LocalHnswSegmentReader::from_segment(
                    collection,
                    segment,
                    dimensionality,
                    self.persist_root.clone(),
                    self.sqlite.clone(),
                )
                .await?;
                // Open the FDs.
                reader.index.start().await;
                self.hnsw_index_pool
                    .insert(index_uuid, reader.index.clone())
                    .await;
                Ok(reader)
            }
        }
    }

    pub async fn get_hnsw_writer(
        &self,
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
    ) -> Result<LocalHnswSegmentWriter, LocalSegmentManagerError> {
        let index_uuid = IndexUuid(segment.id.0);
        match self.hnsw_index_pool.get(&IndexUuid(segment.id.0)).await? {
            Some(hnsw_index) => Ok(LocalHnswSegmentWriter::from_index(hnsw_index)?),
            None => {
                let writer = LocalHnswSegmentWriter::from_segment(
                    collection,
                    segment,
                    dimensionality,
                    self.persist_root.clone(),
                    self.sqlite.clone(),
                )
                .await?;
                // Open the FDs.
                writer.index.start().await;
                // Backfill.
                self.hnsw_index_pool
                    .insert(index_uuid, writer.index.clone())
                    .await;
                Ok(writer)
            }
        }
    }

    pub async fn get_diskann_reader(
        &self,
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
    ) -> Result<LocalDiskAnnSegmentReader, LocalSegmentManagerError> {
        let index_uuid = IndexUuid(segment.id.0);
        let mut pool = self.diskann_index_pool.lock().await;
        if let Some(index) = pool.get(&index_uuid) {
            return Ok(LocalDiskAnnSegmentReader::from_index(index.clone()));
        }
        let reader = LocalDiskAnnSegmentReader::from_segment(
            collection,
            segment,
            dimensionality,
            self.persist_root.clone(),
            self.sqlite.clone(),
        )
        .await?;
        pool.insert(index_uuid, reader.index.clone());
        Ok(reader)
    }

    pub async fn get_diskann_writer(
        &self,
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
    ) -> Result<LocalDiskAnnSegmentWriter, LocalSegmentManagerError> {
        let index_uuid = IndexUuid(segment.id.0);
        let mut pool = self.diskann_index_pool.lock().await;
        if let Some(index) = pool.get(&index_uuid) {
            return Ok(LocalDiskAnnSegmentWriter::from_index(index.clone())?);
        }
        let writer = LocalDiskAnnSegmentWriter::from_segment(
            collection,
            segment,
            dimensionality,
            self.persist_root.clone(),
            self.sqlite.clone(),
        )
        .await?;
        pool.insert(index_uuid, writer.index.clone());
        Ok(writer)
    }

    pub async fn reset(&self) -> Result<(), LocalSegmentManagerError> {
        self.hnsw_index_pool.clear().await?;
        self.diskann_index_pool.lock().await.clear();
        Ok(())
    }
}
