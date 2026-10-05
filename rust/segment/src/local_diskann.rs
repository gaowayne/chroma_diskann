use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use chroma_distance::{normalize, DistanceFunction};
use chroma_error::{ChromaError, ErrorCodes};
use chroma_sqlite::{db::SqliteDb, table::MaxSeqId};
use chroma_types::{
    operator::RecordMeasure, Chunk, Collection, InternalDiskAnnConfiguration, LogRecord, Operation,
    Segment, SegmentUuid, Space,
};
use sea_query::{Expr, Query, SqliteQueryBuilder};
use sea_query_binder::SqlxBinder;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use thiserror::Error;
use tokio::sync::RwLock;

const METADATA_FILE: &str = "diskann_metadata.json";
const NATIVE_INDEX_DIR: &str = "native";
const MIN_NATIVE_POINTS: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IdMap {
    dimensionality: usize,
    total_elements_added: u32,
    id_to_label: HashMap<String, u32>,
    label_to_id: HashMap<u32, String>,
    embeddings: HashMap<u32, Vec<f32>>,
    /// Labels in the row order used to build the last native DiskANN graph.
    #[serde(default)]
    native_labels: Vec<u32>,
}

impl IdMap {
    fn new(dimensionality: usize) -> Self {
        Self {
            dimensionality,
            total_elements_added: 0,
            id_to_label: HashMap::new(),
            label_to_id: HashMap::new(),
            embeddings: HashMap::new(),
            native_labels: Vec::new(),
        }
    }
}

struct Inner {
    id_map: IdMap,
    persist_path: Option<String>,
    sqlite: SqliteDb,
    last_seen_seq_id: u64,
    num_elements_since_last_persist: u64,
    config: InternalDiskAnnConfiguration,
    native_index: Option<Arc<chroma_diskann::DiskAnnIndex>>,
    dirty: bool,
}

#[derive(Clone)]
pub struct LocalDiskAnnIndex {
    inner: Arc<RwLock<Inner>>,
}

impl std::fmt::Debug for LocalDiskAnnIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalDiskAnnIndex").finish_non_exhaustive()
    }
}

pub struct LocalDiskAnnSegmentReader {
    pub index: LocalDiskAnnIndex,
}

#[derive(Error, Debug)]
pub enum LocalDiskAnnSegmentReaderError {
    #[error("Nothing found on disk")]
    UninitializedSegment,
    #[error("Collection is missing DiskANN configuration")]
    MissingDiskAnnConfiguration,
    #[error("Error serializing path to string")]
    PersistPathError,
    #[error("Error finding id")]
    IdNotFound,
    #[error("Error getting embedding")]
    GetEmbeddingError,
    #[error("Error querying knn")]
    QueryError,
    #[error("Persisted DiskANN dimensionality {actual} does not match collection dimensionality {expected}")]
    DimensionalityMismatch { expected: usize, actual: usize },
    #[error("Error reading from sqlite: {0}")]
    SqliteError(#[from] sqlx::error::Error),
    #[error("Error building max sequence id query")]
    QueryBuilderError(#[from] sea_query::error::Error),
    #[error("Error reading DiskANN metadata: {0}")]
    MetadataIo(#[from] std::io::Error),
    #[error("Error parsing DiskANN metadata: {0}")]
    MetadataParse(#[from] serde_json::Error),
}

impl ChromaError for LocalDiskAnnSegmentReaderError {
    fn code(&self) -> ErrorCodes {
        match self {
            LocalDiskAnnSegmentReaderError::UninitializedSegment => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::MissingDiskAnnConfiguration => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::PersistPathError => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::IdNotFound => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::GetEmbeddingError => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::QueryError => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::DimensionalityMismatch { .. } => ErrorCodes::DataLoss,
            LocalDiskAnnSegmentReaderError::SqliteError(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::QueryBuilderError(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::MetadataIo(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentReaderError::MetadataParse(_) => ErrorCodes::Internal,
        }
    }
}

pub struct LocalDiskAnnSegmentWriter {
    pub index: LocalDiskAnnIndex,
}

#[derive(Error, Debug)]
pub enum LocalDiskAnnSegmentWriterError {
    #[error("Nothing found on disk")]
    UninitializedSegment,
    #[error("Collection is missing DiskANN configuration")]
    MissingDiskAnnConfiguration,
    #[error("Error serializing path to string")]
    PersistPathError,
    #[error("Embedding not found")]
    EmbeddingNotFound,
    #[error("Persisted DiskANN dimensionality {actual} does not match collection dimensionality {expected}")]
    DimensionalityMismatch { expected: usize, actual: usize },
    #[error("Error reading from sqlite: {0}")]
    SqliteError(#[from] sqlx::error::Error),
    #[error("Error building max sequence id query")]
    QueryBuilderError(#[from] sea_query::error::Error),
    #[error("Error writing DiskANN metadata: {0}")]
    MetadataIo(#[from] std::io::Error),
    #[error("Error serializing DiskANN metadata: {0}")]
    MetadataParse(#[from] serde_json::Error),
    #[error("Native DiskANN build failed: {0}")]
    NativeBuild(String),
}

impl ChromaError for LocalDiskAnnSegmentWriterError {
    fn code(&self) -> ErrorCodes {
        match self {
            LocalDiskAnnSegmentWriterError::UninitializedSegment => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::MissingDiskAnnConfiguration => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::PersistPathError => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::EmbeddingNotFound => ErrorCodes::InvalidArgument,
            LocalDiskAnnSegmentWriterError::DimensionalityMismatch { .. } => ErrorCodes::DataLoss,
            LocalDiskAnnSegmentWriterError::SqliteError(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::QueryBuilderError(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::MetadataIo(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::MetadataParse(_) => ErrorCodes::Internal,
            LocalDiskAnnSegmentWriterError::NativeBuild(_) => ErrorCodes::Internal,
        }
    }
}

fn diskann_config(
    collection: &Collection,
) -> Result<InternalDiskAnnConfiguration, LocalDiskAnnSegmentReaderError> {
    collection
        .schema
        .as_ref()
        .and_then(|schema| schema.get_internal_diskann_config())
        .or_else(|| collection.config.get_diskann_config())
        .ok_or(LocalDiskAnnSegmentReaderError::MissingDiskAnnConfiguration)
}

fn validate_embedding_dim(embedding: &[f32], expected: usize) -> Result<(), LocalDiskAnnSegmentWriterError> {
    if embedding.len() != expected {
        return Err(LocalDiskAnnSegmentWriterError::DimensionalityMismatch {
            expected,
            actual: embedding.len(),
        });
    }
    Ok(())
}

fn space_query_vector(space: &Space, embedding: &[f32]) -> Vec<f32> {
    if matches!(space, Space::Cosine) {
        normalize(embedding)
    } else {
        embedding.to_vec()
    }
}

fn stored_vector(space: &Space, embedding: &[f32]) -> Vec<f32> {
    space_query_vector(space, embedding)
}

fn native_dir(persist_path: &str) -> PathBuf {
    PathBuf::from(persist_path).join(NATIVE_INDEX_DIR)
}

fn native_is_ready(id_map: &IdMap) -> bool {
    let n = id_map.embeddings.len();
    n >= MIN_NATIVE_POINTS
        && id_map.native_labels.len() == n
        && id_map.native_labels.iter().all(|label| id_map.embeddings.contains_key(label))
}

async fn try_open_native(
    persist_path: &str,
    id_map: &IdMap,
) -> Option<Arc<chroma_diskann::DiskAnnIndex>> {
    if !native_is_ready(id_map) {
        return None;
    }
    let native_dir = native_dir(persist_path);
    if !native_dir.exists() {
        return None;
    }
    match tokio::task::spawn_blocking(move || chroma_diskann::DiskAnnIndex::open(native_dir)).await
    {
        Ok(Ok(index)) if index.len() == id_map.embeddings.len() => Some(Arc::new(index)),
        Ok(Ok(_)) => {
            tracing::warn!("native DiskANN graph size does not match stored vectors");
            None
        }
        Ok(Err(err)) => {
            tracing::warn!("failed to open native DiskANN index: {err}");
            None
        }
        Err(err) => {
            tracing::warn!("failed to join native DiskANN open: {err}");
            None
        }
    }
}

fn brute_force_query(
    id_map: &IdMap,
    space: &Space,
    embedding: &[f32],
    allowed: Option<&HashSet<u32>>,
    k: u32,
) -> Vec<RecordMeasure> {
    let query = space_query_vector(space, embedding);
    let df = DistanceFunction::from(space.clone());
    let mut scored: Vec<RecordMeasure> = Vec::new();
    for (label, vector) in &id_map.embeddings {
        if let Some(allowed) = allowed {
            if !allowed.contains(label) {
                continue;
            }
        }
        scored.push(RecordMeasure {
            offset_id: *label,
            measure: df.distance(&query, vector),
        });
    }
    scored.sort_by(|a, b| {
        a.measure
            .partial_cmp(&b.measure)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k as usize);
    scored
}

async fn load_inner(
    persist_path: String,
    id_map: IdMap,
    sql_db: SqliteDb,
    config: InternalDiskAnnConfiguration,
) -> Inner {
    let native_index = try_open_native(&persist_path, &id_map).await;
    let dirty = id_map.embeddings.len() >= MIN_NATIVE_POINTS && native_index.is_none();
    Inner {
        id_map,
        persist_path: Some(persist_path),
        sqlite: sql_db,
        last_seen_seq_id: 0,
        num_elements_since_last_persist: 0,
        config,
        native_index,
        dirty,
    }
}

impl LocalDiskAnnSegmentReader {
    pub fn from_index(index: LocalDiskAnnIndex) -> Self {
        Self { index }
    }

    pub async fn from_segment(
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
        persist_root: Option<String>,
        sql_db: SqliteDb,
    ) -> Result<Self, LocalDiskAnnSegmentReaderError> {
        let config = diskann_config(collection)?;
        let Some(path_str) = persist_root else {
            return Err(LocalDiskAnnSegmentReaderError::UninitializedSegment);
        };
        let index_folder = Path::new(&path_str).join(segment.id.to_string());
        if !index_folder.exists() {
            return Err(LocalDiskAnnSegmentReaderError::UninitializedSegment);
        }
        let metadata_path = index_folder.join(METADATA_FILE);
        if !metadata_path.exists() {
            return Err(LocalDiskAnnSegmentReaderError::UninitializedSegment);
        }
        let bytes = tokio::fs::read(&metadata_path).await?;
        let id_map: IdMap = serde_json::from_slice(&bytes)?;
        if id_map.dimensionality != dimensionality {
            return Err(LocalDiskAnnSegmentReaderError::DimensionalityMismatch {
                expected: dimensionality,
                actual: id_map.dimensionality,
            });
        }
        let persist_path = index_folder
            .to_str()
            .ok_or(LocalDiskAnnSegmentReaderError::PersistPathError)?
            .to_string();
        Ok(Self {
            index: LocalDiskAnnIndex {
                inner: Arc::new(RwLock::new(
                    load_inner(persist_path, id_map, sql_db, config).await,
                )),
            },
        })
    }

    pub async fn current_max_seq_id(
        &self,
        segment_id: &SegmentUuid,
    ) -> Result<u64, LocalDiskAnnSegmentReaderError> {
        let guard = self.index.inner.read().await;
        let (sql, values) = Query::select()
            .column(MaxSeqId::SeqId)
            .from(MaxSeqId::Table)
            .and_where(Expr::col(MaxSeqId::SegmentId).eq(segment_id.to_string()))
            .build_sqlx(SqliteQueryBuilder);
        let row_opt = sqlx::query_with(&sql, values)
            .fetch_optional(guard.sqlite.get_conn())
            .await?;
        Ok(row_opt
            .map(|row| row.try_get::<u64, _>(0))
            .transpose()?
            .unwrap_or_default())
    }

    pub async fn get_embedding_by_user_id(
        &self,
        user_id: &String,
    ) -> Result<Vec<f32>, LocalDiskAnnSegmentReaderError> {
        let offset_id = self.get_offset_id_by_user_id(user_id).await?;
        self.get_embedding_by_offset_id(offset_id).await
    }

    pub async fn get_offset_id_by_user_id(
        &self,
        user_id: &String,
    ) -> Result<u32, LocalDiskAnnSegmentReaderError> {
        let guard = self.index.inner.read().await;
        guard
            .id_map
            .id_to_label
            .get(user_id)
            .cloned()
            .ok_or(LocalDiskAnnSegmentReaderError::IdNotFound)
    }

    pub async fn get_user_id_by_offset_id(
        &self,
        offset_id: u32,
    ) -> Result<String, LocalDiskAnnSegmentReaderError> {
        let guard = self.index.inner.read().await;
        guard
            .id_map
            .label_to_id
            .get(&offset_id)
            .cloned()
            .ok_or(LocalDiskAnnSegmentReaderError::IdNotFound)
    }

    pub async fn get_embedding_by_offset_id(
        &self,
        offset_id: u32,
    ) -> Result<Vec<f32>, LocalDiskAnnSegmentReaderError> {
        let guard = self.index.inner.read().await;
        guard
            .id_map
            .embeddings
            .get(&offset_id)
            .cloned()
            .ok_or(LocalDiskAnnSegmentReaderError::GetEmbeddingError)
    }

    pub async fn query_embedding(
        &self,
        allowed_offset_ids: &[u32],
        embedding: Vec<f32>,
        k: u32,
    ) -> Result<Vec<RecordMeasure>, LocalDiskAnnSegmentReaderError> {
        let guard = self.index.inner.read().await;
        if embedding.len() != guard.id_map.dimensionality {
            return Err(LocalDiskAnnSegmentReaderError::QueryError);
        }
        if guard.id_map.embeddings.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let allowed: Option<HashSet<u32>> = if allowed_offset_ids.is_empty() {
            None
        } else {
            Some(allowed_offset_ids.iter().copied().collect())
        };

        let can_use_native = allowed.is_none()
            && !guard.dirty
            && guard.native_index.is_some()
            && native_is_ready(&guard.id_map);

        if can_use_native {
            let index = guard
                .native_index
                .clone()
                .expect("native index checked above");
            let labels = guard.id_map.native_labels.clone();
            let options = chroma_diskann::SearchOptions {
                search_list_size: guard.config.search_list_size.max(k).max(1),
                beam_width: guard.config.beam_width.clamp(1, 128),
            };
            drop(guard);
            let query = embedding.clone();
            match tokio::task::spawn_blocking(move || index.search(&query, k as usize, &options))
                .await
            {
                Ok(Ok(result)) => {
                    let mut scored = Vec::with_capacity(result.neighbors.len());
                    for neighbor in result.neighbors {
                        let Some(label) = labels.get(neighbor.offset_id as usize).copied() else {
                            tracing::warn!(
                                "native DiskANN returned out-of-range row {}",
                                neighbor.offset_id
                            );
                            continue;
                        };
                        scored.push(RecordMeasure {
                            offset_id: label,
                            measure: neighbor.distance,
                        });
                    }
                    return Ok(scored);
                }
                Ok(Err(err)) => {
                    tracing::warn!("native DiskANN search failed, using exact scan: {err}");
                }
                Err(err) => {
                    tracing::warn!("native DiskANN search join failed, using exact scan: {err}");
                }
            }
            let guard = self.index.inner.read().await;
            return Ok(brute_force_query(
                &guard.id_map,
                &guard.config.space,
                &embedding,
                None,
                k,
            ));
        }

        Ok(brute_force_query(
            &guard.id_map,
            &guard.config.space,
            &embedding,
            allowed.as_ref(),
            k,
        ))
    }
}

impl LocalDiskAnnSegmentWriter {
    pub fn from_index(index: LocalDiskAnnIndex) -> Result<Self, LocalDiskAnnSegmentWriterError> {
        Ok(Self { index })
    }

    pub async fn from_segment(
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
        persist_root: Option<String>,
        sql_db: SqliteDb,
    ) -> Result<Self, LocalDiskAnnSegmentWriterError> {
        let config = collection
            .schema
            .as_ref()
            .and_then(|schema| schema.get_internal_diskann_config())
            .or_else(|| collection.config.get_diskann_config())
            .ok_or(LocalDiskAnnSegmentWriterError::MissingDiskAnnConfiguration)?;
        let Some(path_str) = persist_root else {
            return Err(LocalDiskAnnSegmentWriterError::PersistPathError);
        };
        let index_folder = Path::new(&path_str).join(segment.id.to_string());
        tokio::fs::create_dir_all(&index_folder).await?;
        let persist_path = index_folder
            .to_str()
            .ok_or(LocalDiskAnnSegmentWriterError::PersistPathError)?
            .to_string();
        let metadata_path = index_folder.join(METADATA_FILE);
        let inner = if metadata_path.exists() {
            let bytes = tokio::fs::read(&metadata_path).await?;
            let id_map: IdMap = serde_json::from_slice(&bytes)?;
            if id_map.dimensionality != dimensionality {
                return Err(LocalDiskAnnSegmentWriterError::DimensionalityMismatch {
                    expected: dimensionality,
                    actual: id_map.dimensionality,
                });
            }
            load_inner(persist_path, id_map, sql_db, config).await
        } else {
            Inner {
                id_map: IdMap::new(dimensionality),
                persist_path: Some(persist_path),
                sqlite: sql_db,
                last_seen_seq_id: 0,
                num_elements_since_last_persist: 0,
                config,
                native_index: None,
                dirty: false,
            }
        };
        Ok(Self {
            index: LocalDiskAnnIndex {
                inner: Arc::new(RwLock::new(inner)),
            },
        })
    }

    pub async fn apply_log_chunk(
        &mut self,
        log_chunk: Chunk<LogRecord>,
    ) -> Result<u32, LocalDiskAnnSegmentWriterError> {
        let mut guard = self.index.inner.write().await;
        let mut next_label = guard.id_map.total_elements_added + 1;
        if log_chunk.is_empty() {
            return Ok(next_label);
        }
        let expected_dim = guard.id_map.dimensionality;
        let space = guard.config.space.clone();
        let mut max_seq_id = u64::MIN;
        for (log, _) in log_chunk.iter() {
            if log.log_offset <= guard.last_seen_seq_id as i64 {
                continue;
            }
            guard.num_elements_since_last_persist += 1;
            max_seq_id = max_seq_id.max(log.log_offset as u64);
            match log.record.operation {
                Operation::BackfillFn => continue,
                Operation::Add => {
                    if guard.id_map.id_to_label.contains_key(&log.record.id) {
                        continue;
                    }
                    let embedding = log
                        .record
                        .embedding
                        .as_ref()
                        .ok_or(LocalDiskAnnSegmentWriterError::EmbeddingNotFound)?;
                    validate_embedding_dim(embedding, expected_dim)?;
                    guard
                        .id_map
                        .id_to_label
                        .insert(log.record.id.clone(), next_label);
                    guard
                        .id_map
                        .label_to_id
                        .insert(next_label, log.record.id.clone());
                    guard
                        .id_map
                        .embeddings
                        .insert(next_label, stored_vector(&space, embedding));
                    next_label += 1;
                    guard.dirty = true;
                    guard.native_index = None;
                }
                Operation::Update => {
                    if let Some(label) = guard.id_map.id_to_label.get(&log.record.id).cloned() {
                        if let Some(embedding) = &log.record.embedding {
                            validate_embedding_dim(embedding, expected_dim)?;
                            guard
                                .id_map
                                .embeddings
                                .insert(label, stored_vector(&space, embedding));
                            guard.dirty = true;
                            guard.native_index = None;
                        }
                    }
                }
                Operation::Upsert => {
                    let embedding = log
                        .record
                        .embedding
                        .as_ref()
                        .ok_or(LocalDiskAnnSegmentWriterError::EmbeddingNotFound)?;
                    validate_embedding_dim(embedding, expected_dim)?;
                    if let Some(label) = guard.id_map.id_to_label.get(&log.record.id).cloned() {
                        guard
                            .id_map
                            .embeddings
                            .insert(label, stored_vector(&space, embedding));
                    } else {
                        guard
                            .id_map
                            .id_to_label
                            .insert(log.record.id.clone(), next_label);
                        guard
                            .id_map
                            .label_to_id
                            .insert(next_label, log.record.id.clone());
                        guard
                            .id_map
                            .embeddings
                            .insert(next_label, stored_vector(&space, embedding));
                        next_label += 1;
                    }
                    guard.dirty = true;
                    guard.native_index = None;
                }
                Operation::Delete => {
                    if let Some(label) = guard.id_map.id_to_label.remove(&log.record.id) {
                        guard.id_map.label_to_id.remove(&label);
                        guard.id_map.embeddings.remove(&label);
                        guard.dirty = true;
                        guard.native_index = None;
                    }
                }
            }
        }
        guard.id_map.total_elements_added = next_label.saturating_sub(1);
        if max_seq_id != u64::MIN {
            guard.last_seen_seq_id = max_seq_id;
        }
        if guard.num_elements_since_last_persist > 0 {
            persist(&mut guard).await?;
            if max_seq_id != u64::MIN {
                let id = PathBuf::from(guard.persist_path.as_ref().unwrap())
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string()
                    .into();
                let max_id = max_seq_id.into();
                let (query, values) = Query::insert()
                    .into_table(MaxSeqId::Table)
                    .replace()
                    .columns([MaxSeqId::SegmentId, MaxSeqId::SeqId])
                    .values([id, max_id])?
                    .build_sqlx(SqliteQueryBuilder);
                sqlx::query_with(&query, values)
                    .execute(guard.sqlite.get_conn())
                    .await?;
            }
            guard.num_elements_since_last_persist = 0;
        }
        Ok(next_label)
    }
}

async fn persist(guard: &mut Inner) -> Result<(), LocalDiskAnnSegmentWriterError> {
    let Some(path) = guard.persist_path.clone() else {
        return Ok(());
    };
    let folder = PathBuf::from(&path);
    tokio::fs::create_dir_all(&folder).await?;
    let bytes = serde_json::to_vec(&guard.id_map)?;
    tokio::fs::write(folder.join(METADATA_FILE), bytes).await?;

    if guard.id_map.embeddings.len() >= MIN_NATIVE_POINTS && guard.dirty {
        rebuild_native(guard).await?;
        guard.dirty = false;
    } else if guard.id_map.embeddings.len() < MIN_NATIVE_POINTS {
        guard.native_index = None;
        guard.id_map.native_labels.clear();
        guard.dirty = false;
        let native_dir = folder.join(NATIVE_INDEX_DIR);
        if native_dir.exists() {
            tokio::fs::remove_dir_all(&native_dir).await?;
        }
        let bytes = serde_json::to_vec(&guard.id_map)?;
        tokio::fs::write(folder.join(METADATA_FILE), bytes).await?;
    }
    Ok(())
}

async fn rebuild_native(guard: &mut Inner) -> Result<(), LocalDiskAnnSegmentWriterError> {
    let Some(path) = guard.persist_path.clone() else {
        return Ok(());
    };
    let mut labels: Vec<u32> = guard.id_map.embeddings.keys().copied().collect();
    labels.sort_unstable();
    let vectors: Vec<Vec<f32>> = labels
        .iter()
        .map(|label| guard.id_map.embeddings[label].clone())
        .collect();
    let config = guard.config.clone();
    let native_dir = PathBuf::from(&path).join(NATIVE_INDEX_DIR);
    if native_dir.exists() {
        tokio::fs::remove_dir_all(&native_dir).await?;
    }
    let metric = match config.space {
        Space::Cosine => chroma_diskann::DistanceMetric::Cosine,
        _ => chroma_diskann::DistanceMetric::L2,
    };
    let options = chroma_diskann::BuildOptions {
        metric,
        graph_degree: config.graph_degree,
        search_list_size: config.build_list_size,
        pq_bytes: config.pq_bytes,
        num_threads: config.num_threads.max(1),
        memory_budget_gb: config.memory_budget_gb,
        alpha: config.alpha,
        seed: 42,
    };
    let build_dir = native_dir.clone();
    tokio::task::spawn_blocking(move || chroma_diskann::build_index(&build_dir, &vectors, &options))
        .await
        .map_err(|e| LocalDiskAnnSegmentWriterError::NativeBuild(e.to_string()))?
        .map_err(|e| LocalDiskAnnSegmentWriterError::NativeBuild(e.to_string()))?;
    let opened = tokio::task::spawn_blocking({
        let native_dir = native_dir.clone();
        move || chroma_diskann::DiskAnnIndex::open(native_dir)
    })
    .await
    .map_err(|e| LocalDiskAnnSegmentWriterError::NativeBuild(e.to_string()))?
    .map_err(|e| LocalDiskAnnSegmentWriterError::NativeBuild(e.to_string()))?;
    guard.id_map.native_labels = labels;
    guard.native_index = Some(Arc::new(opened));
    let bytes = serde_json::to_vec(&guard.id_map)?;
    tokio::fs::write(PathBuf::from(&path).join(METADATA_FILE), bytes).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> IdMap {
        let mut id_map = IdMap::new(2);
        id_map.id_to_label.insert("a".into(), 1);
        id_map.id_to_label.insert("b".into(), 2);
        id_map.label_to_id.insert(1, "a".into());
        id_map.label_to_id.insert(2, "b".into());
        id_map.embeddings.insert(1, vec![0.0, 0.0]);
        id_map.embeddings.insert(2, vec![10.0, 10.0]);
        id_map.total_elements_added = 2;
        id_map
    }

    #[test]
    fn exact_scan_returns_nearest() {
        let id_map = sample_map();
        let results = brute_force_query(&id_map, &Space::L2, &[0.1, 0.1], None, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].offset_id, 1);
    }

    #[test]
    fn native_not_ready_below_threshold() {
        let id_map = sample_map();
        assert!(!native_is_ready(&id_map));
    }
}
