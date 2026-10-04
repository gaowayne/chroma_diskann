use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use chroma::{
    types::{Include, IncludeList},
    ChromaCollection,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{build_index, BuildOptions, DiskAnnError, DiskAnnIndex, DistanceMetric, SearchOptions};

const SNAPSHOT_FILE: &str = "chroma_snapshot.json";
const PAGE_SIZE: u32 = 256;

#[derive(Debug, Error)]
pub enum ChromaSnapshotError {
    #[error("Chroma export failed: {0}")]
    Chroma(String),
    #[error("Invalid Chroma snapshot: {0}")]
    Invalid(String),
    #[error(transparent)]
    Index(#[from] DiskAnnError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Worker(#[from] tokio::task::JoinError),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceCollection {
    pub tenant: String,
    pub database: String,
    pub collection_id: String,
    pub collection_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SnapshotInfo {
    pub source: SourceCollection,
    pub points: usize,
    pub dimensions: usize,
    pub metric: DistanceMetric,
}

#[derive(Deserialize, Serialize)]
struct SnapshotManifest {
    version: u32,
    info: SnapshotInfo,
    ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SnapshotNeighbor {
    pub id: String,
    pub distance: f32,
}

#[derive(Debug, Serialize)]
pub struct SnapshotQueryResult {
    pub results: Vec<SnapshotNeighbor>,
    pub native_comparisons: u32,
}

/// A read-only copy of a Chroma collection, queried by the native DiskANN engine.
pub struct ChromaSnapshot {
    index: DiskAnnIndex,
    manifest: SnapshotManifest,
    offsets: HashMap<String, u32>,
}

/// Exports IDs and embeddings through the Chroma SDK, then builds on a blocking worker.
/// Pause writes to the source collection: count checks do not provide snapshot isolation.
pub async fn build_from_collection(
    collection: &ChromaCollection,
    directory: PathBuf,
    options: BuildOptions,
) -> Result<SnapshotInfo, ChromaSnapshotError> {
    if directory.try_exists()? {
        return Err(invalid("destination already exists"));
    }
    let expected_count = collection.count().await.map_err(export_error)?;
    if expected_count < 256 {
        return Err(invalid("DiskANN PQ requires at least 256 source records"));
    }
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    let mut offset = 0;
    while offset < expected_count {
        let limit = PAGE_SIZE.min(expected_count - offset);
        let page = collection
            .get(
                None,
                None,
                Some(limit),
                Some(offset),
                Some(IncludeList(vec![Include::Embedding])),
            )
            .await
            .map_err(export_error)?;
        let embeddings = page
            .embeddings
            .ok_or_else(|| invalid("Chroma did not return embeddings"))?;
        if page.ids.len() != limit as usize || embeddings.len() != page.ids.len() {
            return Err(invalid(
                "source record count changed or embeddings are missing",
            ));
        }
        ids.extend(page.ids);
        vectors.extend(embeddings);
        offset += limit;
    }
    if collection.count().await.map_err(export_error)? != expected_count {
        return Err(invalid("source record count changed during export"));
    }
    validate_ids(&ids, vectors.len())?;
    let source = SourceCollection {
        tenant: collection.tenant().to_string(),
        database: collection.database().to_string(),
        collection_id: collection.id().to_string(),
        collection_name: collection.name().to_string(),
    };
    tokio::task::spawn_blocking(move || build_snapshot(&directory, source, ids, vectors, &options))
        .await?
}

impl ChromaSnapshot {
    /// Opens a completed snapshot without contacting Chroma. This is a blocking API.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, ChromaSnapshotError> {
        let directory = directory.as_ref();
        let manifest: SnapshotManifest =
            serde_json::from_reader(BufReader::new(File::open(directory.join(SNAPSHOT_FILE))?))?;
        if manifest.version != 1 {
            return Err(invalid("unsupported snapshot version"));
        }
        let index = DiskAnnIndex::open(directory)?;
        if manifest.info.points != index.len()
            || manifest.info.dimensions != index.dimensions()
            || manifest.info.metric != index.metric()
        {
            return Err(invalid("snapshot metadata does not match the native index"));
        }
        let offsets = validate_ids(&manifest.ids, index.len())?;
        Ok(Self {
            index,
            manifest,
            offsets,
        })
    }

    pub fn info(&self) -> &SnapshotInfo {
        &self.manifest.info
    }

    pub fn search(
        &self,
        query: &[f32],
        count: usize,
        options: &SearchOptions,
    ) -> Result<SnapshotQueryResult, ChromaSnapshotError> {
        let result = self.index.search(query, count, options)?;
        let mut results = Vec::with_capacity(result.neighbors.len());
        for neighbor in result.neighbors {
            let id = self
                .manifest
                .ids
                .get(neighbor.offset_id as usize)
                .ok_or_else(|| invalid("native result has no Chroma ID"))?;
            results.push(SnapshotNeighbor {
                id: id.clone(),
                distance: neighbor.distance,
            });
        }
        Ok(SnapshotQueryResult {
            results,
            native_comparisons: result.comparisons,
        })
    }

    pub fn get_vector(&self, id: &str) -> Result<Vec<f32>, ChromaSnapshotError> {
        let offset = self
            .offsets
            .get(id)
            .ok_or_else(|| invalid("unknown Chroma ID"))?;
        Ok(self.index.get_vector(*offset)?)
    }
}

fn build_snapshot(
    directory: &Path,
    source: SourceCollection,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    options: &BuildOptions,
) -> Result<SnapshotInfo, ChromaSnapshotError> {
    validate_ids(&ids, vectors.len())?;
    let dimensions = vectors
        .first()
        .ok_or_else(|| invalid("empty source collection"))?
        .len();
    let info = SnapshotInfo {
        source,
        points: ids.len(),
        dimensions,
        metric: options.metric,
    };
    build_index(directory, &vectors, options)?;
    let manifest = SnapshotManifest {
        version: 1,
        info: info.clone(),
        ids,
    };
    let pending = directory.join("chroma_snapshot.json.tmp");
    let mut output = BufWriter::new(File::create_new(&pending)?);
    serde_json::to_writer(&mut output, &manifest)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    fs::rename(pending, directory.join(SNAPSHOT_FILE))?;
    crate::sync_directory(directory)?;
    Ok(info)
}

fn validate_ids(
    ids: &[String],
    expected_count: usize,
) -> Result<HashMap<String, u32>, ChromaSnapshotError> {
    if ids.len() != expected_count {
        return Err(invalid("ID count does not match vector count"));
    }
    let mut offsets = HashMap::with_capacity(ids.len());
    for (offset, id) in ids.iter().enumerate() {
        let offset = u32::try_from(offset).map_err(|_| invalid("too many records"))?;
        if id.is_empty() || offsets.insert(id.clone(), offset).is_some() {
            return Err(invalid("Chroma IDs must be nonempty and unique"));
        }
    }
    Ok(offsets)
}

fn invalid(message: &str) -> ChromaSnapshotError {
    ChromaSnapshotError::Invalid(message.to_string())
}

fn export_error(error: impl std::fmt::Display) -> ChromaSnapshotError {
    ChromaSnapshotError::Chroma(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn rejects_invalid_id_mapping() {
        assert!(validate_ids(&["id-a".to_string()], 2).is_err());
        assert!(validate_ids(&[String::new()], 1).is_err());
        assert!(validate_ids(&["id-a".to_string(), "id-a".to_string()], 2).is_err());
        let offsets = validate_ids(&["id-b".to_string(), "id-a".to_string()], 2).unwrap();
        assert_eq!(offsets["id-b"], 0);
        assert_eq!(offsets["id-a"], 1);
    }

    #[test]
    fn native_snapshot_reopens_chroma_ids() {
        let parent = tempfile::tempdir().unwrap();
        let directory = parent.path().join("snapshot");
        let source = SourceCollection {
            tenant: "default_tenant".to_string(),
            database: "default_database".to_string(),
            collection_id: "test-source-id".to_string(),
            collection_name: "test-source".to_string(),
        };
        let mut random = StdRng::seed_from_u64(71);
        let vectors: Vec<Vec<f32>> = (0..512)
            .map(|_| (0..4).map(|_| random.gen_range(-1.0..1.0)).collect())
            .collect();
        let query_vector = vectors[17].clone();
        let ids = (0..512).map(|row| format!("chroma-{row:04}")).collect();
        build_snapshot(&directory, source, ids, vectors, &BuildOptions::default()).unwrap();
        let snapshot = ChromaSnapshot::open(&directory).unwrap();
        assert_eq!(snapshot.info().source.collection_name, "test-source");
        assert_eq!(snapshot.info().points, 512);
        assert_eq!(snapshot.get_vector("chroma-0017").unwrap(), query_vector);
        let result = snapshot
            .search(&query_vector, 5, &SearchOptions::default())
            .unwrap();
        assert_eq!(result.results.len(), 5);
        assert_eq!(result.results[0].id, "chroma-0017");
        assert_eq!(result.results[0].distance, 0.0);
        assert!(snapshot.get_vector("missing").is_err());
        drop(snapshot);

        let path = directory.join(SNAPSHOT_FILE);
        let mut manifest: SnapshotManifest =
            serde_json::from_reader(File::open(&path).unwrap()).unwrap();
        manifest.ids.pop();
        serde_json::to_writer(File::create(path).unwrap(), &manifest).unwrap();
        assert!(ChromaSnapshot::open(&directory).is_err());
    }
}
