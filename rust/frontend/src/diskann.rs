use std::{collections::HashMap, path::PathBuf, sync::OnceLock};

use chroma_diskann::{chroma_snapshot::ChromaSnapshot, SearchOptions};
use chroma_error::{ChromaError, ErrorCodes};
use chroma_types::{Collection, CollectionUuid, Include, IncludeList, QueryResponse};
use uuid::Uuid;

const CONFIG_ENV: &str = "CHROMA_DISKANN_SNAPSHOTS";
static BINDINGS: OnceLock<Result<HashMap<Uuid, PathBuf>, String>> = OnceLock::new();

#[derive(Debug, thiserror::Error)]
pub enum DiskAnnRouteError {
    #[error("Invalid DiskANN snapshot configuration: {0}")]
    Configuration(String),
    #[error("Collection is bound to a read-only DiskANN snapshot")]
    ReadOnly,
    #[error("DiskANN snapshot queries do not support filters or ID restrictions")]
    UnsupportedFilter,
    #[error(
        "DiskANN snapshot queries support only distances and embeddings; set include explicitly"
    )]
    UnsupportedInclude,
    #[error("DiskANN snapshots require the local executor")]
    UnsupportedExecutor,
    #[error(
        "DiskANN snapshot does not match the request collection, tenant, database or dimension"
    )]
    SourceMismatch,
    #[error("DiskANN snapshot query failed: {0}")]
    Snapshot(#[from] chroma_diskann::chroma_snapshot::ChromaSnapshotError),
    #[error("DiskANN snapshot worker failed: {0}")]
    Worker(#[from] tokio::task::JoinError),
}

impl ChromaError for DiskAnnRouteError {
    fn code(&self) -> ErrorCodes {
        match self {
            Self::ReadOnly
            | Self::UnsupportedFilter
            | Self::UnsupportedInclude
            | Self::UnsupportedExecutor => ErrorCodes::InvalidArgument,
            Self::Snapshot(chroma_diskann::chroma_snapshot::ChromaSnapshotError::Index(
                chroma_diskann::DiskAnnError::InvalidInput(_),
            )) => ErrorCodes::InvalidArgument,
            _ => ErrorCodes::Internal,
        }
    }
}

fn parse_bindings(value: &str) -> Result<HashMap<Uuid, PathBuf>, String> {
    let bindings: HashMap<Uuid, PathBuf> =
        serde_json::from_str(value).map_err(|error| error.to_string())?;
    if bindings.values().any(|path| !path.is_absolute()) {
        return Err("snapshot paths must be absolute".to_string());
    }
    Ok(bindings)
}

fn bindings() -> Result<&'static HashMap<Uuid, PathBuf>, DiskAnnRouteError> {
    BINDINGS
        .get_or_init(|| match std::env::var(CONFIG_ENV) {
            Ok(value) => parse_bindings(&value),
            Err(std::env::VarError::NotPresent) => Ok(HashMap::new()),
            Err(error) => Err(error.to_string()),
        })
        .as_ref()
        .map_err(|error| DiskAnnRouteError::Configuration(error.clone()))
}

pub fn validate_configuration() -> Result<(), DiskAnnRouteError> {
    bindings().map(|_| ())
}

pub fn snapshot_path(collection_id: CollectionUuid) -> Result<Option<PathBuf>, DiskAnnRouteError> {
    Ok(bindings()?.get(&collection_id.0).cloned())
}

pub fn ensure_writable(collection_id: CollectionUuid) -> Result<(), DiskAnnRouteError> {
    if snapshot_path(collection_id)?.is_some() {
        return Err(DiskAnnRouteError::ReadOnly);
    }
    Ok(())
}

pub async fn query(
    path: PathBuf,
    collection: Collection,
    embeddings: Vec<Vec<f32>>,
    count: u32,
    include: IncludeList,
    has_filter: bool,
) -> Result<QueryResponse, DiskAnnRouteError> {
    if has_filter {
        return Err(DiskAnnRouteError::UnsupportedFilter);
    }
    if include
        .0
        .iter()
        .any(|field| !matches!(field, Include::Distance | Include::Embedding))
    {
        return Err(DiskAnnRouteError::UnsupportedInclude);
    }
    tokio::task::spawn_blocking(move || {
        query_blocking(path, collection, embeddings, count, include)
    })
    .await?
}

fn query_blocking(
    path: PathBuf,
    collection: Collection,
    embeddings: Vec<Vec<f32>>,
    count: u32,
    include: IncludeList,
) -> Result<QueryResponse, DiskAnnRouteError> {
    let snapshot = ChromaSnapshot::open(&path)?;
    let info = snapshot.info();
    if info.source.collection_id != collection.collection_id.to_string()
        || info.source.tenant != collection.tenant
        || info.source.database != collection.database
        || Some(info.dimensions)
            != collection
                .dimension
                .and_then(|value| usize::try_from(value).ok())
    {
        return Err(DiskAnnRouteError::SourceMismatch);
    }
    let return_embeddings = include.0.contains(&Include::Embedding);
    let return_distances = include.0.contains(&Include::Distance);
    let query_count = embeddings.len();
    let mut response = QueryResponse {
        ids: Vec::with_capacity(query_count),
        embeddings: return_embeddings.then(Vec::new),
        documents: None,
        uris: None,
        metadatas: None,
        distances: return_distances.then(Vec::new),
        include: include.0,
    };
    let mut native_comparisons = 0_u64;
    for embedding in embeddings {
        let result = snapshot.search(&embedding, count as usize, &SearchOptions::default())?;
        native_comparisons += u64::from(result.native_comparisons);
        let mut ids = Vec::with_capacity(result.results.len());
        let mut vectors = Vec::new();
        let mut distances = Vec::new();
        for neighbor in result.results {
            if return_embeddings {
                vectors.push(Some(snapshot.get_vector(&neighbor.id)?));
            }
            if return_distances {
                distances.push(Some(neighbor.distance));
            }
            ids.push(neighbor.id);
        }
        response.ids.push(ids);
        if let Some(batches) = response.embeddings.as_mut() {
            batches.push(vectors);
        }
        if let Some(batches) = response.distances.as_mut() {
            batches.push(distances);
        }
    }
    tracing::info!(
        backend = "diskann",
        collection_id = %collection.collection_id,
        query_count,
        native_comparisons,
        "DiskANN snapshot query completed"
    );
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn snapshot_bindings_require_uuid_and_absolute_path() {
        let directory = tempfile::tempdir().unwrap();
        let id = Uuid::new_v4();
        let expected = HashMap::from([(id, directory.path().to_path_buf())]);
        let bindings = parse_bindings(&serde_json::to_string(&expected).unwrap()).unwrap();
        assert_eq!(bindings[&id], directory.path());
        assert!(parse_bindings("not-json").is_err());
        assert!(parse_bindings(r#"{"not-a-uuid":"/tmp/index"}"#).is_err());
        let relative = HashMap::from([(id, PathBuf::from("relative/index"))]);
        assert!(parse_bindings(&serde_json::to_string(&relative).unwrap()).is_err());
    }

    #[tokio::test]
    async fn unsupported_query_options_fail_before_open() {
        let collection = Collection::test_collection(4);
        assert!(matches!(
            query(
                PathBuf::from("not-opened"),
                collection.clone(),
                vec![vec![1.0; 4]],
                5,
                IncludeList(vec![Include::Distance]),
                true
            )
            .await,
            Err(DiskAnnRouteError::UnsupportedFilter)
        ));
        assert!(matches!(
            query(
                PathBuf::from("not-opened"),
                collection,
                vec![vec![1.0; 4]],
                5,
                IncludeList::default_query(),
                false
            )
            .await,
            Err(DiskAnnRouteError::UnsupportedInclude)
        ));
    }

    #[test]
    fn native_query_preserves_chroma_response_and_source_scope() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index");
        let collection = Collection::test_collection(4);
        let mut random = StdRng::seed_from_u64(71);
        let vectors = (0..512)
            .map(|_| {
                (0..4)
                    .map(|_| random.gen_range(-1.0..1.0))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        chroma_diskann::build_index(&path, &vectors, &Default::default()).unwrap();
        let ids = (0..512)
            .map(|row| format!("doc-{row:04}"))
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "version": 1,
            "info": {
                "source": {
                    "tenant": collection.tenant.clone(),
                    "database": collection.database.clone(),
                    "collection_id": collection.collection_id.to_string(),
                    "collection_name": collection.name.clone()
                },
                "points": 512,
                "dimensions": 4,
                "metric": "l2"
            },
            "ids": ids
        });
        let snapshot_manifest = path.join("chroma_snapshot.json");
        std::fs::write(&snapshot_manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let response = query_blocking(
            path.clone(),
            collection.clone(),
            vec![vectors[17].clone(), vectors[43].clone()],
            5,
            IncludeList(vec![Include::Distance, Include::Embedding]),
        )
        .unwrap();
        assert_eq!(response.ids.len(), 2);
        assert_eq!(response.ids[0][0], "doc-0017");
        assert_eq!(response.ids[1][0], "doc-0043");
        assert_eq!(response.distances.as_ref().unwrap()[0][0], Some(0.0));
        assert_eq!(
            response.embeddings.as_ref().unwrap()[0][0],
            Some(vectors[17].clone())
        );
        assert!(response.documents.is_none());
        assert!(response.metadatas.is_none());
        let mut wrong_scope = collection.clone();
        wrong_scope.tenant.push_str("-different");
        assert!(matches!(
            query_blocking(
                path.clone(),
                wrong_scope,
                vec![vectors[17].clone()],
                5,
                IncludeList(vec![Include::Distance])
            ),
            Err(DiskAnnRouteError::SourceMismatch)
        ));
        std::fs::remove_file(snapshot_manifest).unwrap();
        assert!(matches!(
            query_blocking(
                path,
                collection,
                vec![vectors[17].clone()],
                5,
                IncludeList(vec![Include::Distance])
            ),
            Err(DiskAnnRouteError::Snapshot(_))
        ));
    }
}
