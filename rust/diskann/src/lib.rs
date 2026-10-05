use std::{
    borrow::Cow,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use diskann::{graph::config, utils::ONE};
use diskann_disk::{
    build::{
        builder::build::DiskIndexBuilder,
        configuration::{MemoryBudget, NumPQChunks},
    },
    data_model::{AdHoc, CachingStrategy},
    search::{
        provider::{
            aligned_file_reader::AlignedFileReaderFactory, disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
        traits::VertexProviderFactory,
    },
    storage::{disk_index_reader::DiskIndexReader, DiskIndexWriter},
    DiskIndexBuildParameters, QuantizationType,
};
use diskann_providers::{
    model::IndexConfiguration,
    storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
    },
};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "chroma-client")]
pub mod chroma_snapshot;

pub const DISKANN_REVISION: &str = "bceaf45dc6aa694485553816b448d7b8f55df393";
const FORMAT_VERSION: u32 = 1;
const INDEX_PREFIX: &str = "index";
const VECTORS_FILE: &str = "vectors.fbin";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    L2,
    Cosine,
}

#[derive(Debug, Error)]
pub enum DiskAnnError {
    #[error("Invalid DiskANN input: {0}")]
    InvalidInput(String),
    #[error("DiskANN operation failed: {0}")]
    Native(String),
    #[error("Invalid DiskANN index: {0}")]
    InvalidIndex(String),
    #[error("DiskANN search lock is poisoned")]
    Poisoned,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Metadata(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub metric: DistanceMetric,
    pub graph_degree: usize,
    pub search_list_size: usize,
    pub pq_bytes: usize,
    pub num_threads: usize,
    pub memory_budget_gb: f64,
    pub alpha: f32,
    pub seed: u64,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::L2,
            graph_degree: 32,
            search_list_size: 64,
            pq_bytes: 8,
            num_threads: 4,
            memory_budget_gb: 1.0,
            alpha: 1.2,
            seed: 42,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    format_version: u32,
    diskann_revision: String,
    metric: DistanceMetric,
    dimensions: u32,
    num_points: u32,
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub search_list_size: u32,
    pub beam_width: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            search_list_size: 64,
            beam_width: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Neighbor {
    pub offset_id: u32,
    pub distance: f32,
}

#[derive(Debug, Default)]
pub struct QueryResult {
    pub neighbors: Vec<Neighbor>,
    pub comparisons: u32,
}

/// A blocking, immutable DiskANN reader. IDs are zero-based input row numbers.
pub struct DiskAnnIndex {
    searcher: Mutex<DiskIndexSearcher<AdHoc<f32>>>,
    manifest: Manifest,
    vectors_path: PathBuf,
}

impl DiskAnnIndex {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, DiskAnnError> {
        let directory = fs::canonicalize(directory.as_ref())?;
        let manifest: Manifest =
            serde_json::from_reader(BufReader::new(File::open(directory.join(MANIFEST_FILE))?))?;
        if manifest.format_version != FORMAT_VERSION
            || manifest.dimensions == 0
            || manifest.num_points < 256
        {
            return Err(DiskAnnError::InvalidIndex(
                "unsupported manifest version or dimensions".to_string(),
            ));
        }
        let vectors_path = directory.join(VECTORS_FILE);
        let mut vectors = File::open(&vectors_path)?;
        if read_u32(&mut vectors)? != manifest.num_points
            || read_u32(&mut vectors)? != manifest.dimensions
            || vectors.metadata()?.len()
                != vector_byte_offset(manifest.num_points, manifest.dimensions)?
        {
            return Err(DiskAnnError::InvalidIndex(
                "original vectors do not match the manifest".to_string(),
            ));
        }

        let prefix = native_path(&directory.join(INDEX_PREFIX))?;
        let storage = FileStorageProvider;
        let reader = DiskIndexReader::new(
            get_pq_pivot_file(&prefix),
            get_compressed_pq_file(&prefix),
            &storage,
        )
        .map_err(native_error)?;
        let factory =
            DiskVertexProviderFactory::<AdHoc<f32>, AlignedFileReaderFactory>::from_disk_index_path(
                get_disk_index_file(&prefix),
                CachingStrategy::None,
            )
            .map_err(native_error)?;
        let header = factory.get_header().map_err(native_error)?;
        if header.metadata().dims != manifest.dimensions as usize
            || header.metadata().num_pts != u64::from(manifest.num_points)
        {
            return Err(DiskAnnError::InvalidIndex(
                "native graph does not match the manifest".to_string(),
            ));
        }
        let searcher =
            DiskIndexSearcher::new(1, u32::MAX as usize, &reader, factory, Metric::L2, None)
                .map_err(native_error)?;
        Ok(Self {
            searcher: Mutex::new(searcher),
            manifest,
            vectors_path,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.manifest.dimensions as usize
    }

    pub fn len(&self) -> usize {
        self.manifest.num_points as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn metric(&self) -> DistanceMetric {
        self.manifest.metric
    }

    /// Uses native graph search only; there is no exact-scan or HNSW fallback.
    pub fn search(
        &self,
        query: &[f32],
        count: usize,
        options: &SearchOptions,
    ) -> Result<QueryResult, DiskAnnError> {
        validate_vector(query, self.dimensions(), self.metric())?;
        if options.search_list_size == 0 || !(1..=128).contains(&options.beam_width) {
            return Err(invalid_input("invalid search list size or beam width"));
        }
        let count = count.min(self.len()) as u32;
        if count == 0 {
            return Ok(QueryResult::default());
        }
        let indexed_query = if self.metric() == DistanceMetric::Cosine {
            let norm = vector_norm(query);
            Cow::Owned(
                query
                    .iter()
                    .map(|value| (f64::from(*value) / norm) as f32)
                    .collect::<Vec<_>>(),
            )
        } else {
            Cow::Borrowed(query)
        };
        let searcher = self.searcher.lock().map_err(|_| DiskAnnError::Poisoned)?;
        let result = searcher
            .search(
                indexed_query.as_ref(),
                count,
                options
                    .search_list_size
                    .max(count)
                    .min(self.manifest.num_points),
                Some(options.beam_width),
                SearchMode::graph(),
            )
            .map_err(native_error)?;
        let valid_count = result.stats.result_count as usize;
        if valid_count > result.results.len() || valid_count > count as usize {
            return Err(DiskAnnError::InvalidIndex(
                "native search returned an invalid result count".to_string(),
            ));
        }
        let mut neighbors = Vec::with_capacity(valid_count);
        for item in result.results.into_iter().take(valid_count) {
            if item.vertex_id >= self.manifest.num_points || !item.distance.is_finite() {
                return Err(DiskAnnError::InvalidIndex(
                    "native search returned an invalid ID or distance".to_string(),
                ));
            }
            let distance = match self.metric() {
                DistanceMetric::L2 => item.distance,
                DistanceMetric::Cosine => (item.distance * 0.5).clamp(0.0, 2.0),
            };
            neighbors.push(Neighbor {
                offset_id: item.vertex_id,
                distance,
            });
        }
        Ok(QueryResult {
            neighbors,
            comparisons: result.stats.cmps,
        })
    }

    pub fn get_vector(&self, offset_id: u32) -> Result<Vec<f32>, DiskAnnError> {
        if offset_id >= self.manifest.num_points {
            return Err(invalid_input("vector ID is out of range"));
        }
        let mut file = File::open(&self.vectors_path)?;
        file.seek(SeekFrom::Start(vector_byte_offset(
            offset_id,
            self.manifest.dimensions,
        )?))?;
        let mut vector = Vec::with_capacity(self.dimensions());
        for _ in 0..self.dimensions() {
            let mut bytes = [0_u8; 4];
            file.read_exact(&mut bytes)?;
            vector.push(f32::from_le_bytes(bytes));
        }
        validate_vector(&vector, self.dimensions(), self.metric())?;
        Ok(vector)
    }
}

/// Builds a new, immutable disk index. The destination must not already exist.
/// This is a blocking API; async callers must use a blocking worker.
pub fn build_index(
    directory: impl AsRef<Path>,
    vectors: &[Vec<f32>],
    options: &BuildOptions,
) -> Result<(), DiskAnnError> {
    let (num_points, dimensions) = validate_build(vectors, options)?;
    let directory = directory.as_ref();
    let parent = directory
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let index_prefix = native_path(&directory.join(INDEX_PREFIX))?;
    fs::create_dir_all(parent)?;
    fs::create_dir(directory)?;

    let originals = directory.join(VECTORS_FILE);
    write_vectors(&originals, vectors, num_points, dimensions, false)?;
    let input = if options.metric == DistanceMetric::Cosine {
        let normalized = directory.join("normalized.fbin");
        write_vectors(&normalized, vectors, num_points, dimensions, true)?;
        normalized
    } else {
        originals
    };

    let graph_config = config::Builder::new_with(
        options.graph_degree,
        config::MaxDegree::default_slack(),
        options.search_list_size,
        Metric::L2.into(),
        |builder| {
            builder.saturate_after_prune(true).alpha(options.alpha);
        },
    )
    .build()
    .map_err(native_error)?;
    let index_config = IndexConfiguration::new(
        Metric::L2,
        dimensions as usize,
        num_points as usize,
        ONE,
        options.num_threads,
        graph_config,
    )
    .with_pseudo_rng_from_seed(options.seed);
    let build_params = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(options.memory_budget_gb).map_err(native_error)?,
        QuantizationType::FP,
        NumPQChunks::new_with(
            options.pq_bytes.min(dimensions as usize),
            dimensions as usize,
        )
        .map_err(native_error)?,
    );
    let writer = DiskIndexWriter::new(native_path(&input)?, index_prefix, None, 4096)
        .map_err(native_error)?;
    let storage = FileStorageProvider;
    DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, build_params, index_config, writer)
        .map_err(native_error)?
        .build()
        .map_err(native_error)?;

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            OpenOptions::new()
                .write(true)
                .open(entry.path())?
                .sync_all()?;
        }
    }
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        diskann_revision: DISKANN_REVISION.to_string(),
        metric: options.metric,
        dimensions,
        num_points,
    };
    let pending_manifest = directory.join("manifest.json.tmp");
    let mut output = BufWriter::new(File::create_new(&pending_manifest)?);
    serde_json::to_writer(&mut output, &manifest)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    fs::rename(pending_manifest, directory.join(MANIFEST_FILE))?;
    sync_directory(directory)?;
    sync_directory(parent)?;
    Ok(())
}

fn validate_build(
    vectors: &[Vec<f32>],
    options: &BuildOptions,
) -> Result<(u32, u32), DiskAnnError> {
    if vectors.len() < 256 {
        return Err(invalid_input("PQ training requires at least 256 vectors"));
    }
    let num_points =
        u32::try_from(vectors.len()).map_err(|_| invalid_input("too many vectors for u32 IDs"))?;
    let dimensions =
        u32::try_from(vectors[0].len()).map_err(|_| invalid_input("too many dimensions"))?;
    if dimensions == 0 {
        return Err(invalid_input("vectors must have positive dimensionality"));
    }
    if options.graph_degree == 0
        || options.graph_degree >= vectors.len()
        || options.search_list_size < options.graph_degree
        || options.pq_bytes == 0
        || options.num_threads == 0
        || !options.memory_budget_gb.is_finite()
        || options.memory_budget_gb <= 0.0
        || !options.alpha.is_finite()
        || options.alpha < 1.0
    {
        return Err(invalid_input("invalid build parameters"));
    }
    for vector in vectors {
        validate_vector(vector, dimensions as usize, options.metric)?;
    }
    Ok((num_points, dimensions))
}

fn validate_vector(
    vector: &[f32],
    dimensions: usize,
    metric: DistanceMetric,
) -> Result<(), DiskAnnError> {
    if vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "incorrect vector dimensions or non-finite values",
        ));
    }
    if metric == DistanceMetric::Cosine && vector_norm(vector) == 0.0 {
        return Err(invalid_input("cosine vectors must be nonzero"));
    }
    Ok(())
}

fn vector_norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn write_vectors(
    path: &Path,
    vectors: &[Vec<f32>],
    num_points: u32,
    dimensions: u32,
    normalize: bool,
) -> Result<(), DiskAnnError> {
    let mut output = BufWriter::new(File::create_new(path)?);
    output.write_all(&num_points.to_le_bytes())?;
    output.write_all(&dimensions.to_le_bytes())?;
    for vector in vectors {
        let norm = if normalize { vector_norm(vector) } else { 1.0 };
        for value in vector {
            let indexed_value = if normalize {
                (f64::from(*value) / norm) as f32
            } else {
                *value
            };
            output.write_all(&indexed_value.to_le_bytes())?;
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

fn native_path(path: &Path) -> Result<String, DiskAnnError> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| invalid_input("DiskANN paths must be UTF-8"))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, DiskAnnError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn vector_byte_offset(row: u32, dimensions: u32) -> Result<u64, DiskAnnError> {
    u64::from(row)
        .checked_mul(u64::from(dimensions))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| DiskAnnError::InvalidIndex("vector file size overflow".to_string()))
}

fn native_error(error: impl std::fmt::Display) -> DiskAnnError {
    DiskAnnError::Native(error.to_string())
}

fn invalid_input(message: &str) -> DiskAnnError {
    DiskAnnError::InvalidInput(message.to_string())
}

fn sync_directory(path: &Path) -> Result<(), DiskAnnError> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::HashSet;

    fn sample_vectors() -> Vec<Vec<f32>> {
        let mut random = StdRng::seed_from_u64(71);
        (0..512)
            .map(|_| (0..8).map(|_| random.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    fn exact_distance(metric: DistanceMetric, query: &[f32], vector: &[f32]) -> f32 {
        match metric {
            DistanceMetric::L2 => query
                .iter()
                .zip(vector)
                .map(|(left, right)| (left - right).powi(2))
                .sum(),
            DistanceMetric::Cosine => {
                let dot_product: f64 = query
                    .iter()
                    .zip(vector)
                    .map(|(left, right)| f64::from(*left) * f64::from(*right))
                    .sum();
                (1.0 - dot_product / (vector_norm(query) * vector_norm(vector))) as f32
            }
        }
    }

    #[test]
    fn native_build_reopen_and_query() {
        let originals = sample_vectors();
        for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
            let parent = tempfile::tempdir().unwrap();
            let directory = parent.path().join("index");
            let options = BuildOptions {
                metric,
                num_threads: 2,
                pq_bytes: 4,
                ..Default::default()
            };
            build_index(&directory, &originals, &options).unwrap();
            let prefix = native_path(&directory.join(INDEX_PREFIX)).unwrap();
            for artifact in [
                get_disk_index_file(&prefix),
                get_pq_pivot_file(&prefix),
                get_compressed_pq_file(&prefix),
            ] {
                assert!(fs::metadata(artifact).unwrap().len() > 0);
            }

            let index = DiskAnnIndex::open(&directory).unwrap();
            assert_eq!(index.len(), originals.len());
            assert_eq!(index.dimensions(), 8);
            assert_eq!(index.metric(), metric);
            assert!(!index.is_empty());
            let search_options = SearchOptions::default();
            for row in [17_usize, 173, 400] {
                let query = if metric == DistanceMetric::Cosine {
                    originals[row].iter().map(|value| value * 3.0).collect()
                } else {
                    originals[row].clone()
                };
                let result = index.search(&query, 8, &search_options).unwrap();
                assert_eq!(result.neighbors.len(), 8);
                assert_eq!(result.neighbors[0].offset_id, row as u32);
                assert_eq!(index.get_vector(row as u32).unwrap(), originals[row]);
                let mut unique = HashSet::new();
                for neighbor in &result.neighbors {
                    assert!(unique.insert(neighbor.offset_id));
                    let expected =
                        exact_distance(metric, &query, &originals[neighbor.offset_id as usize]);
                    assert!((neighbor.distance - expected).abs() < 1e-4);
                }
                assert!(result
                    .neighbors
                    .windows(2)
                    .all(|pair| { pair[0].distance <= pair[1].distance + 1e-6 }));
            }

            assert!(index.search(&[1.0], 1, &search_options).is_err());
            assert!(index.search(&[f32::NAN; 8], 1, &search_options).is_err());
            assert!(index.get_vector(originals.len() as u32).is_err());
            assert!(index
                .search(&originals[0], 0, &search_options)
                .unwrap()
                .neighbors
                .is_empty());
            let oversized = index
                .search(&originals[0], usize::MAX, &search_options)
                .unwrap();
            assert!(oversized.neighbors.len() <= originals.len());
            assert_eq!(
                oversized
                    .neighbors
                    .iter()
                    .map(|neighbor| neighbor.offset_id)
                    .collect::<HashSet<_>>()
                    .len(),
                oversized.neighbors.len()
            );
            if metric == DistanceMetric::Cosine {
                assert!(index.search(&[0.0; 8], 1, &search_options).is_err());
            }
            drop(index);

            let reopened = DiskAnnIndex::open(&directory).unwrap();
            let result = reopened.search(&originals[17], 1, &search_options).unwrap();
            assert_eq!(result.neighbors[0].offset_id, 17);
            assert_eq!(reopened.get_vector(17).unwrap(), originals[17]);
        }
    }

    #[test]
    fn rejects_invalid_input_before_creating_files() {
        let parent = tempfile::tempdir().unwrap();
        let directory = parent.path().join("invalid");
        let mut vectors = sample_vectors();
        let options = BuildOptions::default();
        assert!(matches!(
            build_index(&directory, &vectors[..255], &options),
            Err(DiskAnnError::InvalidInput(_))
        ));
        vectors[0][0] = f32::INFINITY;
        assert!(build_index(&directory, &vectors, &options).is_err());
        vectors[0] = vec![0.0; 8];
        let cosine = BuildOptions {
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        assert!(build_index(&directory, &vectors, &cosine).is_err());
        vectors[0].push(1.0);
        assert!(build_index(&directory, &vectors, &options).is_err());
        assert!(!directory.exists());
    }

    #[test]
    fn refuses_to_overwrite_an_existing_directory() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("keep.txt");
        fs::write(&marker, b"keep").unwrap();
        assert!(matches!(
            build_index(directory.path(), &sample_vectors(), &BuildOptions::default()),
            Err(DiskAnnError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(fs::read(marker).unwrap(), b"keep");
        assert!(!directory.path().join(MANIFEST_FILE).exists());
    }

    #[test]
    fn rejects_incomplete_and_inconsistent_indexes() {
        let directory = tempfile::tempdir().unwrap();
        assert!(DiskAnnIndex::open(directory.path()).is_err());
        let vectors = sample_vectors();
        write_vectors(
            &directory.path().join(VECTORS_FILE),
            &vectors,
            512,
            8,
            false,
        )
        .unwrap();
        let mut manifest = Manifest {
            format_version: FORMAT_VERSION + 1,
            diskann_revision: DISKANN_REVISION.to_string(),
            metric: DistanceMetric::L2,
            dimensions: 8,
            num_points: 512,
        };
        let manifest_path = directory.path().join(MANIFEST_FILE);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(
            DiskAnnIndex::open(directory.path()),
            Err(DiskAnnError::InvalidIndex(_))
        ));
        manifest.format_version = FORMAT_VERSION;
        manifest.num_points = 513;
        fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(
            DiskAnnIndex::open(directory.path()),
            Err(DiskAnnError::InvalidIndex(_))
        ));
    }
}
