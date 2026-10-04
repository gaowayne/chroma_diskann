use std::{fs::File, io::Read, sync::Mutex};

use diskann::{graph::config, utils::ONE};
use diskann_disk::{
    build::builder::build::DiskIndexBuilder,
    build::configuration::{MemoryBudget, NumPQChunks},
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
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

fn native_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyfunction]
#[pyo3(signature = (data_path, index_prefix, graph_degree=32, build_search_list_size=64, pq_bytes=8, num_threads=4, alpha=1.2))]
fn build_index(
    py: Python<'_>,
    data_path: String,
    index_prefix: String,
    graph_degree: usize,
    build_search_list_size: usize,
    pq_bytes: usize,
    num_threads: usize,
    alpha: f32,
) -> PyResult<()> {
    py.allow_threads(move || {
        let mut data_file = File::open(&data_path).map_err(native_error)?;
        let mut header = [0_u8; 8];
        data_file.read_exact(&mut header).map_err(native_error)?;
        let num_points = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        let dimensions = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        if num_points < 256 || dimensions == 0 || graph_degree >= num_points {
            return Err(PyValueError::new_err(
                "DiskANN requires at least 256 vectors, positive dimensions, and graph_degree < vector count",
            ));
        }
        let expected_size = (num_points as u64)
            .checked_mul(dimensions as u64)
            .and_then(|size| size.checked_mul(4))
            .and_then(|size| size.checked_add(8))
            .ok_or_else(|| PyValueError::new_err("vector file size overflow"))?;
        if data_file.metadata().map_err(native_error)?.len() != expected_size {
            return Err(PyValueError::new_err("invalid float32 vector file length"));
        }
        if num_threads == 0 || pq_bytes == 0 || !alpha.is_finite() || alpha < 1.0 {
            return Err(PyValueError::new_err("invalid DiskANN build parameters"));
        }
        drop(data_file);
        let graph_config = config::Builder::new_with(
            graph_degree,
            config::MaxDegree::default_slack(),
            build_search_list_size,
            Metric::L2.into(),
            |builder| {
                builder.saturate_after_prune(true).alpha(alpha);
            },
        )
        .build()
        .map_err(native_error)?;
        let index_config = IndexConfiguration::new(
            Metric::L2,
            dimensions,
            num_points,
            ONE,
            num_threads,
            graph_config,
        )
        .with_pseudo_rng_from_seed(42);
        let build_params = DiskIndexBuildParameters::new(
            MemoryBudget::try_from_gb(1.0).map_err(native_error)?,
            QuantizationType::FP,
            NumPQChunks::new_with(pq_bytes.min(dimensions), dimensions).map_err(native_error)?,
        );
        let writer = DiskIndexWriter::new(data_path, index_prefix, None, 4096)
            .map_err(native_error)?;
        let storage = FileStorageProvider;
        let mut builder = DiskIndexBuilder::<AdHoc<f32>, _>::new(
            &storage,
            build_params,
            index_config,
            writer,
        )
        .map_err(native_error)?;
        builder.build().map_err(native_error)
    })
}

#[pyclass]
struct DiskIndex {
    inner: Mutex<DiskIndexSearcher<AdHoc<f32>>>,
    #[pyo3(get)]
    dimensions: usize,
    #[pyo3(get)]
    num_points: usize,
}

#[pymethods]
impl DiskIndex {
    #[new]
    fn new(py: Python<'_>, index_prefix: String) -> PyResult<Self> {
        py.allow_threads(move || {
            let storage = FileStorageProvider;
            let reader = DiskIndexReader::new(
                get_pq_pivot_file(&index_prefix),
                get_compressed_pq_file(&index_prefix),
                &storage,
            )
            .map_err(native_error)?;
            let factory = DiskVertexProviderFactory::<AdHoc<f32>, AlignedFileReaderFactory>::from_disk_index_path(
                get_disk_index_file(&index_prefix),
                CachingStrategy::None,
            )
            .map_err(native_error)?;
            let header = factory.get_header().map_err(native_error)?;
            let dimensions = header.metadata().dims;
            let num_points = header.metadata().num_pts as usize;
            let index =
                DiskIndexSearcher::new(1, u32::MAX as usize, &reader, factory, Metric::L2, None)
                    .map_err(native_error)?;
            Ok(Self {
                inner: Mutex::new(index),
                dimensions,
                num_points,
            })
        })
    }

    #[pyo3(signature = (vector, num_results, search_list_size=64, beam_width=4))]
    fn query(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        num_results: u32,
        search_list_size: u32,
        beam_width: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        if vector.len() != self.dimensions || vector.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "invalid query vector dimensions or values",
            ));
        }
        if beam_width == 0 || beam_width > 128 {
            return Err(PyValueError::new_err("beam_width must be in [1, 128]"));
        }
        if num_results == 0 {
            return Ok(Vec::new());
        }
        py.allow_threads(move || {
            let index = self.inner.lock().map_err(native_error)?;
            let result = index
                .search(
                    &vector,
                    num_results.min(self.num_points as u32),
                    search_list_size.max(num_results),
                    Some(beam_width),
                    SearchMode::graph(),
                )
                .map_err(native_error)?;
            Ok(result
                .results
                .into_iter()
                .map(|item| (item.vertex_id, item.distance))
                .collect())
        })
    }
}

#[pymodule]
fn chroma_diskann_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_index, module)?)?;
    module.add_class::<DiskIndex>()?;
    module.add("BACKEND", "diskann-disk")?;
    Ok(())
}
