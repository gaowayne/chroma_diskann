use crate::{default_space, hnsw_configuration::Space, DiskAnnIndexConfig};
use serde::{Deserialize, Serialize};
use validator::Validate;

pub fn default_graph_degree() -> usize {
    32
}

pub fn default_build_list_size() -> usize {
    64
}

pub fn default_search_list_size() -> u32 {
    64
}

pub fn default_beam_width() -> usize {
    4
}

pub fn default_pq_bytes() -> usize {
    8
}

pub fn default_diskann_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

pub fn default_memory_budget_gb() -> f64 {
    1.0
}

pub fn default_alpha() -> f32 {
    1.2
}

fn default_space_diskann() -> Space {
    Space::L2
}

/// Internal (fully-resolved) DiskANN configuration stored on collections/segments.
#[derive(Clone, Debug, Serialize, Deserialize, Validate, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct InternalDiskAnnConfiguration {
    #[serde(default = "default_space_diskann")]
    pub space: Space,
    /// Max graph degree (DiskANN R / graph_degree).
    #[serde(default = "default_graph_degree")]
    #[validate(range(min = 4, max = 256))]
    pub graph_degree: usize,
    /// Candidate list size used during index build (DiskANN L_build).
    #[serde(default = "default_build_list_size")]
    #[validate(range(min = 10, max = 1024))]
    pub build_list_size: usize,
    /// Candidate list size used during search (DiskANN L_search).
    #[serde(default = "default_search_list_size")]
    #[validate(range(min = 10, max = 1024))]
    pub search_list_size: u32,
    /// Beam width for disk-based search.
    #[serde(default = "default_beam_width")]
    #[validate(range(min = 1, max = 64))]
    pub beam_width: usize,
    /// Product-quantization bytes per vector (0 = no PQ).
    #[serde(default = "default_pq_bytes")]
    #[validate(range(max = 64))]
    pub pq_bytes: usize,
    #[serde(default = "default_diskann_num_threads")]
    pub num_threads: usize,
    /// Approximate build-time memory budget in GiB.
    #[serde(default = "default_memory_budget_gb")]
    #[validate(range(min = 0.1, max = 1024.0))]
    pub memory_budget_gb: f64,
    /// Robust pruning alpha.
    #[serde(default = "default_alpha")]
    #[validate(range(min = 1.0, max = 2.0))]
    pub alpha: f32,
}

impl Default for InternalDiskAnnConfiguration {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl From<(Option<&Space>, &DiskAnnIndexConfig)> for InternalDiskAnnConfiguration {
    fn from((space, config): (Option<&Space>, &DiskAnnIndexConfig)) -> Self {
        InternalDiskAnnConfiguration {
            space: space.unwrap_or(&default_space()).clone(),
            graph_degree: config.graph_degree.unwrap_or(default_graph_degree()),
            build_list_size: config
                .build_list_size
                .unwrap_or(default_build_list_size()),
            search_list_size: config
                .search_list_size
                .unwrap_or(default_search_list_size()),
            beam_width: config.beam_width.unwrap_or(default_beam_width()),
            pq_bytes: config.pq_bytes.unwrap_or(default_pq_bytes()),
            num_threads: config
                .num_threads
                .unwrap_or(default_diskann_num_threads()),
            memory_budget_gb: config
                .memory_budget_gb
                .unwrap_or(default_memory_budget_gb()),
            alpha: config.alpha.unwrap_or(default_alpha()),
        }
    }
}

/// Public DiskANN configuration accepted on create-collection (legacy config path).
#[derive(Clone, Debug, Serialize, Deserialize, Validate, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct DiskAnnConfiguration {
    pub space: Option<Space>,
    pub graph_degree: Option<usize>,
    pub build_list_size: Option<usize>,
    pub search_list_size: Option<u32>,
    pub beam_width: Option<usize>,
    pub pq_bytes: Option<usize>,
    pub num_threads: Option<usize>,
    pub memory_budget_gb: Option<f64>,
    pub alpha: Option<f32>,
}

impl From<InternalDiskAnnConfiguration> for DiskAnnConfiguration {
    fn from(config: InternalDiskAnnConfiguration) -> Self {
        Self {
            space: Some(config.space),
            graph_degree: Some(config.graph_degree),
            build_list_size: Some(config.build_list_size),
            search_list_size: Some(config.search_list_size),
            beam_width: Some(config.beam_width),
            pq_bytes: Some(config.pq_bytes),
            num_threads: Some(config.num_threads),
            memory_budget_gb: Some(config.memory_budget_gb),
            alpha: Some(config.alpha),
        }
    }
}

impl From<DiskAnnConfiguration> for InternalDiskAnnConfiguration {
    fn from(config: DiskAnnConfiguration) -> Self {
        Self {
            space: config.space.unwrap_or(default_space_diskann()),
            graph_degree: config.graph_degree.unwrap_or(default_graph_degree()),
            build_list_size: config
                .build_list_size
                .unwrap_or(default_build_list_size()),
            search_list_size: config
                .search_list_size
                .unwrap_or(default_search_list_size()),
            beam_width: config.beam_width.unwrap_or(default_beam_width()),
            pq_bytes: config.pq_bytes.unwrap_or(default_pq_bytes()),
            num_threads: config
                .num_threads
                .unwrap_or(default_diskann_num_threads()),
            memory_budget_gb: config
                .memory_budget_gb
                .unwrap_or(default_memory_budget_gb()),
            alpha: config.alpha.unwrap_or(default_alpha()),
        }
    }
}

impl Default for DiskAnnConfiguration {
    fn default() -> Self {
        InternalDiskAnnConfiguration::default().into()
    }
}

/// Runtime-updatable DiskANN search parameters.
#[derive(Clone, Default, Debug, Serialize, Deserialize, Validate, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
pub struct UpdateDiskAnnConfiguration {
    #[validate(range(min = 10, max = 1024))]
    pub search_list_size: Option<u32>,
    #[validate(range(min = 1, max = 64))]
    pub beam_width: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diskann_configuration_defaults() {
        let internal = InternalDiskAnnConfiguration::default();
        assert_eq!(internal.graph_degree, default_graph_degree());
        assert_eq!(internal.build_list_size, default_build_list_size());
        assert_eq!(internal.search_list_size, default_search_list_size());
        assert_eq!(internal.beam_width, default_beam_width());
        assert_eq!(internal.pq_bytes, default_pq_bytes());
        assert_eq!(internal.alpha, default_alpha());
        assert_eq!(internal.space, default_space_diskann());
    }

    #[test]
    fn test_diskann_configuration_roundtrip() {
        let public = DiskAnnConfiguration {
            space: Some(Space::Cosine),
            graph_degree: Some(64),
            build_list_size: Some(100),
            search_list_size: Some(80),
            beam_width: Some(8),
            pq_bytes: Some(16),
            num_threads: Some(2),
            memory_budget_gb: Some(2.0),
            alpha: Some(1.1),
        };
        let internal: InternalDiskAnnConfiguration = public.into();
        assert_eq!(internal.space, Space::Cosine);
        assert_eq!(internal.graph_degree, 64);
        assert_eq!(internal.build_list_size, 100);
        assert_eq!(internal.search_list_size, 80);
        assert_eq!(internal.beam_width, 8);
        assert_eq!(internal.pq_bytes, 16);
        assert_eq!(internal.num_threads, 2);
        assert_eq!(internal.memory_budget_gb, 2.0);
        assert_eq!(internal.alpha, 1.1);
    }
}
