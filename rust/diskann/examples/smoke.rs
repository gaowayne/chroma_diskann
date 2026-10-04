use std::{collections::HashSet, error::Error, path::Path};

use chroma_diskann::{
    build_index, BuildOptions, DiskAnnIndex, DistanceMetric, SearchOptions, DISKANN_REVISION,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const VECTOR_COUNT: usize = 512;
const DIMENSIONS: usize = 16;
const NEIGHBORS: usize = 10;
const MIN_RECALL: f64 = 0.90;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [command, directory, metric] if command == "build" => {
            let metric = match metric.as_str() {
                "l2" => DistanceMetric::L2,
                "cosine" => DistanceMetric::Cosine,
                _ => return Err("metric must be l2 or cosine".into()),
            };
            build(Path::new(directory), metric)
        }
        [command, directory] if command == "query" => query(Path::new(directory)),
        _ => Err("usage: smoke build <new-directory> <l2|cosine> | smoke query <directory>".into()),
    }
}

fn vectors(seed: u64, count: usize) -> Vec<Vec<f32>> {
    let mut random = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (0..DIMENSIONS)
                .map(|_| random.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn build(directory: &Path, metric: DistanceMetric) -> Result<(), Box<dyn Error>> {
    println!(
        "stage=build backend=diskann-disk revision={DISKANN_REVISION} metric={metric:?} points={VECTOR_COUNT} dimensions={DIMENSIONS} path={}",
        directory.display()
    );
    let dataset = vectors(71, VECTOR_COUNT);
    build_index(
        directory,
        &dataset,
        &BuildOptions {
            metric,
            num_threads: 2,
            ..Default::default()
        },
    )?;
    println!("stage=build status=ok");
    Ok(())
}

fn query(directory: &Path) -> Result<(), Box<dyn Error>> {
    println!("stage=open path={}", directory.display());
    let index = DiskAnnIndex::open(directory)?;
    if index.len() != VECTOR_COUNT || index.dimensions() != DIMENSIONS {
        return Err("the index does not match this smoke dataset".into());
    }
    let dataset = vectors(71, VECTOR_COUNT);
    for (offset_id, expected) in dataset.iter().enumerate() {
        if index.get_vector(offset_id as u32)? != *expected {
            return Err(format!("original vector {offset_id} changed after reopening").into());
        }
    }
    println!("stage=open status=ok metric={:?}", index.metric());

    let mut queries = vectors(97, 16);
    if index.metric() == DistanceMetric::Cosine {
        for query in &mut queries {
            for value in query {
                *value *= 3.0;
            }
        }
    }
    let options = SearchOptions {
        search_list_size: 128,
        ..Default::default()
    };
    let mut matched = 0;
    let mut comparisons = 0_u64;
    for (query_number, query) in queries.iter().enumerate() {
        let result = index.search(query, NEIGHBORS, &options)?;
        if result.neighbors.len() != NEIGHBORS {
            return Err(
                format!("query {query_number} returned fewer than {NEIGHBORS} neighbors").into(),
            );
        }
        let mut exact = dataset
            .iter()
            .enumerate()
            .map(|(offset_id, vector)| (offset_id as u32, distance(index.metric(), query, vector)))
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
        let expected_ids: HashSet<u32> =
            exact.iter().take(NEIGHBORS).map(|entry| entry.0).collect();
        let mut returned_ids = HashSet::new();
        for neighbor in &result.neighbors {
            if !returned_ids.insert(neighbor.offset_id) {
                return Err(format!("query {query_number} returned a duplicate ID").into());
            }
            let original = dataset
                .get(neighbor.offset_id as usize)
                .ok_or("native search returned an out-of-range ID")?;
            let expected = distance(index.metric(), query, original);
            if (f64::from(neighbor.distance) - expected).abs() > 1e-4 * expected.abs().max(1.0) {
                return Err(format!("query {query_number} returned an incorrect distance").into());
            }
        }
        if result
            .neighbors
            .windows(2)
            .any(|pair| pair[0].distance > pair[1].distance + 1e-5)
        {
            return Err(format!("query {query_number} results are not sorted").into());
        }
        matched += returned_ids.intersection(&expected_ids).count();
        comparisons += u64::from(result.comparisons);
    }
    let recall = matched as f64 / (queries.len() * NEIGHBORS) as f64;
    println!(
        "stage=query queries={} recall_at_{NEIGHBORS}={recall:.4} native_comparisons={comparisons}",
        queries.len()
    );
    if comparisons == 0 {
        return Err("no native distance comparisons were reported".into());
    }
    if recall < MIN_RECALL {
        return Err(format!("smoke recall {recall:.4} is below {MIN_RECALL:.2}").into());
    }
    println!("stage=query status=ok");
    Ok(())
}

fn distance(metric: DistanceMetric, query: &[f32], vector: &[f32]) -> f64 {
    match metric {
        DistanceMetric::L2 => query
            .iter()
            .zip(vector)
            .map(|(left, right)| (f64::from(*left) - f64::from(*right)).powi(2))
            .sum(),
        DistanceMetric::Cosine => {
            let dot: f64 = query
                .iter()
                .zip(vector)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum();
            let query_norm: f64 = query.iter().map(|value| f64::from(*value).powi(2)).sum();
            let vector_norm: f64 = vector.iter().map(|value| f64::from(*value).powi(2)).sum();
            1.0 - dot / (query_norm * vector_norm).sqrt()
        }
    }
}
