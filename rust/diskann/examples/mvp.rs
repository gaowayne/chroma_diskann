use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use chroma_diskann::{build_index, BuildOptions, DiskAnnIndex, DistanceMetric, SearchOptions};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;

const VECTOR_COUNT: usize = 512;
const DIMENSIONS: usize = 4;
const DEFAULT_QUERY_ROW: u32 = 17;
const ID_FILE: &str = "mvp_ids.json";

#[derive(Deserialize, Serialize)]
struct IdMap {
    version: u32,
    ids: Vec<String>,
}

impl IdMap {
    fn validate(&self, expected_count: usize) -> Result<(), Box<dyn Error>> {
        if self.version != 1 || self.ids.len() != expected_count {
            return Err("MVP ID map version or count does not match the index".into());
        }
        let mut unique = HashSet::new();
        for id in &self.ids {
            if id.is_empty() || !unique.insert(id) {
                return Err("MVP IDs must be nonempty and unique".into());
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [command, directory] if command == "build" => {
            build(Path::new(directory), DistanceMetric::L2)
        }
        [command, directory, metric] if command == "build" => {
            let metric = match metric.as_str() {
                "l2" => DistanceMetric::L2,
                "cosine" => DistanceMetric::Cosine,
                _ => return Err("metric must be l2 or cosine".into()),
            };
            build(Path::new(directory), metric)
        }
        [command, directory] if command == "query" => query(Path::new(directory), None),
        [command, directory, vector] if command == "query" => {
            query(Path::new(directory), Some(serde_json::from_str(vector)?))
        }
        _ => Err(
            "usage: mvp build <new-directory> [l2|cosine] | mvp query <directory> [vector-json]"
                .into(),
        ),
    }
}

fn build(directory: &Path, metric: DistanceMetric) -> Result<(), Box<dyn Error>> {
    let mut random = StdRng::seed_from_u64(71);
    let vectors: Vec<Vec<f32>> = (0..VECTOR_COUNT)
        .map(|_| {
            (0..DIMENSIONS)
                .map(|_| random.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();
    let id_map = IdMap {
        version: 1,
        ids: (0..VECTOR_COUNT)
            .map(|row| format!("doc-{row:04}"))
            .collect(),
    };
    id_map.validate(vectors.len())?;
    build_index(
        directory,
        &vectors,
        &BuildOptions {
            metric,
            num_threads: 2,
            ..Default::default()
        },
    )?;

    let pending_ids = directory.join("mvp_ids.json.tmp");
    let mut output = BufWriter::new(File::create_new(&pending_ids)?);
    serde_json::to_writer(&mut output, &id_map)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    fs::rename(pending_ids, directory.join(ID_FILE))?;
    #[cfg(unix)]
    File::open(directory)?.sync_all()?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "built",
            "backend": "diskann-disk",
            "metric": metric,
            "points": vectors.len(),
            "dimensions": DIMENSIONS,
            "directory": directory.display().to_string()
        }))?
    );
    Ok(())
}

fn query(directory: &Path, supplied_vector: Option<Vec<f32>>) -> Result<(), Box<dyn Error>> {
    let id_map: IdMap =
        serde_json::from_reader(BufReader::new(File::open(directory.join(ID_FILE))?))?;
    let index = DiskAnnIndex::open(directory)?;
    id_map.validate(index.len())?;
    let query_vector = match supplied_vector {
        Some(vector) => vector,
        None => index.get_vector(DEFAULT_QUERY_ROW)?,
    };
    let result = index.search(&query_vector, 5, &SearchOptions::default())?;
    let neighbors = result
        .neighbors
        .iter()
        .map(|neighbor| {
            json!({
                "id": id_map.ids[neighbor.offset_id as usize],
                "distance": neighbor.distance
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": "diskann-disk",
            "metric": index.metric(),
            "query": query_vector,
            "results": neighbors,
            "native_comparisons": result.comparisons
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_map_rejects_inconsistent_or_invalid_ids() {
        let mut id_map = IdMap {
            version: 1,
            ids: vec!["doc-a".to_string(), "doc-b".to_string()],
        };
        assert!(id_map.validate(2).is_ok());
        assert!(id_map.validate(3).is_err());
        id_map.version = 2;
        assert!(id_map.validate(2).is_err());
        id_map.version = 1;
        id_map.ids[1] = "doc-a".to_string();
        assert!(id_map.validate(2).is_err());
        id_map.ids[1].clear();
        assert!(id_map.validate(2).is_err());
    }

    #[test]
    fn native_mvp_reopens_ids_and_queries() {
        for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
            let parent = tempfile::tempdir().unwrap();
            let directory = parent.path().join("index");
            build(&directory, metric).unwrap();

            let mut id_map: IdMap =
                serde_json::from_slice(&fs::read(directory.join(ID_FILE)).unwrap()).unwrap();
            let index = DiskAnnIndex::open(&directory).unwrap();
            id_map.validate(index.len()).unwrap();
            let vector = index.get_vector(DEFAULT_QUERY_ROW).unwrap();
            let result = index.search(&vector, 5, &SearchOptions::default()).unwrap();
            assert_eq!(result.neighbors.len(), 5);
            assert_eq!(
                id_map.ids[result.neighbors[0].offset_id as usize],
                "doc-0017"
            );
            assert!(result.neighbors[0].distance.abs() < 1e-4);
            drop(index);

            query(&directory, None).unwrap();
            query(&directory, Some(vec![0.1, 0.2, 0.3, 0.4])).unwrap();
            id_map.ids.pop();
            fs::write(
                directory.join(ID_FILE),
                serde_json::to_vec(&id_map).unwrap(),
            )
            .unwrap();
            assert!(query(&directory, None).is_err());
        }
    }
}
