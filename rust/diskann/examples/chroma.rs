use std::{
    error::Error,
    path::{Path, PathBuf},
};

use chroma::{ChromaHttpClient, ChromaHttpClientOptions};
use chroma_diskann::{
    chroma_snapshot::{build_from_collection, ChromaSnapshot},
    BuildOptions, DistanceMetric, SearchOptions,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [command, name] if command == "seed" => {
            tokio::runtime::Runtime::new()?.block_on(seed(name))
        }
        [command, name, directory] if command == "build" => {
            tokio::runtime::Runtime::new()?.block_on(build(name, directory.into(), DistanceMetric::L2))
        }
        [command, name, directory, metric] if command == "build" => {
            let metric = match metric.as_str() {
                "l2" => DistanceMetric::L2,
                "cosine" => DistanceMetric::Cosine,
                _ => return Err("metric must be l2 or cosine".into()),
            };
            tokio::runtime::Runtime::new()?.block_on(build(name, directory.into(), metric))
        }
        [command, directory, vector] if command == "query" => {
            query(Path::new(directory), serde_json::from_str(vector)?)
        }
        _ => Err("usage: chroma seed <new-collection> | chroma build <collection> <new-directory> [l2|cosine] | chroma query <directory> <vector-json>".into()),
    }
}

fn client() -> Result<ChromaHttpClient, Box<dyn Error>> {
    Ok(ChromaHttpClient::new(ChromaHttpClientOptions::from_env()?))
}

async fn seed(name: &str) -> Result<(), Box<dyn Error>> {
    let client = client()?;
    let collection = client.create_collection(name, None, None).await?;
    let mut random = StdRng::seed_from_u64(71);
    let vectors: Vec<Vec<f32>> = (0..512)
        .map(|_| (0..4).map(|_| random.gen_range(-1.0..1.0)).collect())
        .collect();
    let ids = (0..512).map(|row| format!("doc-{row:04}")).collect();
    collection.add(ids, Some(vectors), None, None, None).await?;
    let count = collection.count().await?;
    if count != 512 {
        return Err("Chroma did not report all 512 inserted demo records".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "seeded",
            "collection_id": collection.id().to_string(),
            "collection": collection.name(),
            "points": count,
            "dimensions": 4
        }))?
    );
    Ok(())
}

async fn build(
    name: &str,
    directory: PathBuf,
    metric: DistanceMetric,
) -> Result<(), Box<dyn Error>> {
    let collection = client()?.get_collection(name).await?;
    let info = build_from_collection(
        &collection,
        directory,
        BuildOptions {
            metric,
            num_threads: 2,
            ..Default::default()
        },
    )
    .await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "built",
            "backend": "diskann-disk",
            "snapshot": info
        }))?
    );
    Ok(())
}

fn query(directory: &Path, vector: Vec<f32>) -> Result<(), Box<dyn Error>> {
    let snapshot = ChromaSnapshot::open(directory)?;
    let result = snapshot.search(&vector, 5, &SearchOptions::default())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": "diskann-disk",
            "snapshot": snapshot.info(),
            "query": vector,
            "results": result.results,
            "native_comparisons": result.native_comparisons
        }))?
    );
    Ok(())
}
