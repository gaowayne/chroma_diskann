"""SIFT subset ingest + Recall@k through local PersistentClient DiskANN.

This is the first dataset demo (not the 258-point smoke test). It loads official
SIFT files (.fvecs/.ivecs or DiskANN .bin), crops the base, rebuilds exact
ground truth on that crop, upserts into a DiskANN collection, and reports Recall@k.

Cropped GT is required: the official ivecs file is computed against the full
1M base, so neighbors may lie outside the subset.

Usage (chroma_diskann repo root, after pip install -e .):

    python examples/diskann/sift_recall.py --data-dir /path/to/sift1M

    # defaults: first 10000 base vectors, 100 queries, k=10

Ubuntu example if PageANN data lives next to chroma:

    python examples/diskann/sift_recall.py \\
        --data-dir /mnt/nvme4n1/wayne/vectorsearchstudy/PageANN/sift1M \\
        --max-points 10000 --n-queries 100
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import chromadb


def read_fvecs(path: Path, max_rows: int | None = None) -> np.ndarray:
    """TexMex .fvecs: each record is uint32 dim + dim float32."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise SystemExit(f"empty or missing fvecs: {path}")
    dim = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
    record = 4 + dim * 4
    n = raw.size // record
    if max_rows is not None:
        n = min(n, max_rows)
    mat = raw[: n * record].view(np.uint32).reshape(n, 1 + dim)
    return mat[:, 1:].view(np.float32).copy()


def read_ivecs(path: Path, max_rows: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise SystemExit(f"empty or missing ivecs: {path}")
    dim = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
    record = 4 + dim * 4
    n = raw.size // record
    if max_rows is not None:
        n = min(n, max_rows)
    mat = raw[: n * record].view(np.uint32).reshape(n, 1 + dim)
    return mat[:, 1:].astype(np.int64, copy=True)


def read_bin_float(path: Path, max_rows: int | None = None) -> np.ndarray:
    """DiskANN/PageANN .bin: uint32 npts, uint32 dim, then npts*dim float32."""
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.uint32, count=2)
        if header.size != 2:
            raise SystemExit(f"invalid bin header: {path}")
        npts, dim = int(header[0]), int(header[1])
        if max_rows is not None:
            npts = min(npts, max_rows)
        data = np.fromfile(handle, dtype=np.float32, count=npts * dim)
    if data.size != npts * dim:
        raise SystemExit(f"truncated float bin: {path}")
    return data.reshape(npts, dim)


def load_vectors(data_dir: Path, kind: str, max_rows: int | None) -> np.ndarray:
    """Prefer .bin (already converted), else TexMex .fvecs/.ivecs."""
    if kind == "base":
        bin_path, vecs_path = data_dir / "sift_base.bin", data_dir / "sift_base.fvecs"
        if bin_path.exists():
            return read_bin_float(bin_path, max_rows)
        if vecs_path.exists():
            return read_fvecs(vecs_path, max_rows)
    elif kind == "query":
        bin_path, vecs_path = data_dir / "sift_query.bin", data_dir / "sift_query.fvecs"
        if bin_path.exists():
            return read_bin_float(bin_path, max_rows)
        if vecs_path.exists():
            return read_fvecs(vecs_path, max_rows)
    raise SystemExit(
        f"missing SIFT {kind} files in {data_dir} "
        f"(need sift_{kind}.bin or sift_{kind}.fvecs / sift_query.*)"
    )


def exact_knn(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Chunked L2 exact neighbors on the cropped base. Shape [n_queries, k]."""
    base_sq = np.einsum("ij,ij->i", base, base)
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    chunk = 256
    for start in range(0, queries.shape[0], chunk):
        q = queries[start : start + chunk]
        q_sq = np.einsum("ij,ij->i", q, q)[:, None]
        # ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q·b
        d2 = q_sq + base_sq[None, :] - 2.0 * (q @ base.T)
        out[start : start + q.shape[0]] = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        row = np.arange(q.shape[0])[:, None]
        order = np.argsort(d2[row, out[start : start + q.shape[0]]], axis=1)
        out[start : start + q.shape[0]] = out[start : start + q.shape[0]][row, order]
    return out


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = 0
    for i in range(pred.shape[0]):
        hits += len(set(pred[i, :k].tolist()) & set(truth[i, :k].tolist()))
    return hits / float(pred.shape[0] * k)


def has_sift_base(data_dir: Path) -> bool:
    return (data_dir / "sift_base.bin").is_file() or (data_dir / "sift_base.fvecs").is_file()


def resolve_data_dir(cli_path: Path | None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    if cli_path is not None:
        candidates.append(cli_path)
    env = os.environ.get("SIFT_DIR", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(repo_root.parent / "sift1M")
    candidates.append(repo_root.parent / "PageANN" / "sift1M")

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.expanduser()
        key = resolved.resolve() if resolved.exists() else resolved
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)

    for path in unique:
        if path.is_dir() and has_sift_base(path):
            return path

    lines = [
        "Could not find SIFT base vectors (sift_base.bin or sift_base.fvecs).",
        "Tried:",
    ]
    for path in unique:
        lines.append(f"  {path}  exists={path.is_dir()}  has_base={has_sift_base(path) if path.is_dir() else False}")
    lines.append("Download into a sibling sift1M/ folder, or pass --data-dir explicitly.")
    raise SystemExit("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIFT DiskANN Recall@k via Chroma PersistentClient")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with sift_base.bin/.fvecs and sift_query.bin/.fvecs",
    )
    parser.add_argument("--max-points", type=int, default=10_000, help="Crop of sift_base (default 10000)")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries (default 100)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--search-list-size", type=int, default=64)
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(os.environ.get("PERSIST_DIR", "./chroma-sift-testdata")),
    )
    parser.add_argument("--collection", default="sift_diskann")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    if args.max_points < 256:
        raise SystemExit("--max-points must be >= 256 so native DiskANN is built")

    print(f"data dir:     {args.data_dir.resolve()}")
    print(f"max points:   {args.max_points}")
    print(f"n queries:    {args.n_queries}  k={args.k}")

    t0 = time.perf_counter()
    base = load_vectors(args.data_dir, "base", args.max_points)
    queries = load_vectors(args.data_dir, "query", args.n_queries)
    if base.shape[1] != queries.shape[1]:
        raise SystemExit(f"dim mismatch base={base.shape} query={queries.shape}")
    print(f"loaded base {base.shape} query {queries.shape} in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    gt = exact_knn(base, queries, args.k)
    print(f"exact GT on crop in {time.perf_counter() - t0:.1f}s")

    persist_dir = args.persist_dir.resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if args.collection in [c.name for c in client.list_collections()]:
        client.delete_collection(args.collection)

    col = client.create_collection(
        name=args.collection,
        configuration={
            "diskann": {
                "space": "l2",
                "graph_degree": args.graph_degree,
                "search_list_size": args.search_list_size,
            }
        },
        embedding_function=None,
    )
    print("collection configuration:", col.configuration)

    t0 = time.perf_counter()
    for start in range(0, base.shape[0], args.batch_size):
        stop = min(start + args.batch_size, base.shape[0])
        ids = [str(i) for i in range(start, stop)]
        col.upsert(ids=ids, embeddings=base[start:stop].tolist())
        print(f"  upserted [{start}, {stop})")
    print(f"ingest {base.shape[0]} vectors in {time.perf_counter() - t0:.1f}s")

    native = [p for p in persist_dir.rglob("native") if p.is_dir()]
    print("native dirs:", native)
    if not native:
        raise SystemExit("FAIL: expected native/ after ingest (N >= 256)")

    t0 = time.perf_counter()
    pred = np.empty((queries.shape[0], args.k), dtype=np.int64)
    q_batch = 32
    for start in range(0, queries.shape[0], q_batch):
        stop = min(start + q_batch, queries.shape[0])
        result = col.query(
            query_embeddings=queries[start:stop].tolist(),
            n_results=args.k,
        )
        for i, ids in enumerate(result["ids"]):
            pred[start + i] = np.array([int(x) for x in ids], dtype=np.int64)
    query_s = time.perf_counter() - t0
    rec = recall_at_k(pred, gt, args.k)
    qps = queries.shape[0] / query_s if query_s > 0 else 0.0
    print(f"query {queries.shape[0]} in {query_s:.2f}s  ({qps:.1f} qps)")
    print(f"Recall@{args.k} = {rec:.4f}")
    print("OK")


if __name__ == "__main__":
    main()
