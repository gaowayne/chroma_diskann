# Experimental Local DiskANN Backend

This backend integrates the local Rust DiskANN3 `diskann-disk` implementation
with Chroma's Python SegmentAPI. It is experimental, not a production-ready
replacement for Chroma's default Rust backend. The default client still uses
Rust/HNSW. Existing collections are not converted.

## Scope

- Persistent, single-process, dense-vector collections.
- Squared L2 and cosine distance. Inner product is rejected.
- Add, update, upsert, delete, get, query, and restart recovery.
- Chroma metadata/document filtering with an exact filtered vector scan.
- Immutable SSD graph snapshots, in-memory PQ, and a durable SQLite delta.
- Explicit status and synchronous snapshot rebuilding.

Distributed workers, HttpClient, custom schemas, online graph mutation, and
background compaction are not implemented. Use a dedicated data directory and
reopen it with this client, not the default Rust client. A file lock prevents
multiple active writers to the same vector segment.

## Build

The optional binding has its own Cargo workspace and does not add DiskANN
dependencies to Chroma's default build. Cargo fetches Microsoft's DiskANN at
the pinned revision `bceaf45dc6aa694485553816b448d7b8f55df393`, matching the local
source used for this implementation. Rust `1.97.1` and the binding's Cargo.lock
are committed. No external sibling checkout or machine-specific path is needed.
The resulting extension is a normal platform-specific Python wheel and does not
require the source at runtime.

Recommended setup is Linux with a native compiler, Python 3.11, uv, and rustup.
Run from the Chroma project root, using a fresh platform-specific environment:

```bash
rustup toolchain install 1.97.1 --profile minimal
uv venv .venv-diskann --python 3.11
source .venv-diskann/bin/activate
uv pip install -r pyproject.toml --extra dev --extra diskann maturin pytest hypothesis pytest-asyncio pytest-timeout
cd rust/diskann_bindings
maturin develop --release --locked
cd ../..
python -m examples.diskann.basic --path ./diskann-demo-data
```

## Linux Server Validation

From your local terminal, log in using your existing SSH key or enter the SSH
password directly in that terminal, never in a chat message:

```bash
ssh root@10.74.40.88
```

On `salab-hpedl380g11-04`, clone under the requested workspace directory. This
creates a `chroma_diskann` child directory and does not replace existing files:

```bash
mkdir -p /mnt/ps1010/wayne/vectordbstudy/chromadiskann
cd /mnt/ps1010/wayne/vectordbstudy/chromadiskann
git clone --branch feat/chroma-diskann-local-backend --single-branch https://github.com/gaowayne/chroma_diskann.git
cd chroma_diskann
bash bin/validate_diskann_linux.sh --install-system-deps
```

If the clone already exists, do not delete it. Check `git status`, select the
branch, and update with `git pull --ff-only` before rerunning the script.
The runner supports root package setup on apt-get/dnf systems. Without
`--install-system-deps`, it only checks existing system prerequisites. It then
installs Rust/uv when needed, creates `.venv-diskann`, builds the pinned extension,
requires all native tests, and runs separate L2/cosine demos. Downloads require
access to GitHub, crates.io, Rust distribution servers, and PyPI.

The current suite has 20 cases; a complete native run should not skip the two
`real_native` cases. Only a fully successful run creates a `SUCCESS` file.
Each run saves `run.log`, `test-results.xml`, package versions, and demo data in
a new `.diskann-validation/run-*` directory. On failure, inspect the first error
in `run.log`; do not bypass `CHROMA_DISKANN_REQUIRE_NATIVE=1` to obtain a pass.

For a long SSH session, run the script inside your existing `tmux`/`screen`
session so a network disconnect does not interrupt compilation. Build concurrency
defaults to four jobs and can be changed with `CARGO_BUILD_JOBS`.

On Windows, use the environment's `Scripts/Activate.ps1` and install the MSVC
C++ tools. Windows must permit execution of locally built Cargo build scripts.
An `Access is denied` error at `build-script-build.exe` is an environment
blocker; do not treat skipped native tests as a successful native build.

## Client

```python
from chromadb.config import Settings
from chromadb.experimental.diskann import PersistentClient, index_status, rebuild

client = PersistentClient("./diskann-data", Settings(anonymized_telemetry=False))
collection = client.create_collection(
    "documents",
    embedding_function=None,
    metadata={"diskann:space": "cosine", "diskann:rebuild_threshold": 1000},
)
collection.add(ids=["first", "second"], embeddings=[[1, 0], [0, 1]])
print(collection.query(query_embeddings=[[1, 0]], n_results=2))
print(index_status(collection))
print(rebuild(collection))
client.close()
```

Two vectors use the exact path: PQ training requires at least 256 vectors and
more vectors than the graph degree. The native extension is nevertheless
required at startup, so a missing binding never silently becomes a different
ANN implementation. The full example generates 512 vectors and exercises the
real disk snapshot path, filtering, mutation, and reopening.

## Configuration

Parameters are immutable for a collection. Create a new collection to change
them; updating ordinary collection metadata preserves the reserved parameters
on the server. Refetch the collection to inspect server-preserved metadata after
`modify`, since the Python collection object caches the submitted metadata.
The legacy configuration envelope carries Chroma's distance/embedding-function
metadata, but the persisted DiskANN segment type chooses the actual backend.
HNSW tuning parameters are rejected, not silently applied to DiskANN.

| Metadata parameter | Default | Purpose |
| --- | --- | --- |
| `diskann:space` | `l2` | `l2` or `cosine` |
| `diskann:graph_degree` | 32 | Vamana graph degree |
| `diskann:build_search_list_size` | 64 | Build candidate size, at least graph degree |
| `diskann:search_list_size` | 64 | Query candidate size |
| `diskann:beam_width` | 4 | Disk search beam width, 1 to 128 |
| `diskann:rebuild_threshold` | 1000 | Distinct changed IDs before synchronous rebuild |
| `diskann:pq_bytes` | 8 | PQ bytes, capped at vector dimension |
| `diskann:num_threads` | 4 | Native build threads |
| `diskann:alpha` | 1.2 | Vamana pruning alpha, at least 1 |

The native builder currently uses a 1 GiB build-budget parameter. This is not
a total process RSS guarantee. PQ memory grows with the snapshot size, and raw
vectors are retained in SQLite for retrieval, reranking, and rebuilding.

## Persistence And Search

1. Consume Chroma log records and transactionally update vectors, their revision
   numbers, dirty IDs, and the consumed sequence ID in the segment SQLite DB.
2. Build each new DiskANN snapshot in a fresh generation directory. Stream raw
   vectors in ID order; normalize a separate copy for cosine indexing.
3. Sync files and load the new reader before atomically publishing its generation
   and ordinal-to-ID/revision mapping in SQLite. Only then discard obsolete files.
4. Query the disk graph and reject candidates whose vector revisions changed or
   were deleted. Merge with exact delta candidates and rerank original vectors.
5. Use exact scanning for small collections, metadata/document filters, or when
   too few live graph candidates survive. A missing/unreadable snapshot on reopen
   also uses exact recovery and reports `last_error`; explicit rebuild repairs it.

Rebuilds hold the segment lock and block queries/writes. A failed automatic build
does not roll back already committed data: the existing snapshot plus delta
remain usable, and status reports the failure. Crash-before-publication leaves
an unreferenced generation; later successful rebuilds clean it up.

`index_status` reports the available snapshot mode, live count, snapshot count,
delta count, generation, and last build/load error. A `diskann+delta` status does
not mean filtered queries use ANN: filtered queries always use exact scans in
this version. No performance or recall benchmark is claimed.

## Verification

```bash
python -m pytest chromadb/test/segment/test_diskann.py -q
CHROMA_DISKANN_REQUIRE_NATIVE=1 python -m pytest chromadb/test/segment/test_diskann.py -q -k real_native
```

Lifecycle tests use an explicitly named test double to isolate transactional
behavior. The two `real_native` tests build real indexes for L2 and cosine,
query them through Chroma, and check mutation plus restart behavior. They skip
without the binding unless `CHROMA_DISKANN_REQUIRE_NATIVE=1`, which makes absence
an error. Run this required-native gate before treating the backend as usable.