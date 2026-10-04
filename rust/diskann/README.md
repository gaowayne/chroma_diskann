# DiskANN Rust adapter

An independent Rust crate for building and reading immutable `diskann-disk`
indexes. It does not modify, wrap, or depend on Chroma's HNSW implementation.
It is deliberately outside Chroma's default Cargo workspace and has its own
lockfile. It is not yet selectable through the Chroma collection API.

## Implemented

- `build_index`: builds a real disk graph and PQ files using the pinned
  Microsoft DiskANN revision in `Cargo.toml`.
- `DiskAnnIndex::open`: validates the manifest, original-vector file and native
  graph header, then opens the native disk reader.
- `DiskAnnIndex::search`: calls native graph search, returning only
  `result_count` valid neighbors. There is no HNSW or exact-scan fallback.
- `DiskAnnIndex::get_vector`: reads the original float32 vector by row ID.
- Squared L2 and cosine distance. Cosine uses normalized vectors in the native
  L2 index and converts squared L2 to `1 - cosine`; original vectors are retained.

## Contract and limits

IDs are zero-based input row numbers, not Chroma user IDs. The build API accepts
the complete vector batch in memory. PQ training requires at least 256 vectors;
smaller batches are rejected rather than silently routed to another algorithm.
PQ bytes are capped at the vector dimensionality. Non-finite vectors and zero
cosine vectors are rejected.

All APIs are blocking. Create, use and drop readers on a blocking worker, not
inside a Tokio async task: the native implementation owns a runtime. Searches
on one reader are serialized. Dropping the reader releases its native resources.

A build requires a new output directory and never overwrites an existing one.
Native files are synchronized before the manifest is published. A failed build
may leave a partial directory without a valid manifest; it is not a usable index.
Do not mutate index files while readers are open.

Updates, deletion, metadata filters, Chroma ID mapping, log checkpoints, snapshot
replacement and Chroma API routing are not implemented in this crate yet.

## Linux verification

The integration has three gates:

1. Build and validate the real native index on Linux. The tooling for this gate
  is provided here; execution is still pending.
2. Implement DiskANN-only user-ID mapping, filtering, mutable data and durable
  log/checkpoint handling, with recovery tests.
3. Add an explicit Chroma entry point without rewriting the existing HNSW path,
  then run collection-level end-to-end tests.

Passing gate 1 does not implement gates 2 or 3.

### 1. Prepare the Linux checkout

Use host `10.74.40.88`. The checkout must contain the current crate, including
`Cargo.lock`, `examples/smoke.rs` and `validate_linux.sh`; pulling an older branch
does not transfer uncommitted local files. Exclude `target/` and `.validation/`
when synchronizing the source.

Install Rust/rustup and any required Linux compiler/linker prerequisites. Install
`gdb` for interactive debugging. From the Chroma repository root:

```bash
rustup toolchain install 1.97.1 --profile minimal --component rustfmt
bash rust/diskann/validate_linux.sh
```

The script refuses to run outside Linux. It does not install packages, change
security settings, or build the main Chroma/HNSW workspace.

### 2. Read the first failing stage

Every run has a unique directory under `.validation/`, with one log per stage:

| Stage | Gate |
| --- | --- |
| `library-compile` | Compile the library and its tests, without running them. |
| `unit-tests` | Run the four native/input/filesystem tests serially. |
| `smoke-compile` | Build the standalone debug executable. |
| `build-l2`, `build-cosine` | Build real 512-vector, 16-dimensional disk indexes. |
| `query-l2`, `query-cosine` | Reopen in a separate process and verify original vectors and query results. |

Only a completed run contains `SUCCESS`. Failures stop the script and preserve
logs and partial indexes. Do not erase the failed run before investigating it.

The query smoke test uses 16 independent queries, checks returned distances,
unique IDs and sorting, and requires mean recall@10 of at least 0.90. Exact
distances are used only as a test oracle; results still come from native DiskANN.
This threshold is a small deterministic smoke gate, not a production recall or
performance claim. Debug builds must not be used for performance conclusions.

If compilation fails, inspect that log before attempting any native debugging.
If native I/O fails, check the host/container policy: restrictions on `io_uring`
can prevent disk reads. Do not silently replace the native backend to pass tests.

### 3. Reproduce individual stages

From the repository root, use a fresh output directory for every build:

```bash
export RUST_BACKTRACE=1
cargo +1.97.1 test --manifest-path rust/diskann/Cargo.toml --locked --lib --no-run
cargo +1.97.1 test --manifest-path rust/diskann/Cargo.toml --locked --lib native_build_reopen_and_query -- --nocapture --test-threads=1
cargo +1.97.1 run --manifest-path rust/diskann/Cargo.toml --locked --example smoke -- build /tmp/diskann-debug-l2 l2
cargo +1.97.1 run --manifest-path rust/diskann/Cargo.toml --locked --example smoke -- query /tmp/diskann-debug-l2
```

Use a different build directory if `/tmp/diskann-debug-l2` already exists. A
query command can be repeated against an unchanged index. Replace `l2` with
`cosine` and use a separate directory for the cosine build.

### 4. Debug with GDB

Use the crate directory and the default Linux host target. All binaries below
are unoptimized debug builds with symbols:

```bash
cd rust/diskann
export CARGO_TARGET_DIR="$PWD/target"
export RUST_BACKTRACE=1
cargo +1.97.1 build --locked --example smoke
mkdir -p .validation
debug_dir="$(mktemp -d "$PWD/.validation/debug-XXXXXX")"
rustup run 1.97.1 rust-gdb --args "$CARGO_TARGET_DIR/debug/examples/smoke" build "$debug_dir/l2" l2
```

At the GDB prompt:

```text
set breakpoint pending on
rbreak chroma_diskann::build_index
run
bt
info locals
next
continue
quit
```

Let the build finish before debugging a query against its output:

```bash
rustup run 1.97.1 rust-gdb --args "$CARGO_TARGET_DIR/debug/examples/smoke" query "$debug_dir/l2"
```

```text
set breakpoint pending on
rbreak chroma_diskann::DiskAnnIndex::open
rbreak chroma_diskann::DiskAnnIndex::search
run
bt
info locals
continue
```

Start at the wrapper boundary to identify input, persistence or search failures;
step into the pinned Microsoft implementation only after locating the phase.

Tests build real 512-vector L2 and cosine indexes, check queries against exact
distance calculations, reopen files, read original vectors, check invalid input,
and verify existing directories are not overwritten. They are not ignored and
do not substitute a fake index. Linux compilation and execution are still pending;
only dependency resolution and source-level checks have been performed.