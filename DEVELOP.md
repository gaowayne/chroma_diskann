# Development Instructions

This project uses the testing, build and release standards specified
by the PyPA organization and documented at
<https://packaging.python.org>.

## Setup

Set up a virtual environment and install the project's requirements
and dev requirements:

```bash
python3 -m venv venv      # Only need to do this once
source venv/bin/activate  # Do this each time you use a new shell for the project
pip install -r requirements.txt
pip install -r requirements_dev.txt
pre-commit install # install the precommit hooks
```

Install protobuf:
for MacOS `brew install protobuf`

You can also install `chromadb` the `pypi` package locally and in editable mode with `pip install -e .`.

## Single-node Rust/HNSW baseline

Build the original Chroma backend before testing an optional index backend.
This path uses the default Rust bindings and HNSW, not the experimental DiskANN
client, and does not need Docker, Kubernetes, or Tilt.

The following commands assume Ubuntu/Debian on the Linux build server and a
root shell. On other distributions, install equivalent development packages.
Use a separate checkout and data directory so the baseline does not modify the
DiskANN checkout or its collections. The pre-DiskANN baseline in this fork is
`e5c22977da46f9410c2e8f2aea54c45d84d29299`.

```bash
mkdir -p /mnt/ps1010/wayne/vectordbstudy/chromadiskann
cd /mnt/ps1010/wayne/vectordbstudy/chromadiskann
git clone --branch main --single-branch https://github.com/gaowayne/chroma_diskann.git chroma_baseline
cd chroma_baseline
git switch --detach e5c22977da46f9410c2e8f2aea54c45d84d29299
apt-get update
apt-get install -y build-essential cmake pkg-config libssl-dev libclang-dev protobuf-compiler python3-dev python3-venv git curl ca-certificates
```

If `chroma_baseline` already exists, inspect it with `git status` and reuse it
only after confirming its revision; do not delete or overwrite existing work.
Install Rust only when not already available, then create a Linux-native virtual
environment with Python's standard `venv` module. No uv is needed. Use Python 3.9
or later; Python 3.10 through 3.12 are suitable starting points. If `python3`
is older than 3.9, select a supported interpreter before creating the environment.
Reuse an existing compatible Linux virtual environment instead of recreating it.
The Rust CI uses the stable toolchain.

```bash
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/chroma-rustup-init.sh
  sh /tmp/chroma-rustup-init.sh -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install stable --profile minimal
rustup override set stable
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'maturin>=1.8,<2' build
python -m pip install -r requirements.txt
set -o pipefail
CARGO_BUILD_JOBS=4 python -m maturin develop --release --locked 2>&1 | tee /tmp/chroma-baseline-build.log
```

Run Maturin from the project root: it selects `rust/python_bindings/Cargo.toml`.
`pip -r` reads `requirements.txt`, not `pyproject.toml`; Maturin also installs
the package dependencies declared by the project when installing the build.
Do not build `rust/diskann_bindings`. Verify the compiled bindings, default API,
vector insertion, nearest-neighbor query, and reopening without downloading an
embedding model:

```bash
python - <<'PY'
import chromadb
import chromadb_rust_bindings
from chromadb.config import Settings

settings = Settings(anonymized_telemetry=False)
assert settings.chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
print("Python package:", chromadb.__file__)
print("Native bindings:", chromadb_rust_bindings.__file__)
client = chromadb.PersistentClient("./baseline-data", settings)
collection = client.get_or_create_collection("baseline-smoke", embedding_function=None)
collection.upsert(ids=["first", "second"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
result = collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)
assert result["ids"] == [["first"]], result
client.close()
client = chromadb.PersistentClient("./baseline-data", settings)
collection = client.get_collection("baseline-smoke", embedding_function=None)
assert collection.count() == 2
assert collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"] == [["first"]]
client.close()
print("BASELINE PASS: default Rust/HNSW query and persistence")
PY
```

Optionally start the server in the foreground, using a different data directory:

```bash
chroma run --path ./baseline-server-data --host 127.0.0.1 --port 8000
```

From another terminal on the same server, verify it with
`curl -fsS http://127.0.0.1:8000/api/v2/heartbeat`. If port 8000 is occupied, use
another free port in both commands. Keep loopback binding unless authentication
and network access controls have been configured. To access it from your local
machine, use an SSH tunnel such as `ssh -L 18000:127.0.0.1:8000 root@10.74.40.88`
and connect to `http://127.0.0.1:18000` locally.

## Local dev setup for distributed chroma

We use tilt for providing local dev setup. Tilt is an open source project

### Requirement

- Docker
- Local Kubernetes cluster (Recommended: [OrbStack](https://orbstack.dev/) for mac, [Kind](https://kind.sigs.k8s.io/) for linux)
- [Tilt](https://docs.tilt.dev/)
- [Helm](https://helm.sh)

1. Start Kubernetes. If you're using OrbStack, navigate to `Kubernetes - Pods`, and select `Turn On`
2. Start a distributed Chroma cluster by running `tilt up` from the root of the repository.
3. Once done, it will expose Chroma on port 8000. You can also visit the Tilt dashboard UI at `http://localhost:10350/`.
4. To clean and remove all the resources created by Tilt, use `tilt down`.

## Testing

Unit tests are in the `/chromadb/test` directory.

To run unit tests using your current environment, run `pytest`.

Make sure to have `tilt up` running for these tests otherwise some distributed Chroma tests will fail.

## Manual Build

Make sure the following is only done in the virtual environment created in the [Setup](#setup) section above.

To manually build the rust codebase and bindings for type safety, run `maturin dev`.

To manually build a distribution, run `python -m build`.

The project's source and wheel distributions will be placed in the `dist` directory.

If you have `tilt up` running, saving changes to your files will automatically rebuild new binaries with your changes and deploy to the local cluster `tilt` has running.

## IDE Recommendations

If you are developing with VSCode or its derivatives (Windsurf/Cursor etc), make sure to install the `rust-analyzer` extension. It helps with auto-formatting, Intellisense and code navigation.

For debugging it is recommended to install the `CodeLLDB` extension.

You should be able to run and debug the rust tests by clicking on the 'Run Test' or 'Debug' button found above the test method definitions.

![rust-analyzer extension](https://github.com/user-attachments/assets/a7779e4d-9d64-4511-9271-b790bed7b68b)

## Setting breakpoints in Distributed Chroma

Debugging binaries in the Kubernetes pods that `tilt up` spins up is a bit more involved. Right now the only reliable way to set a breakpoint in this scenario is to log in to the pod, install lldb/gdb and set a breakpoint that way. For example after running `tilt up` you can set a breakpoint in the query-service-0 pod as follows:

```bash
kubectl exec -it query-service-0 -n chroma -- /bin/sh
apt-get update && apt-get install gdb
gdb
(gdb) b <relative_file_path>:<lineno>
```

## Manual Release

Not yet implemented.

## Versioning

This project uses PyPA's `setuptools_scm` module to determine the
version number for build artifacts, meaning the version number is
derived from Git rather than hardcoded in the repository. For full
details, see the
[documentation for setuptools_scm](https://github.com/pypa/setuptools_scm/).

In brief, version numbers are generated as follows:

- If the current git head is tagged, the version number is exactly the
  tag (e.g, `0.0.1`).
- If the current git head is a clean checkout, but is not tagged,
  the version number is a patch version increment of the most recent
  tag, plus `devN` where N is the number of commits since the most
  recent tag. For example, if there have been 5 commits since the
  `0.0.1` tag, the generated version will be `0.0.2-dev5`.
- If the current head is not a clean checkout, a `+dirty` local
  version will be appended to the version number. For example,
  `0.0.2-dev5+dirty`.

At any point, you can manually run `python -m setuptools_scm` to see
what version would be assigned given your current state.

## Continuous Integration

This project uses Github Actions to run unit tests automatically upon
every commit to the main branch. See the documentation for Github
Actions and the flow definitions in `.github/workflows` for details.

## Continuous Delivery

Not yet implemented.
