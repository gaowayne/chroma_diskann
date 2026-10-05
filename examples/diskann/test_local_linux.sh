#!/usr/bin/env bash
# Build local Chroma Python bindings (with DiskANN) and run the PersistentClient smoke test.
set -euo pipefail

CHROMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# rust/diskann/Cargo.toml expects: <parent-of-codecommit1>/DiskANN
DISKANN_ROOT="$(cd "${CHROMA_ROOT}/../../DiskANN" 2>/dev/null && pwd || true)"
PERSIST_DIR="${PERSIST_DIR:-${CHROMA_ROOT}/chroma-diskann-testdata}"

echo "chroma root:  ${CHROMA_ROOT}"
echo "DiskANN root: ${DISKANN_ROOT:-MISSING}"
echo "persist dir:  ${PERSIST_DIR}"

if [[ ! -f "${DISKANN_ROOT}/diskann/Cargo.toml" ]]; then
  echo "DiskANN rust crates not found at ${CHROMA_ROOT}/../../DiskANN"
  echo "On the Ubuntu machine use this layout:"
  echo "  ~/vectorsearch/DiskANN"
  echo "  ~/vectorsearch/codecommit1/chroma_diskann"
  exit 1
fi

if ! command -v rustc >/dev/null; then
  echo "Install rustup first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  exit 1
fi

cd "${CHROMA_ROOT}"
rustup show

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements_dev.txt
python -m pip install -e .

export PERSIST_DIR
python examples/diskann/local_persistent.py
echo "Inspect native files with: find ${PERSIST_DIR} -path '*/native/*' -print"
