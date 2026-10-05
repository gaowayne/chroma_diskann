#!/usr/bin/env bash
# Build local Chroma Python bindings (with DiskANN) and run the PersistentClient smoke test.
set -euo pipefail

CHROMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT_DIR="$(cd "${CHROMA_ROOT}/.." && pwd)"
SIBLING="${PARENT_DIR}/DiskANN"
PERSIST_DIR="${PERSIST_DIR:-${CHROMA_ROOT}/chroma-diskann-testdata}"

resolve_diskann() {
  local candidate
  for candidate in \
    "${DISKANN_ROOT:-}" \
    "${SIBLING}" \
    "${PARENT_DIR}/../DiskANN" \
    "${CHROMA_ROOT}/DiskANN"
  do
    if [[ -n "${candidate}" && -f "${candidate}/diskann/Cargo.toml" ]]; then
      cd "${candidate}" && pwd
      return 0
    fi
  done
  return 1
}

DISKANN_FOUND="$(resolve_diskann || true)"

echo "chroma root:  ${CHROMA_ROOT}"
echo "DiskANN root: ${DISKANN_FOUND:-MISSING}"
echo "persist dir:  ${PERSIST_DIR}"

if [[ -z "${DISKANN_FOUND}" ]]; then
  echo
  echo "DiskANN rust crates were not found."
  echo "Cargo expects DiskANN next to chroma_diskann:"
  echo "  ${SIBLING}"
  echo
  echo "If DiskANN is already on this machine:"
  echo "  export DISKANN_ROOT=/absolute/path/to/DiskANN"
  echo "  ln -sfn \"\$DISKANN_ROOT\" \"${SIBLING}\""
  echo
  echo "Or clone it:"
  echo "  git clone https://github.com/microsoft/DiskANN.git ${SIBLING}"
  exit 1
fi

# rust/diskann/Cargo.toml uses ../../../DiskANN from rust/diskann/
if [[ "${DISKANN_FOUND}" != "${SIBLING}" ]]; then
  echo "Linking ${SIBLING} -> ${DISKANN_FOUND}"
  ln -sfn "${DISKANN_FOUND}" "${SIBLING}"
fi

if ! command -v rustc >/dev/null; then
  echo "Install rustup first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  exit 1
fi

cd "${CHROMA_ROOT}"
rustup show

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements_dev.txt
python -m pip install -e .

export PERSIST_DIR
python examples/diskann/local_persistent.py
echo "Inspect native files with: find ${PERSIST_DIR} -path '*/native/*' -print"
