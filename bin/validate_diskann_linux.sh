#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash bin/validate_diskann_linux.sh [--install-system-deps]' \
        'Build the pinned DiskANN extension, require native tests, and run both demos.' \
        '--install-system-deps installs build prerequisites with apt-get or dnf (root only).' \
        'Environment: CARGO_BUILD_JOBS (default 4), CHROMA_DISKANN_RUNS_DIR (log root).'
}

install_system_deps=false
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --install-system-deps) install_system_deps=true ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux ]]; then
    printf 'This validation runner requires Linux.\n' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_root="${CHROMA_DISKANN_RUNS_DIR:-${repo_root}/.diskann-validation}"
mkdir -p "$runs_root"
run_dir="$(mktemp -d "${runs_root}/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee -a "${run_dir}/run.log") 2>&1
trap 'result=$?; printf "FAILED at line %s (exit %s). Log: %s/run.log\n" "$LINENO" "$result" "$run_dir"; exit "$result"' ERR

printf 'Validation directory: %s\n' "$run_dir"
printf '\n[1/7] System prerequisites\n'
uname -srm
if "$install_system_deps"; then
    if (( EUID != 0 )); then
        printf 'Run package installation as root, or provision prerequisites separately.\n' >&2
        exit 2
    fi
    if command -v apt-get >/dev/null; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y \
            build-essential cmake pkg-config python3-dev libssl-dev git curl ca-certificates
    elif command -v dnf >/dev/null; then
        dnf install -y gcc gcc-c++ make cmake pkgconf-pkg-config \
            python3-devel openssl-devel git curl ca-certificates
    else
        printf 'Unsupported package manager. Install a C/C++ compiler, Python headers, OpenSSL headers, pkg-config, git, and curl.\n' >&2
        exit 2
    fi
fi
for tool in git curl cc c++ pkg-config; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Missing prerequisite: %s\n' "$tool" >&2
        exit 2
    fi
done
git -C "$repo_root" log -1 --format='%H %s'

printf '\n[2/7] Rust 1.97.1 and uv\n'
export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
if ! command -v rustup >/dev/null; then
    curl --fail --location --proto '=https' --tlsv1.2 \
        https://sh.rustup.rs --output "${run_dir}/rustup-init.sh"
    sh "${run_dir}/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain none
fi
rustup toolchain install 1.97.1 --profile minimal
if ! command -v uv >/dev/null; then
    curl --fail --location --proto '=https' --tlsv1.2 \
        https://astral.sh/uv/install.sh --output "${run_dir}/uv-install.sh"
    UV_NO_MODIFY_PATH=1 sh "${run_dir}/uv-install.sh"
fi
uv --version
rustc +1.97.1 --version

printf '\n[3/7] Isolated Python 3.11 environment\n'
cd "$repo_root"
venv="${repo_root}/.venv-diskann"
if [[ ! -x "${venv}/bin/python" ]]; then
    if [[ -e "$venv" ]]; then
        printf 'Refusing to replace existing invalid environment: %s\n' "$venv" >&2
        exit 2
    fi
    uv venv "$venv" --python 3.11
fi
export VIRTUAL_ENV="$venv"
export PATH="${venv}/bin:${PATH}"
uv pip install --python "${venv}/bin/python" -r pyproject.toml \
    --extra dev --extra diskann 'maturin>=1.8,<2' \
    pytest hypothesis pytest-asyncio pytest-timeout
uv pip freeze --python "${venv}/bin/python" > "${run_dir}/python-packages.txt"

printf '\n[4/7] Build real DiskANN native extension\n'
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
cd "${repo_root}/rust/diskann_bindings"
maturin develop --release --locked
cd "$repo_root"
python -c 'import chromadb, chroma_diskann_native as native; print("Chroma:", chromadb.__file__); print("Native:", native.__file__); assert native.BACKEND == "diskann-disk"; assert callable(native.build_index)'

printf '\n[5/7] Full focused suite with native tests required\n'
CHROMA_DISKANN_REQUIRE_NATIVE=1 python -m pytest \
    chromadb/test/segment/test_diskann.py -q -ra --timeout=300 \
    --junitxml="${run_dir}/test-results.xml"

printf '\n[6/7] Real L2 snapshot, filtering, mutation, and reopen demo\n'
python -m examples.diskann.basic --path "${run_dir}/demo-l2" --space l2

printf '\n[7/7] Real cosine snapshot and reopen demo\n'
python -m examples.diskann.basic --path "${run_dir}/demo-cosine" --space cosine
printf 'PASS\n' > "${run_dir}/SUCCESS"
printf '\nAll steps passed. Results: %s\n' "$run_dir"