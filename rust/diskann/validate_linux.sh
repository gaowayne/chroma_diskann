#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    printf 'This validation must run on Linux, not Windows.\n' >&2
    exit 2
fi
if [[ $# -ne 0 ]]; then
    printf 'Usage: bash validate_linux.sh\n' >&2
    exit 2
fi
for tool in cargo rustc rustup tee mktemp; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing prerequisite: %s\n' "$tool" >&2
        exit 2
    fi
done

crate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$crate_dir"
manifest="$crate_dir/Cargo.toml"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$crate_dir/target}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
mkdir -p -- "$crate_dir/.validation"
run_dir="$(mktemp -d "$crate_dir/.validation/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
stage="environment"
trap 'status=$?; printf "FAILED stage=%s exit=%s logs=%s\n" "$stage" "$status" "$run_dir" >&2; exit "$status"' ERR

run_step() {
    stage="$1"
    shift
    printf '\nSTAGE %s\n' "$stage"
    "$@" 2>&1 | tee "$run_dir/$stage.log"
}

environment_info() {
    uname -a
    rustc +1.97.1 -Vv
    cargo +1.97.1 -V
    printf 'crate=%s\ntarget=%s\n' "$crate_dir" "$CARGO_TARGET_DIR"
}

printf 'Validation artifacts: %s\n' "$run_dir"
run_step environment environment_info
run_step library-compile cargo +1.97.1 test --manifest-path "$manifest" --locked --lib --no-run
run_step unit-tests cargo +1.97.1 test --manifest-path "$manifest" --locked --lib -- --nocapture --test-threads=1
run_step smoke-compile cargo +1.97.1 build --manifest-path "$manifest" --locked --example smoke
for metric in l2 cosine; do
    run_step "build-$metric" cargo +1.97.1 run --manifest-path "$manifest" --locked --example smoke -- build "$run_dir/$metric" "$metric"
    run_step "query-$metric" cargo +1.97.1 run --manifest-path "$manifest" --locked --example smoke -- query "$run_dir/$metric"
done
printf 'All native validation stages passed.\n' | tee "$run_dir/SUCCESS"
printf 'PASS logs=%s\n' "$run_dir"