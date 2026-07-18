#!/usr/bin/env bash

set -euo pipefail

readonly WHISPER_CPP_REPOSITORY="https://github.com/ggml-org/whisper.cpp.git"
readonly WHISPER_CPP_COMMIT="5cad7abb1e229a27c00730c1882305d282ee6941"

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv"
jobs="${JOBS:-$(nproc)}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

for command_name in awk cmake git hipcc install ldd nproc rocminfo sed sha256sum sort; do
    require_command "${command_name}"
done

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "Project environment not found: ${venv_dir}" >&2
    echo "Run uv sync before installing whisper-cli." >&2
    exit 1
fi

if [[ ! "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer: ${jobs}" >&2
    exit 1
fi

if ! rocminfo_output="$(rocminfo 2>&1)"; then
    echo "rocminfo failed:" >&2
    echo "${rocminfo_output}" >&2
    exit 1
fi

mapfile -t amdgpu_targets < <(
    awk '
        $1 == "Name:" && $2 ~ /^gfx[0-9]+[[:alnum:]]*$/ && $2 != "gfx000" {
            print $2
        }
    ' <<<"${rocminfo_output}" | sort -u
)
if ((${#amdgpu_targets[@]} == 0)); then
    echo "rocminfo did not report an AMD GPU target." >&2
    exit 1
fi
amdgpu_targets_cmake="$(IFS=';'; echo "${amdgpu_targets[*]}")"
echo "Detected AMD GPU target(s): ${amdgpu_targets_cmake}"

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/simple-radeon-subs-whisper.XXXXXX")"
source_dir="${work_dir}/source"
build_dir="${work_dir}/build"
binary_tmp=""
manifest_tmp=""

cleanup() {
    rm -rf -- "${work_dir}"
    [[ -z "${binary_tmp}" ]] || rm -f -- "${binary_tmp}"
    [[ -z "${manifest_tmp}" ]] || rm -f -- "${manifest_tmp}"
}
trap cleanup EXIT

echo "Fetching whisper.cpp ${WHISPER_CPP_COMMIT}"
git init --quiet "${source_dir}"
git -C "${source_dir}" remote add origin "${WHISPER_CPP_REPOSITORY}"
git -C "${source_dir}" fetch --quiet --depth=1 origin "${WHISPER_CPP_COMMIT}"
git -C "${source_dir}" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD

actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${WHISPER_CPP_COMMIT}" ]]; then
    echo "Unexpected whisper.cpp commit: ${actual_commit}" >&2
    exit 1
fi

cmake \
    -S "${source_dir}" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${amdgpu_targets_cmake}" \
    -DWHISPER_BUILD_EXAMPLES=ON

cmake --build "${build_dir}" --target whisper-cli --parallel "${jobs}"

built_binary="${build_dir}/bin/whisper-cli"
if [[ ! -x "${built_binary}" ]]; then
    echo "Build completed without producing ${built_binary}" >&2
    exit 1
fi

mkdir -p "${venv_dir}/bin"
binary_tmp="$(mktemp "${venv_dir}/bin/.whisper-cli.XXXXXX")"
install -m 0755 "${built_binary}" "${binary_tmp}"

ldd_output="$(ldd "${binary_tmp}")"
if grep -q "not found" <<<"${ldd_output}"; then
    echo "whisper-cli has unresolved shared libraries:" >&2
    echo "${ldd_output}" >&2
    exit 1
fi
if grep -Eq 'lib(whisper|ggml)' <<<"${ldd_output}"; then
    echo "whisper-cli unexpectedly depends on shared whisper.cpp libraries:" >&2
    echo "${ldd_output}" >&2
    exit 1
fi

"${binary_tmp}" --version >/dev/null

binary_sha256="$(sha256sum "${binary_tmp}" | awk '{print $1}')"
cmake_version="$(cmake --version | sed -n '1p')"
hipcc_version="$(hipcc --version 2>&1 | sed -n '1p')"
manifest_path="${venv_dir}/.whisper-cli-build.json"
manifest_tmp="$(mktemp "${venv_dir}/.whisper-cli-build.XXXXXX")"

MANIFEST_REPOSITORY="${WHISPER_CPP_REPOSITORY}" \
MANIFEST_COMMIT="${WHISPER_CPP_COMMIT}" \
MANIFEST_AMDGPU_TARGETS="${amdgpu_targets_cmake}" \
MANIFEST_BINARY_SHA256="${binary_sha256}" \
MANIFEST_CMAKE_VERSION="${cmake_version}" \
MANIFEST_HIPCC_VERSION="${hipcc_version}" \
"${venv_dir}/bin/python" - <<'PY' >"${manifest_tmp}"
import json
import os

keys = (
    "REPOSITORY",
    "COMMIT",
    "AMDGPU_TARGETS",
    "BINARY_SHA256",
    "CMAKE_VERSION",
    "HIPCC_VERSION",
)
print(
    json.dumps(
        {key.lower(): os.environ[f"MANIFEST_{key}"] for key in keys},
        indent=2,
    )
)
PY

mv -f -- "${binary_tmp}" "${venv_dir}/bin/whisper-cli"
binary_tmp=""
mv -f -- "${manifest_tmp}" "${manifest_path}"
manifest_tmp=""

echo "Installed ${venv_dir}/bin/whisper-cli"
echo "Recorded ${manifest_path}"
