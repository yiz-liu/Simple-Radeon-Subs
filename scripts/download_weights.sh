#!/usr/bin/env bash

set -euo pipefail

readonly WHISPER_REPOSITORY="timeless/whispercpp"
readonly WHISPER_REVISION="master"
readonly WHISPER_FILENAME="ggml-large-v3-turbo.bin"
readonly WHISPER_SHA256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"

readonly VAD_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
readonly VAD_FILENAME="ggml-silero-v6.2.0.bin"
readonly VAD_SHA256="2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${project_root}/models/whisper"
modelscope_cli="${project_root}/.venv/bin/ms"
work_dir=""

cleanup() {
    [[ -z "${work_dir}" ]] || rm -rf -- "${work_dir}"
}
trap cleanup EXIT

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

verify_existing_file() {
    local path="$1"
    local expected_sha256="$2"

    if [[ ! -e "${path}" ]]; then
        return
    fi
    if [[ ! -f "${path}" ]]; then
        echo "Expected a regular file: ${path}" >&2
        exit 1
    fi

    local actual_sha256
    actual_sha256="$(sha256 "${path}")"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "Refusing to overwrite a model with an unexpected checksum: ${path}" >&2
        echo "Expected SHA256: ${expected_sha256}" >&2
        echo "Observed SHA256: ${actual_sha256}" >&2
        exit 1
    fi

    echo "Already downloaded: ${path}"
}

install_verified_file() {
    local source_path="$1"
    local destination_path="$2"
    local expected_sha256="$3"

    local actual_sha256
    actual_sha256="$(sha256 "${source_path}")"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "Downloaded model failed checksum verification: ${source_path}" >&2
        echo "Expected SHA256: ${expected_sha256}" >&2
        echo "Observed SHA256: ${actual_sha256}" >&2
        exit 1
    fi

    chmod 0644 "${source_path}"
    mv -- "${source_path}" "${destination_path}"
    echo "Installed ${destination_path}"
}

for command_name in awk curl mktemp mv sha256sum; do
    require_command "${command_name}"
done

if [[ ! -x "${modelscope_cli}" ]]; then
    echo "ModelScope CLI not found: ${modelscope_cli}" >&2
    echo "Run uv sync before downloading model weights." >&2
    exit 1
fi

mkdir -p "${model_dir}"

whisper_path="${model_dir}/${WHISPER_FILENAME}"
vad_path="${model_dir}/${VAD_FILENAME}"
verify_existing_file "${whisper_path}" "${WHISPER_SHA256}"
verify_existing_file "${vad_path}" "${VAD_SHA256}"

if [[ -f "${whisper_path}" && -f "${vad_path}" ]]; then
    echo "Whisper and VAD weights are ready."
    exit 0
fi

work_dir="$(mktemp -d "${model_dir}/.download.XXXXXX")"

if [[ ! -f "${whisper_path}" ]]; then
    whisper_download_dir="${work_dir}/whisper"
    mkdir -p "${whisper_download_dir}"
    echo "Downloading ${WHISPER_FILENAME} from ModelScope (${WHISPER_REPOSITORY})"
    "${modelscope_cli}" download \
        "${WHISPER_REPOSITORY}" \
        "${WHISPER_FILENAME}" \
        --revision "${WHISPER_REVISION}" \
        --local-dir "${whisper_download_dir}"
    install_verified_file \
        "${whisper_download_dir}/${WHISPER_FILENAME}" \
        "${whisper_path}" \
        "${WHISPER_SHA256}"
fi

if [[ ! -f "${vad_path}" ]]; then
    vad_download_path="${work_dir}/${VAD_FILENAME}"
    echo "Downloading ${VAD_FILENAME} from the official whisper.cpp VAD repository"
    curl \
        --fail \
        --location \
        --retry 3 \
        --retry-all-errors \
        --show-error \
        --output "${vad_download_path}" \
        "${VAD_URL}"
    install_verified_file "${vad_download_path}" "${vad_path}" "${VAD_SHA256}"
fi

echo "Whisper and VAD weights are ready in ${model_dir}."
