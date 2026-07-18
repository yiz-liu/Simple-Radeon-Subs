#!/usr/bin/env bash

set -euo pipefail

readonly WHISPER_REPOSITORY="timeless/whispercpp"
readonly WHISPER_REVISION="master"
readonly WHISPER_FILENAME="ggml-large-v3-turbo.bin"
readonly WHISPER_SHA256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"

readonly VAD_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
readonly VAD_FILENAME="ggml-silero-v6.2.0.bin"
readonly VAD_SHA256="2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"

readonly TRANSLATION_REPOSITORY="cyankiwi/Qwen3.5-9B-AWQ-4bit"
readonly TRANSLATION_REVISION="b7243f9852c1b88094c00565b61a8624c63ff90e"
readonly -a TRANSLATION_FILENAMES=(
    "model-00001-of-00003.safetensors"
    "model-00002-of-00003.safetensors"
    "model-00003-of-00003.safetensors"
    "tokenizer.json"
)
readonly -a TRANSLATION_SHA256S=(
    "13a023add7c4fa37636fd28b15ba80511d9041a331d7a610560b553cac9a0de8"
    "c837545c600ace6064c8009fc7ae917cb0ab9c96747174df813aec1cb28bb61d"
    "b03792cfc8278a228b3b452edb57625b5c6a2a1e77b860d21395d474c03968ea"
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
whisper_model_dir="${project_root}/models/whisper"
translation_model_dir="${project_root}/models/Qwen3.5-9B-AWQ-4bit"
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

verify_translation_model() {
    local model_dir="$1"

    if [[ ! -e "${model_dir}" ]]; then
        return 1
    fi
    if [[ ! -d "${model_dir}" ]]; then
        echo "Expected a model directory: ${model_dir}" >&2
        exit 1
    fi

    local required_file
    for required_file in config.json model.safetensors.index.json tokenizer_config.json; do
        if [[ ! -s "${model_dir}/${required_file}" ]]; then
            echo "Translation model is incomplete: ${model_dir}/${required_file}" >&2
            exit 1
        fi
    done

    local index
    for index in "${!TRANSLATION_FILENAMES[@]}"; do
        local path="${model_dir}/${TRANSLATION_FILENAMES[${index}]}"
        if [[ ! -f "${path}" ]]; then
            echo "Translation model is incomplete: ${path}" >&2
            exit 1
        fi

        local actual_sha256
        actual_sha256="$(sha256 "${path}")"
        if [[ "${actual_sha256}" != "${TRANSLATION_SHA256S[${index}]}" ]]; then
            echo "Translation model failed checksum verification: ${path}" >&2
            echo "Expected SHA256: ${TRANSLATION_SHA256S[${index}]}" >&2
            echo "Observed SHA256: ${actual_sha256}" >&2
            exit 1
        fi
    done
    return 0
}

for command_name in awk curl mktemp mv sha256sum; do
    require_command "${command_name}"
done

if [[ ! -x "${modelscope_cli}" ]]; then
    echo "ModelScope CLI not found: ${modelscope_cli}" >&2
    echo "Run uv sync before downloading model weights." >&2
    exit 1
fi

mkdir -p "${whisper_model_dir}"

whisper_path="${whisper_model_dir}/${WHISPER_FILENAME}"
vad_path="${whisper_model_dir}/${VAD_FILENAME}"
verify_existing_file "${whisper_path}" "${WHISPER_SHA256}"
verify_existing_file "${vad_path}" "${VAD_SHA256}"

translation_model_ready=false
if verify_translation_model "${translation_model_dir}"; then
    translation_model_ready=true
    echo "Already downloaded: ${translation_model_dir}"
fi

if [[ -f "${whisper_path}" && -f "${vad_path}" && "${translation_model_ready}" == true ]]; then
    echo "Model weights are ready."
    exit 0
fi

work_dir="$(mktemp -d "${project_root}/models/.download.XXXXXX")"

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

if [[ "${translation_model_ready}" == false ]]; then
    translation_download_dir="${work_dir}/translation"
    echo "Downloading translation model from ModelScope (${TRANSLATION_REPOSITORY})"
    "${modelscope_cli}" download \
        "${TRANSLATION_REPOSITORY}" \
        --revision "${TRANSLATION_REVISION}" \
        --local-dir "${translation_download_dir}"
    verify_translation_model "${translation_download_dir}"
    mv -- "${translation_download_dir}" "${translation_model_dir}"
    echo "Installed ${translation_model_dir}"
fi

echo "Model weights are ready in ${project_root}/models."
