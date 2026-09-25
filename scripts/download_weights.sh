#!/usr/bin/env bash

set -euo pipefail

asr_backend="whisper"
check_only=false
while (($# > 0)); do
    case "$1" in
        --asr-backend)
            if [[ $# -lt 2 || ! "$2" =~ ^(whisper|qwen)$ ]]; then
                echo "Expected --asr-backend whisper|qwen" >&2
                exit 2
            fi
            asr_backend="$2"
            shift 2
            ;;
        --check)
            check_only=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--asr-backend whisper|qwen] [--check]"
            echo "Download or verify the selected ASR and shared translation weights."
            echo "--check verifies existing files without downloading."
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

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

readonly QWEN_ASR_REVISION="7278e1e70fe206f11671096ffdd38061171dd6e5"
readonly QWEN_ALIGNER_REVISION="c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
readonly QWEN_VAD_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
readonly QWEN_VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.2/src/silero_vad/data"

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
        if [[ "${check_only}" == true ]]; then
            echo "Required model file is missing: ${path}" >&2
            exit 1
        fi
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

prepare_work_dir() {
    if [[ -z "${work_dir}" ]]; then
        mkdir -p "${project_root}/models"
        work_dir="$(mktemp -d "${project_root}/models/.download.XXXXXX")"
    fi
}

require_modelscope() {
    if [[ ! -x "${modelscope_cli}" ]]; then
        echo "ModelScope CLI not found: ${modelscope_cli}; run uv sync first." >&2
        exit 1
    fi
}

download_http_file() {
    local url="$1" destination="$2" expected_sha256="$3"
    prepare_work_dir
    local downloaded="${work_dir}/$(basename -- "${destination}")"
    curl --fail --location --retry 3 --retry-all-errors --show-error \
        --output "${downloaded}" "${url}"
    mkdir -p "$(dirname -- "${destination}")"
    install_verified_file "${downloaded}" "${destination}" "${expected_sha256}"
}

qwen_weight_files() {
    cat <<'QWEN_FILES'
Qwen3-ASR-1.7B/chat_template.json 75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff
Qwen3-ASR-1.7B/config.json 2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f
Qwen3-ASR-1.7B/generation_config.json 1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c
Qwen3-ASR-1.7B/merges.txt 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5
Qwen3-ASR-1.7B/model-00001-of-00002.safetensors a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6
Qwen3-ASR-1.7B/model-00002-of-00002.safetensors 6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc
Qwen3-ASR-1.7B/model.safetensors.index.json f994739fe38e5210b9e3e8ce6c6307315e2ceac3cb630e7b7414d69dce520f60
Qwen3-ASR-1.7B/preprocessor_config.json 45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d
Qwen3-ASR-1.7B/tokenizer_config.json 4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c
Qwen3-ASR-1.7B/vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
Qwen3-ForcedAligner-0.6B/chat_template.json 75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff
Qwen3-ForcedAligner-0.6B/config.json d616c65d46c4b90bdc651b0a0963ea932732241140f337f9bb6b0335a9c8ef09
Qwen3-ForcedAligner-0.6B/generation_config.json 948d089b23bca1d214e768d59c4438365665f52ec6d33678f4062206b3fbbb8c
Qwen3-ForcedAligner-0.6B/merges.txt 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5
Qwen3-ForcedAligner-0.6B/model.safetensors 47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7
Qwen3-ForcedAligner-0.6B/preprocessor_config.json 45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d
Qwen3-ForcedAligner-0.6B/tokenizer_config.json 3ab80063f8511deb9566e6ad438d17b7a6277fcffd52d92854112f19d36bd81c
Qwen3-ForcedAligner-0.6B/vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
QWEN_FILES
}

download_qwen_model() {
    local model_name="$1" revision="$2"
    local relative_path expected_sha256 path
    local -a filenames=() checksums=()
    while read -r relative_path expected_sha256; do
        [[ "${relative_path}" == "${model_name}/"* ]] || continue
        path="${project_root}/models/${relative_path}"
        verify_existing_file "${path}" "${expected_sha256}"
        if [[ ! -f "${path}" ]]; then
            filenames+=("${relative_path#*/}")
            checksums+=("${expected_sha256}")
        fi
    done < <(qwen_weight_files)
    if ((${#filenames[@]} == 0)); then
        return
    fi
    require_command uvx
    prepare_work_dir
    local download_dir="${work_dir}/${model_name}"
    uvx hf download "Qwen/${model_name}" \
        --revision "${revision}" --local-dir "${download_dir}"
    mkdir -p "${project_root}/models/${model_name}"
    local index
    for index in "${!filenames[@]}"; do
        install_verified_file \
            "${download_dir}/${filenames[${index}]}" \
            "${project_root}/models/${model_name}/${filenames[${index}]}" \
            "${checksums[${index}]}"
    done
}

prepare_qwen_weights() {
    download_qwen_model "Qwen3-ASR-1.7B" "${QWEN_ASR_REVISION}"
    download_qwen_model "Qwen3-ForcedAligner-0.6B" "${QWEN_ALIGNER_REVISION}"
    local vad_path="${project_root}/models/silero-vad/silero_vad.onnx"
    verify_existing_file "${vad_path}" "${QWEN_VAD_SHA256}"
    if [[ ! -f "${vad_path}" ]]; then
        download_http_file "${QWEN_VAD_URL}/silero_vad.onnx" "${vad_path}" "${QWEN_VAD_SHA256}"
    fi
}

for command_name in awk sha256sum; do
    require_command "${command_name}"
done
if [[ "${check_only}" == false ]]; then
    for command_name in curl mktemp mv; do
        require_command "${command_name}"
    done
fi

whisper_path="${whisper_model_dir}/${WHISPER_FILENAME}"
vad_path="${whisper_model_dir}/${VAD_FILENAME}"
if [[ "${asr_backend}" == "qwen" ]]; then
    prepare_qwen_weights
else
    verify_existing_file "${whisper_path}" "${WHISPER_SHA256}"
    verify_existing_file "${vad_path}" "${VAD_SHA256}"
fi

translation_model_ready=false
if verify_translation_model "${translation_model_dir}"; then
    translation_model_ready=true
    echo "Already downloaded: ${translation_model_dir}"
fi

if [[ ( "${asr_backend}" == "qwen" || ( -f "${whisper_path}" && -f "${vad_path}" ) ) && "${translation_model_ready}" == true ]]; then
    echo "Model weights are ready."
    exit 0
fi

if [[ "${check_only}" == true ]]; then
    echo "Required model files are missing; run without --check to download." >&2
    exit 1
fi
prepare_work_dir

if [[ "${asr_backend}" == "whisper" && ! -f "${whisper_path}" ]]; then
    require_modelscope
    mkdir -p "${whisper_model_dir}"
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

if [[ "${asr_backend}" == "whisper" && ! -f "${vad_path}" ]]; then
    download_http_file "${VAD_URL}" "${vad_path}" "${VAD_SHA256}"
fi

if [[ "${translation_model_ready}" == false ]]; then
    require_modelscope
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
