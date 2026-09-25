#!/usr/bin/env bash

set -uo pipefail

asr_backend="whisper"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [--asr-backend whisper|qwen]"
    exit 0
fi
if (($# > 0)); then
    if [[ $# != 2 || "$1" != "--asr-backend" || ! "$2" =~ ^(whisper|qwen)$ ]]; then
        echo "Usage: $0 [--asr-backend whisper|qwen]" >&2
        exit 2
    fi
    asr_backend="$2"
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv"
python="${venv_dir}/bin/python"
whisper_cli="${venv_dir}/bin/whisper-cli"
whisper_model="${project_root}/models/whisper/ggml-large-v3-turbo.bin"
vad_model="${project_root}/models/whisper/ggml-silero-v6.2.0.bin"
failures=0

pass() {
    printf 'PASS  %s\n' "$1"
}

fail() {
    printf 'FAIL  %s\n' "$1" >&2
    failures=$((failures + 1))
}

for command_name in ffmpeg ffprobe cmake git hipcc rocminfo; do
    if command_path="$(command -v "${command_name}" 2>/dev/null)"; then
        pass "${command_name}: ${command_path}"
    else
        fail "required command not found: ${command_name}"
    fi
done

if [[ -x "${python}" ]]; then
    pass "project Python: ${python}"
else
    fail "project Python not found: ${python}"
fi

if [[ -x "${python}" ]]; then
    if "${python}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12, found {sys.version.split()[0]}")

for package in ("modelscope", "torch", "triton", "vllm"):
    try:
        print(f"  {package}: {version(package)}")
    except PackageNotFoundError as error:
        raise SystemExit(f"Required package is not installed: {package}") from error

import torch
import triton
import vllm
import vllm._C
import vllm._rocm_C
from vllm.platforms import current_platform

if not torch.version.hip:
    raise SystemExit("Installed PyTorch is not a ROCm build")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the ROCm GPU")
if not current_platform.is_rocm():
    raise SystemExit(f"vLLM selected the wrong platform: {current_platform}")

device_name = torch.cuda.get_device_name(0)
x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
if y.shape != (1024, 1024):
    raise SystemExit(f"Unexpected GPU result shape: {y.shape}")

print(f"  Python: {sys.version.split()[0]}")
print(f"  HIP: {torch.version.hip}")
print(f"  GPU: {device_name}")
print("  vLLM ROCm extensions: loaded")
print("  BF16 GPU computation: passed")
PY
    then
        pass "Python and ROCm runtime"
    else
        fail "Python or ROCm runtime"
    fi
fi

if [[ "${asr_backend}" == "qwen" ]]; then
    if "${project_root}/scripts/download_weights.sh" --asr-backend qwen --check; then
        pass "Qwen ASR, ForcedAligner and Silero weights"
    else
        fail "Qwen model files"
    fi
    if "${python}" -c 'import nagisa, silero_vad; from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor'; then
        pass "Qwen processor and VAD dependencies"
    else
        fail "Qwen Python dependencies"
    fi
else
    if [[ -x "${whisper_cli}" ]]; then
        if "${whisper_cli}" --version >/dev/null 2>&1; then
            pass "whisper-cli executable"
        else
            fail "whisper-cli could not start: ${whisper_cli}"
        fi

        ldd_output="$(ldd "${whisper_cli}" 2>&1)"
        if grep -q "not found" <<<"${ldd_output}"; then
            fail "whisper-cli has unresolved shared libraries"
        elif grep -Eq 'lib(whisper|ggml)' <<<"${ldd_output}"; then
            fail "whisper-cli depends on non-system whisper.cpp shared libraries"
        else
            pass "whisper-cli native library linkage"
        fi
    else
        fail "whisper-cli not found: ${whisper_cli}"
    fi

    if [[ -f "${whisper_model}" && -s "${whisper_model}" ]]; then
        pass "Whisper model: ${whisper_model}"
    else
        fail "Whisper model missing or empty: ${whisper_model}"
    fi

    if [[ -f "${vad_model}" && -s "${vad_model}" ]]; then
        pass "VAD model: ${vad_model}"
    else
        fail "VAD model missing or empty: ${vad_model}"
    fi
fi

if ((failures > 0)); then
    printf '\nEnvironment doctor found %d failure(s).\n' "${failures}" >&2
    exit 1
fi

printf '\nEnvironment doctor passed.\n'
