#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${project_root}/.venv/bin/python"

if [[ ! -x "${python}" ]]; then
    echo "Project environment not found: ${project_root}/.venv" >&2
    echo "Run uv sync before patching vLLM." >&2
    exit 1
fi

exec "${python}" - <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXPECTED_VERSION = "0.25.1+rocm723"

PATCHES = {
    "vllm/platforms/interface.py": {
        "original_sha256": "6fbe3057db1f540bf92c3d09bfbcdb5461f6c2ee70ff4ba069ad319e33d42cbd",
        "patched_sha256": "450535f2ab31f35a9032855df925840a4c362e49eaa0a66bef036e6ea0d3a52d",
        "replacements": (
            (
                "            logger.warning_once(\n"
                "                \"Using 'pin_memory=False' as WSL is detected. \"\n",
                "            logger.warning(\n"
                "                \"Using 'pin_memory=False' as WSL is detected. \"\n",
            ),
        ),
    },
    "vllm/platforms/__init__.py": {
        "original_sha256": "a2bd800acc39b3215ccb78808d43317b351f137072b03e7f0f0ab3d069d91521",
        "patched_sha256": "b776d44b4727f0d49da6996af09d703b7316649a026252037c01fadaf1de9fe2",
        "replacements": (
            (
                "from vllm.utils.import_utils import resolve_obj_by_qualname\n\n"
                "from .interface",
                "from vllm.utils.import_utils import resolve_obj_by_qualname\n\n"
                "from . import interface\n"
                "from .interface",
            ),
            (
                "    except Exception as e:\n"
                "        logger.debug(\"ROCm platform is not available because: %s\", str(e))\n\n"
                "    return \"vllm.platforms.rocm.RocmPlatform\" if is_rocm else None\n",
                "    except Exception as e:\n"
                "        logger.debug(\"ROCm platform is not available because: %s\", str(e))\n\n"
                "    if not is_rocm and interface.in_wsl():\n"
                "        try:\n"
                "            import torch\n\n"
                "            if getattr(torch.version, \"hip\", None):\n"
                "                is_rocm = True\n"
                "                logger.debug(\"Confirmed ROCm platform is available in WSL via torch.version.hip.\")\n"
                "        except Exception as e:\n"
                "            logger.debug(\"WSL ROCm fallback detection failed because: %s\", str(e))\n\n"
                "    return \"vllm.platforms.rocm.RocmPlatform\" if is_rocm else None\n",
            ),
        ),
    },
    "vllm/platforms/rocm.py": {
        "original_sha256": "52a4adae10bdcd8ce6b0b59865315bbb8690639d12e0bca6924667da9adbce88",
        "patched_sha256": "189304eaf017ff3a4e2cf72f1e29b7addf6e5e184561d13187d7f8b8e0bfdc95",
        "replacements": (
            (
                "        logger.warning_once(\n"
                "            \"Failed to get GCN arch via amdsmi, falling back to torch.cuda. \"",
                "        # warning_once -> circular import via vllm.distributed -> current_platform\n"
                "        logger.warning(\n"
                "            \"Failed to get GCN arch via amdsmi, falling back to torch.cuda. \"",
            ),
        ),
    },
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def replace_file(path: Path, content: bytes) -> None:
    mode = path.stat().st_mode
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


try:
    distribution = importlib.metadata.distribution("vllm")
except importlib.metadata.PackageNotFoundError as error:
    raise SystemExit("vLLM is not installed in the project environment.") from error

if distribution.version != EXPECTED_VERSION:
    raise SystemExit(
        f"Unsupported vLLM version: {distribution.version}; expected {EXPECTED_VERSION}."
    )

site_packages = Path(distribution.locate_file(""))
pending: dict[Path, bytes] = {}

for relative_path, patch in PATCHES.items():
    path = site_packages / relative_path
    if not path.is_file():
        raise SystemExit(f"Expected vLLM file not found: {path}")

    original_content = path.read_bytes()
    current_sha256 = sha256(original_content)
    if current_sha256 == patch["patched_sha256"]:
        continue
    if current_sha256 != patch["original_sha256"]:
        raise SystemExit(
            f"Refusing to modify unexpected vLLM file: {path}\n"
            f"Observed SHA256: {current_sha256}"
        )

    patched_text = original_content.decode("utf-8")
    for old, new in patch["replacements"]:
        if patched_text.count(old) != 1:
            raise SystemExit(f"Expected patch context not found exactly once in {path}")
        patched_text = patched_text.replace(old, new, 1)

    patched_content = patched_text.encode("utf-8")
    actual_patched_sha256 = sha256(patched_content)
    if actual_patched_sha256 != patch["patched_sha256"]:
        raise SystemExit(
            f"Generated unexpected patched content for {path}\n"
            f"Observed SHA256: {actual_patched_sha256}"
        )
    compile(patched_text, str(path), "exec")
    pending[path] = patched_content

if not pending:
    print(f"vLLM {EXPECTED_VERSION} is already patched.")
else:
    for path, content in pending.items():
        replace_file(path, content)
        print(f"Patched {path}")

smoke_test = """
import torch
import vllm
import vllm._C
import vllm._rocm_C
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    raise SystemExit(f"vLLM selected the wrong platform: {current_platform}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the ROCm GPU")
"""
subprocess.run([sys.executable, "-c", smoke_test], check=True)
print("vLLM WSL patch smoke test passed.")
PY
