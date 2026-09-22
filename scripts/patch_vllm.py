#!/usr/bin/env python3
import importlib.metadata
from pathlib import Path
import subprocess
import sys

EXPECTED_VERSION = "0.27.1+rocm723"
try:
    distribution = importlib.metadata.distribution("vllm")
except importlib.metadata.PackageNotFoundError as error:
    raise SystemExit("vLLM is not installed in the project environment.") from error
if distribution.version != EXPECTED_VERSION:
    raise SystemExit(
        f"Unexpected vLLM version: {distribution.version}; expected {EXPECTED_VERSION}."
    )

vllm_package = Path(str(distribution.locate_file("vllm")))
platform_init = vllm_package / "platforms" / "__init__.py"
rocm_module = vllm_package / "platforms" / "rocm.py"
PLATFORM_MARKER = "# Simple-Radeon-Subs WSL ROCm fallback"
DEVICE_MARKER = "# Simple-Radeon-Subs WSL amdsmi device-name fallback"


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    source = path.read_text(encoding="utf-8")
    if marker in source:
        return False
    if old not in source:
        raise SystemExit(f"vLLM source layout is not recognized: {path}")
    patched = source.replace(old, new, 1)
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")
    return True


installed = distribution.version

smoke_test = """
import torch
import vllm
import vllm._C
import vllm._rocm_C
from vllm.platforms import current_platform

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the ROCm GPU")
if not current_platform.is_rocm():
    raise SystemExit(f"vLLM selected the wrong platform: {current_platform}")
print(type(current_platform).__name__, current_platform.get_device_name())
"""


def check_runtime() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", smoke_test],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode == 0:
        return True
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return False

platform_old = '''def rocm_platform_plugin() -> str | None:
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available.")
            else:
                logger.debug("ROCm platform is not available because no GPU is found.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is not available because: %s", str(e))

    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None
'''
platform_new = '''def rocm_platform_plugin() -> str | None:
    """Detect ROCm, including WSL where AMD SMI is unavailable."""
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if amdsmi.amdsmi_get_processor_handles():
                logger.debug("Confirmed ROCm platform is available.")
                return "vllm.platforms.rocm.RocmPlatform"
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as error:
        logger.debug("AMD SMI ROCm detection failed: %s", error)

    # Simple-Radeon-Subs WSL ROCm fallback
    try:
        import torch

        if (
            not vllm_version_matches_substr("cpu")
            and torch.version.hip is not None
            and torch.accelerator.is_available()
        ):
            logger.debug("Confirmed ROCm platform via PyTorch WSL fallback.")
            return "vllm.platforms.rocm.RocmPlatform"
    except (ImportError, AttributeError, RuntimeError) as error:
        logger.debug("PyTorch ROCm fallback failed: %s", error)
    return None
'''

device_old = '''    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        asic_info_device_id: str = asic_info["device_id"]
        if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]
        return asic_info["market_name"]
'''
device_new = '''    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        # Simple-Radeon-Subs WSL amdsmi device-name fallback
        try:
            return cls._get_device_name_from_amdsmi(device_id)
        except (AmdSmiException, RuntimeError):
            return torch.cuda.get_device_properties(device_id).name

    @classmethod
    @with_amdsmi_context
    def _get_device_name_from_amdsmi(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        asic_info_device_id: str = asic_info["device_id"]
        if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]
        return asic_info["market_name"]
'''

def apply_patches() -> bool:
    changed = replace_once(platform_init, platform_old, platform_new, PLATFORM_MARKER)
    return replace_once(rocm_module, device_old, device_new, DEVICE_MARKER) or changed


def main() -> int:
    if check_runtime():
        print(f"vLLM {installed} already detects ROCm; no patch needed.")
        return 0

    changed = apply_patches()
    if not check_runtime():
        print("vLLM still cannot detect ROCm after patching.", file=sys.stderr)
        return 1
    action = "patched" if changed else "already patched"
    print(f"vLLM {installed} WSL ROCm detection: {action}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
