from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


EXPECTED_TORCH_VERSION = "2.12.0"
EXPECTED_TORCH_CUDA = "13.2"
EXPECTED_RWKV_VERSION = "0.8.32"


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _run_text(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


def _git_info(project_root: Path) -> dict[str, Any]:
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"], project_root),
        "branch": _run_text(["git", "branch", "--show-current"], project_root),
        "dirty": bool(
            _run_text(["git", "status", "--porcelain"], project_root)
        ),
    }


def _nvidia_smi(project_root: Path) -> dict[str, Any]:
    query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        project_root,
    )
    if query is None:
        return {"available": False, "gpus": []}

    gpus = []
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        index, name, driver, memory_mib, capability = fields
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "memory_mib": int(memory_mib),
                "compute_capability": capability,
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def _torch_info() -> dict[str, Any]:
    try:
        torch = import_module("torch")
    except (ImportError, OSError) as exc:
        return {
            "available": False,
            "import_error": f"{type(exc).__name__}: {exc}",
            "devices": [],
        }

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )

    cudnn_version = None
    if getattr(torch.backends, "cudnn", None) is not None:
        cudnn_version = torch.backends.cudnn.version()

    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime": str(torch.version.cuda),
        "cudnn_version": cudnn_version,
        "device_count": len(devices),
        "devices": devices,
    }


def collect_environment(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root).resolve()
    disk = shutil.disk_usage(root)
    torch_info = _torch_info()
    rwkv_version = _package_version("rwkv")

    checks = {
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
        "torch_version_pinned": str(torch_info.get("version", "")).split("+")[0]
        == EXPECTED_TORCH_VERSION,
        "torch_cuda_pinned": torch_info.get("cuda_runtime") == EXPECTED_TORCH_CUDA,
        "cuda_available": torch_info.get("cuda_available") is True,
        "rwkv_version_pinned": rwkv_version == EXPECTED_RWKV_VERSION,
    }

    return {
        "report_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "project_root": str(root),
        "git": _git_info(root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
        },
        "nvidia_smi": _nvidia_smi(root),
        "torch": torch_info,
        "rwkv": {"version": rwkv_version},
        "runtime_environment": {
            key: os.environ.get(key)
            for key in ("RWKV_V7_ON", "RWKV_JIT_ON", "RWKV_CUDA_ON")
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "expected": {
            "torch": EXPECTED_TORCH_VERSION,
            "torch_cuda": EXPECTED_TORCH_CUDA,
            "rwkv": EXPECTED_RWKV_VERSION,
        },
        "checks": checks,
        "valid": all(checks.values()),
    }
