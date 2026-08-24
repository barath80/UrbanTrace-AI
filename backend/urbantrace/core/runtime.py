"""Runtime/GPU setup. MUST be imported before onnxruntime creates a session.

onnxruntime's CUDA execution provider needs cudnn64_9.dll. We do not add a
separate cuDNN dependency: the installed PyTorch build already ships the exact
cuDNN 9 binaries, so we expose torch's lib directory to the Windows DLL loader.
"""
from __future__ import annotations

import os
import platform
from functools import lru_cache


@lru_cache(maxsize=1)
def enable_gpu_dlls() -> str:
    """Expose torch's bundled CUDA/cuDNN DLLs. Returns the resolved device."""
    try:
        import torch
    except ImportError:
        return "cpu"

    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if platform.system() == "Windows" and os.path.isdir(lib):
        try:
            os.add_dll_directory(lib)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def device() -> str:
    return enable_gpu_dlls()


@lru_cache(maxsize=1)
def onnx_providers() -> list:
    if device() == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def describe() -> dict:
    info = {"device": device(), "cuda": False, "gpu_name": None, "torch": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    return info
