"""Devices: which processor a job runs on, decided once and recorded.

A job's fit section names a device: cpu, cuda, mps or auto. A worker offers what it has (a GPU worker cuda, a Mac
mps, any machine cpu). resolve() turns the request and the offer into the device used:

    cpu    always the CPU
    cuda   an NVIDIA GPU; the job stops if none is visible (it never falls back to the CPU silently)
    mps    Apple's GPU through PyTorch; the job stops if it is not available
    auto   the worker's device when the model has a path for it, else the CPU; also the CPU when the model's GPU
           path cannot take these settings (a model's gpu_unsupported(settings) gives the reason, which the run
           records among its fallbacks: cuML's HDBSCAN with a metric other than Euclidean, for example)

Every run records the device requested and the device used in provenance.json, and analyze warns when the runs of
one configuration used different devices.
"""
import importlib.util
import os
import shutil
import subprocess

DEVICES = ("cpu", "cuda", "mps", "auto")


def gpu_visible() -> bool:
    """True when an NVIDIA GPU is visible to this process (CUDA_VISIBLE_DEVICES respected)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    if importlib.util.find_spec("torch") is not None:
        import torch

        return bool(torch.cuda.is_available())
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy

            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:  # no driver, no device: not visible
            return False
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    done = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=60)
    return done.returncode == 0 and "GPU" in done.stdout


def mps_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def auto_note(requested: str, offered: str, used: str, unsupported: str | None) -> list:
    """The fallback a run records when auto ran on the CPU on a GPU worker because of its settings."""
    if requested == "auto" and offered == "cuda" and used == "cpu" and unsupported is not None:
        return [f"fit.device auto ran on the CPU: {unsupported}"]
    return []


def gpu_path_ready(gpu: str) -> bool:
    """Whether a model's declared GPU path can run here: its packages are installed and a GPU is visible. cuGraph's
    path also needs cuML, which builds the neighbour graph on the GPU."""
    needs = {"cuml": ("cuml",), "torch": ("torch",), "xgboost": ("xgboost",), "lightgbm": ("lightgbm",),
             "faiss": ("faiss",), "rapids_singlecell": ("rapids_singlecell",), "cugraph": ("cugraph", "cuml")}
    if gpu == "none" or gpu not in needs:
        return False
    return all(has_package(name) for name in needs[gpu]) and gpu_visible()


def resolve(requested: str, offered: str, gpu: str, unsupported: str | None = None) -> str:
    """The device a job runs on. requested: the fit section's device; offered: the worker's (cpu, cuda or mps);
    gpu: the model's declared GPU path ("none" when it has none); unsupported: why the GPU path cannot take this
    job's settings, or None."""
    if requested not in DEVICES:
        raise ValueError(f"device {requested!r} is not one of {DEVICES}")
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        if offered == "cuda" and gpu != "none" and unsupported is None and gpu_path_ready(gpu):
            return "cuda"
        if offered == "mps" and gpu == "torch" and mps_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        if gpu == "none":
            raise ValueError("this model has no GPU path; use device cpu")
        if unsupported is not None:
            raise ValueError(f"{unsupported}; use fit.device cpu, or auto to run these settings on the CPU")
        if offered != "cuda" or not gpu_path_ready(gpu):
            raise RuntimeError(f"device cuda was requested, but this worker offers {offered} or the GPU path "
                               f"({gpu}) is not installed or no GPU is visible; the job stops rather than run on the "
                               "CPU")
        return "cuda"
    if gpu != "torch":
        raise ValueError("device mps is PyTorch's; this model has no PyTorch path")
    if offered != "mps" or not mps_available():
        raise RuntimeError("device mps was requested, but PyTorch's MPS backend is not available here")
    return "mps"
