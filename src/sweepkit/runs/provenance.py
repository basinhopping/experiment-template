"""How a result was produced, stamped into its folder at write time.

Records the code (hash of every package file of src/sweepkit and src/project, the pushed archive if known, the git
commit if the tree is a repository), the machine, Python and package versions, and the environment variables that
change behaviour. Package versions are read from installed metadata, so recording them imports nothing heavy. The
git commit is read from the files under .git, never by running git (running git from some machines leaves lock
files behind).
"""
import datetime
import importlib.metadata
import os
import platform
import socket
import sys

from sweepkit import paths
from sweepkit.hashing import code_hash

# Distributions recorded in every provenance file when installed. Each entry: one name to show, then the
# distribution names to try (the first one installed is recorded).
DISTRIBUTIONS = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("scikit-learn", "scikit-learn"),
    ("matplotlib", "matplotlib"), ("networkx", "networkx"), ("pyarrow", "pyarrow"), ("polars", "polars"),
    ("h5py", "h5py"), ("openpyxl", "openpyxl"), ("psutil", "psutil"), ("joblib", "joblib"),
    ("threadpoolctl", "threadpoolctl"), ("torch", "torch"), ("torchvision", "torchvision"),
    ("hdbscan", "hdbscan"), ("fast_hdbscan", "fast_hdbscan", "fast-hdbscan"), ("umap-learn", "umap-learn"),
    ("pynndescent", "pynndescent"), ("openTSNE", "openTSNE", "opentsne"), ("pacmap", "pacmap"), ("trimap", "trimap"),
    ("phate", "phate"), ("faiss", "faiss-cpu", "faiss-gpu", "faiss"), ("tslearn", "tslearn"), ("minisom", "MiniSom",
                                                                                               "minisom"),
    ("numba", "numba"), ("llvmlite", "llvmlite"), ("cuml", "cuml-cu13", "cuml-cu12", "cuml"),
    ("cudf", "cudf-cu13", "cudf-cu12", "cudf"), ("cupy", "cupy-cuda13x", "cupy-cuda12x", "cupy"),
    ("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("tpot", "tpot"), ("transformers", "transformers"),
    ("peft", "peft"), ("huggingface_hub", "huggingface_hub", "huggingface-hub"), ("accelerate", "accelerate"),
    ("timm", "timm"), ("safetensors", "safetensors"), ("tokenizers", "tokenizers"),
    ("torch_geometric", "torch_geometric", "torch-geometric"), ("scanpy", "scanpy"), ("anndata", "anndata"),
    ("scvi-tools", "scvi-tools"), ("rapids-singlecell", "rapids-singlecell-cu13", "rapids-singlecell"),
    ("optuna", "optuna"), ("genieclust", "genieclust"), ("lumbermark", "lumbermark"), ("leidenalg", "leidenalg"),
    ("igraph", "igraph", "python-igraph"), ("kmedoids", "kmedoids"), ("flowsom", "flowsom"),
    ("harmonypy", "harmonypy"), ("inmoose", "inmoose"), ("kmodes", "kmodes"), ("gower", "gower"),
)
ENV_VARS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
            "SWEEPKIT_CLEAR_MPS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
            "HF_HOME", "CUML_ACCEL_LOG_LEVEL", "PYTHONPATH")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def package_versions() -> dict:
    out = {}
    for shown, *candidates in DISTRIBUTIONS:
        version = None
        for dist in candidates:
            try:
                version = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        out[shown] = version
    return out


def git_commit() -> str | None:
    """The commit HEAD points to, read from .git without running git. None outside a repository."""
    git = paths.repo_root() / ".git"
    if not git.is_dir():
        return None
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:]
        loose = git / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def pushed_code_version() -> str | None:
    """Last line of CODE_VERSIONS.log, written on the node by node/unpack.sh."""
    log = paths.repo_root() / "CODE_VERSIONS.log"
    if not log.is_file():
        return None
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else None


def code_hashes() -> dict:
    out = {"sweepkit": code_hash(paths.package_dir())}
    project = paths.project_code_dir()
    out["project"] = code_hash(project) if project.is_dir() and any(project.rglob("*.py")) else None
    return out


def collect(entry_point: str) -> dict:
    hashes = code_hashes()
    return {
        "entry_point": entry_point,
        "argv": sys.argv,
        "utc": utc_now(),
        "code_sha256": hashes["sweepkit"],
        "project_code_sha256": hashes["project"],
        "pushed_code_version": pushed_code_version(),
        "git_commit": git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": package_versions(),
        "environment": {name: os.environ.get(name) for name in ENV_VARS},
    }
