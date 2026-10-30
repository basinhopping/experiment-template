"""Environment profiles: which packages each kind of problem needs, at which exact versions, under which licences, on
which platforms.

The pins live in env/requirements-<profile>.txt (one file per profile, read here); this module adds what a person or
a Claude session needs to choose profiles: what each one is for, its licences, where it installs, and the import
names the tests may require of an environment built from it (node/check.sh passes them to SWEEPKIT_REQUIRE).

Commands built on it:
    env-check --profiles core torch ...     installed versions against the pins, and the devices this process sees
    required-modules --profiles ...         the import names an environment built from those profiles must have
    profiles-doc --write yes                writes docs/PROFILES.md from this table and the registry's declarations
"""
import importlib.metadata
import importlib.util
import re

from sweepkit import paths

# name -> what it is for, the environment it goes in, licences, platforms, import names (for SWEEPKIT_REQUIRE)
PROFILES = {
    "core": {
        "for": "everything: settings, sweeps, the queue, analysis, data preparation, scikit-learn and SciPy models, "
               "and the template's own algorithms",
        "environment": "main", "default": "everywhere", "licences": "BSD, MIT, Apache-2.0 (permissive)",
        "platforms": "Windows, Mac, Linux; Python 3.12 or 3.13",
        "imports": ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "networkx", "pyarrow", "h5py", "openpyxl",
                    "psutil", "polars"),
    },
    "torch": {
        "for": "neural networks (the torch family): MLPs, CNNs, RNNs, Transformers, spiking models, autoencoders",
        "environment": "main", "default": "everywhere", "licences": "BSD-3-Clause",
        "platforms": "Linux (CUDA 13 from PyPI; cu126 index for GPUs older than Turing), Windows (cu130 index for an "
                     "NVIDIA GPU, else CPU), Mac (CPU and MPS)",
        "imports": ("torch",),
    },
    "unsupervised": {
        "for": "HDBSCAN (hdbscan, fast_hdbscan), UMAP, openTSNE, PaCMAP, FAISS and pynndescent neighbours, tslearn",
        "environment": "main", "default": "everywhere", "licences": "BSD, MIT, Apache-2.0 (permissive)",
        "platforms": "Windows, Mac (FAISS needs macOS 14), Linux",
        "imports": ("numba", "hdbscan", "fast_hdbscan", "umap", "pynndescent", "openTSNE", "pacmap", "faiss",
                    "tslearn"),
    },
    "rapids": {
        "for": "cuML, cuDF and cuGraph: k-means, DBSCAN, HDBSCAN, UMAP, PCA, t-SNE, random forests, Leiden and "
               "Louvain and more on the GPU, and someone's own scikit-learn script unchanged under cuml.accel "
               "(node/rapids_run.sh)",
        "environment": "rapids", "default": "on the node", "licences": "Apache-2.0",
        "platforms": "Linux with an NVIDIA GPU only; its own environment (numpy 2.4, numba 0.64)",
        "imports": ("cuml", "cudf", "cugraph", "cupy", "hdbscan", "umap"),
    },
    "rapids-singlecell": {
        "for": "the single-cell workflow on the GPU (rapids_singlecell), added to the rapids environment",
        "environment": "rapids", "default": "on request", "licences": "MIT, BSD-3-Clause",
        "platforms": "Linux with an NVIDIA GPU", "imports": ("rapids_singlecell", "scanpy", "anndata"),
    },
    "boosting": {
        "for": "XGBoost (CPU or CUDA) and LightGBM", "environment": "main", "default": "on request",
        "licences": "Apache-2.0, MIT", "platforms": "Windows, Mac (brew install libomp), Linux",
        "imports": ("xgboost", "lightgbm"),
    },
    "pretrained": {
        "for": "Hugging Face models: fine-tuning (full, top layers, LoRA) and frozen encoders for features",
        "environment": "main", "default": "on request", "licences": "Apache-2.0", "platforms": "all; downloads weights",
        "imports": ("transformers", "tokenizers", "safetensors", "peft", "huggingface_hub", "accelerate", "timm",
                    "torchvision"),
    },
    "graph": {
        "for": "PyTorch Geometric (optional: the template's own GCN and GAT need only torch)",
        "environment": "main", "default": "on request", "licences": "MIT", "platforms": "all",
        "imports": ("torch_geometric",),
    },
    "singlecell": {
        "for": "scanpy, anndata and scvi-tools on the CPU", "environment": "main", "default": "on request",
        "licences": "BSD-3-Clause", "platforms": "all; Python 3.12 or 3.13", "imports": ("scanpy", "anndata", "scvi"),
    },
    "search": {
        "for": "Optuna, for propose (Bayesian search that writes the next sweep file)", "environment": "main",
        "default": "on request", "licences": "MIT", "platforms": "all", "imports": ("optuna",),
    },
    "automl": {
        "for": "TPOT: genetic programming over scikit-learn pipelines", "environment": "automl",
        "default": "on request", "licences": "LGPL-3.0", "platforms": "all; its own environment",
        "imports": ("tpot",),
    },
    "extras": {
        "for": "Genie and Lumbermark (genieclust, lumbermark), leidenalg and igraph, FasterPAM k-medoids, FlowSOM, "
               "Harmony, ComBat (inmoose), PHATE, TriMap",
        "environment": "main", "default": "on request, knowing the licences",
        "licences": "AGPL-3.0 (genieclust, lumbermark, deadwood), GPL-3.0 (leidenalg, igraph is GPL-2.0+, kmedoids, "
                    "flowsom, harmonypy, inmoose), PHATE non-commercial (Yale) by its repository, Apache-2.0 (trimap)",
        "platforms": "where they install; annoy (trimap) and inmoose build from source, harmonypy has no Windows wheel",
        "imports": ("genieclust", "lumbermark", "leidenalg", "igraph", "kmedoids", "flowsom", "harmonypy", "inmoose",
                    "phate", "trimap"),
    },
}

_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(==|<|<=|>=)\s*([^\s#;]+)")


def requirement_file(profile: str):
    if profile not in PROFILES:
        raise ValueError(f"no profile {profile!r}; profiles: {list(PROFILES)}")
    return paths.repo_root() / "env" / f"requirements-{profile}.txt"


def pins(profile: str) -> list:
    """[(distribution, operator, version)] from the profile's requirement file."""
    out = []
    for line in requirement_file(profile).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "--")):
            continue
        match = _LINE.match(line)
        if match is None:
            raise ValueError(f"{requirement_file(profile).name}: cannot read {line!r}")
        out.append((match.group(1), match.group(3), match.group(4)))
    return out


def required_imports(profiles: list) -> list:
    out = []
    for p in profiles:
        if p not in PROFILES:
            raise ValueError(f"no profile {p!r}; profiles: {list(PROFILES)}")
        out += [m for m in PROFILES[p]["imports"] if m not in out]
    return out


def _installed(distribution: str):
    for name in (distribution, distribution.replace("-", "_"), distribution.replace("_", "-")):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def env_check(profiles: list) -> dict:
    """Installed versions against the pins of the profiles, and the devices this process sees."""
    from sweepkit import devices

    rows, ok = [], True
    for profile in profiles:
        for dist, op, version in pins(profile):
            have = _installed(dist)
            if op == "==":
                fine = have == version
            elif op == "<":
                fine = have is not None
            else:
                fine = have is not None
            ok = ok and fine
            rows.append({"profile": profile, "package": dist, "pinned": f"{op}{version}", "installed": have,
                         "ok": fine})
    missing_imports = [m for m in required_imports(profiles) if importlib.util.find_spec(m) is None]
    ok = ok and not missing_imports
    report = {"profiles": list(profiles), "packages": rows, "missing_imports": missing_imports,
              "gpu_visible": devices.gpu_visible(), "mps_available": devices.mps_available(), "ok": ok}
    if importlib.util.find_spec("torch") is not None and report["gpu_visible"]:
        import torch

        report["cuda"] = {"devices": torch.cuda.device_count(), "names": [torch.cuda.get_device_name(i)
                                                                          for i in range(torch.cuda.device_count())],
                          "torch_cuda": torch.version.cuda}
    return report


def render_check(report: dict) -> str:
    lines = [f"env-check of profiles {', '.join(report['profiles'])}:"]
    for r in report["packages"]:
        if not r["ok"]:
            lines.append(f"  DIFFERENT {r['package']}: pinned {r['pinned']}, installed {r['installed']}")
    n_ok = sum(r["ok"] for r in report["packages"])
    lines.append(f"  {n_ok} of {len(report['packages'])} pinned packages as pinned")
    if report["missing_imports"]:
        lines.append(f"  cannot import: {', '.join(report['missing_imports'])}")
    lines.append(f"  GPU visible: {report['gpu_visible']}; MPS available: {report['mps_available']}")
    if "cuda" in report:
        lines.append(f"  CUDA devices: {report['cuda']['devices']} ({', '.join(report['cuda']['names'])}), torch built "
                     f"for CUDA {report['cuda']['torch_cuda']}")
    lines.append("ENV OK" if report["ok"] else "ENV DIFFERS")
    return "\n".join(lines)


def profiles_doc() -> str:
    """docs/PROFILES.md: for each profile its purpose, pins, licences and platforms, and the components that need it
    (from the registry's declarations, so the page lists every registered component of the template). A project's
    own components (src/project/) are left out, so that adding one never makes the template's page stale and a later
    `git pull upstream main` never conflicts on it."""
    from sweepkit import catalog
    from sweepkit.registry import REGISTRIES, from_template

    catalog.load_all()
    by_profile = {}
    for registry in REGISTRIES.values():
        for entry in registry.entries():
            profile = entry.meta.get("profile")
            if profile is not None and from_template(entry.target):
                by_profile.setdefault(profile, []).append((registry.kind, entry.name, entry.meta.get("gpu", "none")))
    lines = ["# Environment profiles", "",
             "Written by `python -m sweepkit.cli profiles-doc --write yes` from `src/sweepkit/env.py` and the "
             "registry's declarations; do not edit by hand (a test checks it is current).", "",
             "A profile is a set of pinned packages (`env/requirements-<profile>.txt`). An environment is built from "
             "one or more profiles: `main` (core, torch and unsupervised by default, and any on-request profile but "
             "rapids and automl), `rapids` (the node's GPU libraries) and `automl`. At home an environment is "
             "`.venv-<environment>` in the repository folder (`env/setup.sh` or `env/setup.ps1`); on the node it is "
             "`~/venvs/<project>-<environment>` (`node/setup_env.sh`). Install only with these scripts, never with an "
             "unpinned `pip install`.", "",
             "## Which profiles a problem needs", "",
             "| Problem | Profiles |", "|---|---|",
             "| Tables, scikit-learn models, k-means, GMM, hierarchical, spectral, DBSCAN, PCA, t-SNE | core |",
             "| Neural networks on tables, sequences, images or audio | core, torch |",
             "| HDBSCAN (the hdbscan package), UMAP, openTSNE, PaCMAP, time-series clustering | core, unsupervised |",
             "| Large data on the node's GPU (cuML) | rapids (its own environment) |",
             "| Single-cell data | core, singlecell (CPU); rapids, rapids-singlecell (GPU) |",
             "| Gradient boosting | core, boosting |",
             "| Pretrained Hugging Face models | core, torch, pretrained |",
             "| Bayesian search (propose) | core, search |",
             "| Genie, Lumbermark, leidenalg, FasterPAM, FlowSOM, Harmony, ComBat, PHATE, TriMap | core, extras (read "
             "the licences first) |", "",
             "## The profiles", ""]
    for name, p in PROFILES.items():
        lines += [f"### {name}", "", f"For: {p['for']}.", "",
                  f"Environment: {p['environment']}. Installed: {p['default']}. Licences: {p['licences']}. "
                  f"Platforms: {p['platforms']}.", "", "| Package | Pin |", "|---|---|"]
        lines += [f"| {d} | {op}{v} |" for d, op, v in pins(name)]
        components = sorted(by_profile.get(name, []))
        if components:
            lines += ["", "Components that need it: " + ", ".join(
                f"{n} ({kind}{'; GPU ' + gpu if gpu not in ('none', None) else ''})" for kind, n, gpu in components)
                + "."]
        lines.append("")
    lines += ["## Setup lines", "",
              "Home, Linux or Mac: `bash env/setup.sh main core torch unsupervised`.", "",
              "Home, Windows: `powershell -ExecutionPolicy Bypass -File env\\setup.ps1 -Environment main -Profiles "
              "\"core,torch,unsupervised\" -TorchIndex cpu` (or `-TorchIndex cu130` with an NVIDIA GPU).", "",
              "Node: `bash node/setup_env.sh main core torch unsupervised`, then `bash node/setup_env.sh rapids "
              "rapids`.", ""]
    return "\n".join(lines)
