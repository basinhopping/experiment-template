"""Where things live, relative to the repository root.

Every location is fixed by convention inside the repository, on every machine (a laptop, a home GPU, the shared
node). No command accepts an output path from a caller, so no run can write somewhere unexpected. Data paths in
settings are relative to the repository's data/ folder, so the same sweep file works on every machine.
"""
from pathlib import Path, PurePosixPath


def repo_root() -> Path:
    """The repository root: the folder that holds src/sweepkit."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "src" / "sweepkit" / "__init__.py").is_file():
        raise RuntimeError(f"expected src/sweepkit under {root}; is the package being run from a copy?")
    return root


def package_dir() -> Path:
    return repo_root() / "src" / "sweepkit"


def project_code_dir() -> Path:
    """src/project/: the project's own datasets, models and steps, kept apart from the template's code."""
    return repo_root() / "src" / "project"


def data_dir() -> Path:
    """data/: the project's data. Never in git."""
    return repo_root() / "data"


def data_path(relative: str) -> Path:
    """A path given in settings, relative to data/. Absolute paths and '..' are refused, so a sweep file cannot
    point outside data/ and works unchanged on every machine."""
    if type(relative) is not str or not relative:
        raise ValueError(f"data path {relative!r}: give a path relative to data/")
    if "\\" in relative:
        raise ValueError(f"data path {relative!r}: use forward slashes")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or relative.startswith("~"):
        raise ValueError(f"data path {relative!r}: must stay inside data/ (no absolute path, no '..')")
    return data_dir().joinpath(*pure.parts)


def received_dir() -> Path:
    """data/received/<name>/: verified copies of folders received from elsewhere (ingest)."""
    return data_dir() / "received"


def manifests_dir() -> Path:
    """data/manifests/: one manifest per ingested folder (path, bytes, SHA-256 of every file) and its summary."""
    return data_dir() / "manifests"


def data_work_dir() -> Path:
    """data/_work/: resumable state of ingests (listings, copy and read-back logs)."""
    return data_dir() / "_work"


def synthetic_dir() -> Path:
    """data/synthetic/: tables written by make-synthetic."""
    return data_dir() / "synthetic"


def artifacts_dir() -> Path:
    """artifacts/: results. Written where jobs run, pulled to the user's computer. Never in git."""
    return repo_root() / "artifacts"


def checks_dir() -> Path:
    """artifacts/checks/: one folder per data or environment check."""
    return artifacts_dir() / "checks"


def env_record_dir() -> Path:
    """artifacts/env/: pip freeze of every environment when it was built."""
    return artifacts_dir() / "env"


def queue_dir() -> Path:
    """queue/: pending/, running/, done/, failed/, parked/ job files."""
    return repo_root() / "queue"


def logs_dir() -> Path:
    """logs/: one log per worker and per round. Never in git."""
    return repo_root() / "logs"


def sweeps_dir() -> Path:
    """artifacts/sweeps/<sweep_id>/: what a sweep was, as enqueued."""
    return artifacts_dir() / "sweeps"


def runs_dir(sweep_id: str) -> Path:
    """artifacts/runs/<sweep_id>/<run_id>/: one folder per job."""
    return artifacts_dir() / "runs" / sweep_id


def all_runs_dir() -> Path:
    return artifacts_dir() / "runs"


def analysis_dir() -> Path:
    """artifacts/analysis/<sweep_id>_<utc>/: one folder per validation analysis of a sweep."""
    return artifacts_dir() / "analysis"


def selections_dir() -> Path:
    """artifacts/selections/<name>/: a recorded choice of a configuration, made before any test or reveal file is
    read."""
    return artifacts_dir() / "selections"


def test_reports_dir() -> Path:
    """artifacts/test_reports/<selection>_<utc>/: test results of a selection."""
    return artifacts_dir() / "test_reports"


def reveal_reports_dir() -> Path:
    """artifacts/reveal_reports/<selection>_<utc>/: agreement of a selected clustering with known labels."""
    return artifacts_dir() / "reveal_reports"


def rules_dir() -> Path:
    """artifacts/rules/<name>_<utc>/: RULE files, each stored once with its UTC time and SHA-256."""
    return artifacts_dir() / "rules"


def comparisons_dir() -> Path:
    """artifacts/comparisons/<rule>_<utc>/: test-compare results of a RULE's comparison families."""
    return artifacts_dir() / "comparisons"


def analyses_dir() -> Path:
    """artifacts/analyses/<round>_prep_<date>/ and <round>_<date>/: a round's preparation and its reading."""
    return artifacts_dir() / "analyses"


def exports_dir() -> Path:
    """artifacts/exports/<selection>_<utc>/: results to take home (labels with row IDs, embeddings, summaries)."""
    return artifacts_dir() / "exports"


def atlas_dir() -> Path:
    """artifacts/atlas/atlas_<utc>/: every sweep, configuration, setting and value tried."""
    return artifacts_dir() / "atlas"


def architectures_dir() -> Path:
    """artifacts/architectures/<name>_<utc>/: exact networks of chosen configurations."""
    return artifacts_dir() / "architectures"


def estimates_dir() -> Path:
    """artifacts/estimates/<sweep_id>_<utc>/: measured time and memory on growing subsamples."""
    return artifacts_dir() / "estimates"


def smoke_dir() -> Path:
    """artifacts/smoke/<name>_<utc>/: smoke-all tables."""
    return artifacts_dir() / "smoke"


def proposals_dir() -> Path:
    """artifacts/proposals/<name>_<utc>/: sweep files proposed from finished runs (Bayesian search)."""
    return artifacts_dir() / "proposals"


def reports_dir() -> Path:
    """reports/: results docs and meeting summaries. Never in git (they hold results)."""
    return repo_root() / "reports"
