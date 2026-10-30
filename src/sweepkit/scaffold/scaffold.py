"""scaffold: one new component in one step, with everything check-additions asks of it.

    python -m sweepkit.cli scaffold --kind clusterer --name my_method --where project

writes, from the text templates in src/sweepkit/scaffold/templates/ (text files, never imported):
    the module     a working starting point (it runs as written) with a docstring that says what to change
    its catalog    line in src/project/catalog.py (where project) or the right tuple of src/sweepkit/catalog.py
                   (where template)
    an example     sweep file: sweeps/project/<name>.json or sweeps/examples/<name>.json, on synthetic data
    a test         tests/test_project_<name>.py or tests/test_<kind>_<name>.py, a known-structure or by-hand test
Nothing is overwritten: an existing target stops it before anything is written. Then edit the module, run its test
and check-additions. A project's own components go in src/project/ (where project), so the template's files stay
untouched and later template fixes merge cleanly; where template is for changes meant for the template itself.
"""
import re

from sweepkit import paths

KINDS = ("clusterer", "embedder", "detector", "estimator", "prep", "metric", "split", "dataset", "stage_kind",
         "torch_model")
# where a template component's module goes, and the catalog tuple that lists it
TEMPLATE_PLACE = {"clusterer": ("models/cluster", "CLUSTERERS"), "embedder": ("models/embed", "EMBEDDERS"),
                  "detector": ("models/outlier", "DETECTORS"), "estimator": ("models", "ESTIMATORS"),
                  "prep": ("prep", "PREPARATION"), "metric": ("metrics", "METRICS"),
                  "split": ("splits", "SPLITS_AND_TASKS"), "dataset": ("datasets", "DATA"),
                  "stage_kind": ("sweep", "SPLITS_AND_TASKS"), "torch_model": ("models/torch", "TORCH")}
NO_EXAMPLE = ("metric",)
PROJECT_CATALOG = '''"""The project's own components: datasets, models, steps and metrics written for this project with
`python -m sweepkit.cli scaffold --where project`. sweepkit.catalog.load_all() imports this file when it exists, so
the template's own files stay untouched (and later template fixes merge cleanly)."""
import importlib

MODULES = (
)


def load_all() -> None:
    for name in MODULES:
        importlib.import_module(name)
'''


def camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def targets(kind: str, name: str, where: str) -> dict:
    root = paths.repo_root()
    if where == "project":
        return {"module": root / "src" / "project" / f"{name}.py", "module_name": f"project.{name}",
                "example": None if kind in NO_EXAMPLE else root / "sweeps" / "project" / f"{name}.json",
                "test": root / "tests" / f"test_project_{name}.py", "catalog": root / "src" / "project" / "catalog.py"}
    folder, _ = TEMPLATE_PLACE[kind]
    module = root / "src" / "sweepkit" / folder / f"{name}.py"
    dotted = "sweepkit." + folder.replace("/", ".") + f".{name}"
    return {"module": module, "module_name": dotted,
            "example": None if kind in NO_EXAMPLE else root / "sweeps" / "examples" / f"{name}.json",
            "test": root / "tests" / f"test_{kind}_{name}.py", "catalog": root / "src" / "sweepkit" / "catalog.py"}


def _render(text: str, values: dict) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def add_to_template_catalog(text: str, tuple_name: str, module: str) -> str:
    """The catalog's text with the module appended to the named tuple."""
    match = re.search(rf"^{tuple_name} = \((.*?)\)\n", text, flags=re.S | re.M)
    if match is None:
        raise ValueError(f"no tuple {tuple_name} in sweepkit/catalog.py")
    body = match.group(1).rstrip()
    if f'"{module}"' in body:
        raise ValueError(f"{module} is listed in {tuple_name} already")
    if body.endswith(","):
        body = body[:-1]
    new_body = (body + ", " if body else "") + f'"{module}"' + ("," if not body else "")
    return text[:match.start(1)] + new_body + text[match.end(1):]


def add_to_project_catalog(text: str, module: str) -> str:
    """The project catalog's text with the module appended to MODULES (in the order the components were added)."""
    if f'"{module}"' in text:
        raise ValueError(f"{module} is listed in src/project/catalog.py already")
    match = re.search(r"^MODULES = \(\n(.*?)^\)\n", text, flags=re.S | re.M)
    if match is None:
        raise ValueError("no MODULES = ( ... ) in src/project/catalog.py")
    return text[:match.end(1)] + f'    "{module}",\n' + text[match.end(1):]


def scaffold(kind: str, name: str, where: str) -> dict:
    if kind not in KINDS:
        raise ValueError(f"--kind {kind!r}: one of {KINDS} (torch_model: a network of the torch family, which "
                         "trains it)")
    if where not in ("project", "template"):
        raise ValueError("--where project or template")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", name) or name.endswith("_"):
        raise ValueError(f"--name {name!r}: lowercase letters, digits and _, starting with a letter")
    from sweepkit import catalog
    from sweepkit.registry import METRICS, REGISTRIES
    from sweepkit.sweep.design import STAGE_KINDS

    catalog.load_all()
    taken = {n for r in REGISTRIES.values() for n in r.names()} | set(METRICS.names()) | set(STAGE_KINDS)
    if name in taken:
        raise ValueError(f"{name!r} is registered already")
    t = targets(kind, name, where)
    for key in ("module", "example", "test"):
        if t[key] is not None and t[key].exists():
            raise FileExistsError(f"{t[key]} exists; scaffold never overwrites")
    folder = paths.package_dir() / "scaffold" / "templates"
    values = {"name": name, "Name": camel(name), "example": name}
    t["module"].parent.mkdir(parents=True, exist_ok=True)
    if where == "project":
        init = t["module"].parent / "__init__.py"
        if not init.exists():
            init.write_text('"""The project\'s own components (see catalog.py)."""\n', encoding="utf-8")
        if not t["catalog"].exists():
            t["catalog"].write_text(PROJECT_CATALOG, encoding="utf-8")
        catalog_text = add_to_project_catalog(t["catalog"].read_text(encoding="utf-8"), t["module_name"])
    else:
        catalog_text = add_to_template_catalog(t["catalog"].read_text(encoding="utf-8"), TEMPLATE_PLACE[kind][1],
                                               t["module_name"])
    t["module"].write_text(_render((folder / f"{kind}.py.tmpl").read_text(encoding="utf-8"), values),
                           encoding="utf-8", newline="\n")
    t["test"].write_text(_render((folder / f"test_{kind}.py.tmpl").read_text(encoding="utf-8"), values),
                         encoding="utf-8", newline="\n")
    if t["example"] is not None:
        t["example"].parent.mkdir(parents=True, exist_ok=True)
        t["example"].write_text(_render((folder / f"example_{kind}.json.tmpl").read_text(encoding="utf-8"), values),
                                encoding="utf-8", newline="\n")
    t["catalog"].write_text(catalog_text, encoding="utf-8", newline="\n")
    return {k: str(v) for k, v in t.items() if v is not None}
