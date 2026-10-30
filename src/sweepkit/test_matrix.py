"""The testing matrix: the last section of TESTING.md, written from the code so that it cannot drift.

For every registered component of the template, and for every method a preparation step offers, it lists:

- its tests: the test methods that name it (in their own code, in a constant of their module or class that they use,
  or in a helper of their module or class that they call), the most specific first, with the bounds each asserts
  (assertGreaterEqual and the like against a number: for a clustering method, the adjusted Rand index it must reach
  on data with known clusters; for a classifier, its accuracy on data it can learn);
- its smoke entry: the example sweep whose first configuration using it is what `smoke-all` runs;
- its GPU path, and for a GPU path other than PyTorch's (cuML, cuGraph, rapids-singlecell, XGBoost's CUDA), the tests
  that run the CPU and the GPU version on the same data and compare them.

Only the template's own components and tests are listed (src/sweepkit/, and tests/test_*.py but not a project's
tests/test_project_*.py), so a project's additions never make the template's TESTING.md stale; check-additions and
smoke-all cover those.

    python -m sweepkit.cli test-matrix --write yes     rewrites the section between the two markers in TESTING.md
    python -m sweepkit.cli test-matrix --write no      prints it

tests/test_docs.py fails when TESTING.md's section differs from what this writes.
"""
import ast
import typing

from sweepkit import paths

START = ("<!-- The testing matrix below is written by `python -m sweepkit.cli test-matrix --write yes` from the code; "
         "edit the code or the tests, not this section. -->")
END = "<!-- End of the testing matrix. -->"
OPS = {"assertGreaterEqual": "≥", "assertGreater": ">", "assertLessEqual": "≤", "assertLess": "<"}
# GPU paths that are a separate implementation, so the CPU and GPU versions need a test that they agree; PyTorch runs
# the same code on either device
SEPARATE_GPU_PATHS = ("cuml", "cugraph", "rapids_singlecell", "xgboost")
FAMILY_ORDER = ("clusterer", "embedder", "detector", "estimator", "torch")
FAMILY_TITLE = {"clusterer": "Clustering", "embedder": "Dimensionality reduction and embeddings",
                "detector": "Outlier detection", "estimator": "Estimators (classification and regression)",
                "torch": "Neural networks (PyTorch)"}
OTHER_KINDS = (("dataset", "Datasets"), ("split", "Splits"), ("task", "Tasks"), ("feature", "Features"),
               ("augmentation", "Augmentations"), ("fit procedure", "Fit procedures"))


# ---------------------------------------------------------------- the tests, read without importing them

def _strings(node) -> set:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and type(n.value) is str}


def _name_like(node) -> set:
    """The strings in node that can name a component: everything except a value given to another setting
    ({"kernel": "linear"} and kernel="linear" name a kernel, not the model `linear`); {"name": ...} counts."""
    out = set()

    def visit(n, allowed: bool) -> None:
        if isinstance(n, ast.Constant):
            if allowed and type(n.value) is str:
                out.add(n.value)
            return
        if isinstance(n, ast.Dict):
            for key, value in zip(n.keys, n.values):
                # a key with a plain value is a setting's name ({"knn": 5}); a key with a section is a label
                if key is not None:
                    visit(key, not isinstance(value, ast.Constant))
                other_setting = isinstance(key, ast.Constant) and type(key.value) is str and key.value != "name"
                visit(value, not (other_setting and isinstance(value, ast.Constant)))
            return
        if isinstance(n, ast.keyword):
            visit(n.value, n.arg == "name" or not isinstance(n.value, ast.Constant))
            return
        for child in ast.iter_child_nodes(n):
            visit(child, True)

    visit(node, True)
    return out


def _configured(node) -> set:
    """Strings given as a section's name: {"name": "kmeans", ...} or dict(KMEANS, name="minibatch_kmeans")."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Dict):
            for key, value in zip(n.keys, n.values):
                if (isinstance(key, ast.Constant) and key.value == "name" and isinstance(value, ast.Constant)
                        and type(value.value) is str):
                    out.add(value.value)
        elif (isinstance(n, ast.keyword) and n.arg == "name" and isinstance(n.value, ast.Constant)
              and type(n.value.value) is str):
            out.add(n.value.value)
    return out


def _assigned(body: list) -> dict:
    """name -> (the strings in the values assigned to it, the value nodes), for the assignments directly in body."""
    out = {}
    for node in body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    names, strings, configured, values = out.get(n.id, (set(), set(), set(), []))
                    out[n.id] = (names | _name_like(value), strings | _strings(value),
                                 configured | _configured(value), values + [value])
    return out


def _closed(constants: dict) -> dict:
    """Each constant's strings together with those of the constants its value names (CLASSIFIERS = (SVM, KNN))."""
    out = {}

    def gather(name: str, seen: set) -> tuple:
        names, strings, configured, values = constants[name]
        for value in values:
            for n in ast.walk(value):
                if isinstance(n, ast.Name) and n.id in constants and n.id not in seen:
                    more = gather(n.id, seen | {n.id})
                    names, strings, configured = names | more[0], strings | more[1], configured | more[2]
        return names, strings, configured, values

    for name in constants:
        out[name] = gather(name, {name})
    return out


def _referenced(fn) -> set:
    """Names a function uses: bare names, and attributes of self or cls."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"):
            out.add(n.attr)
    return out


def _number(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    return None


def _show(value) -> str:
    return f"{value:g}"


def _loop_values(fn, name: str, constants: dict) -> list:
    """The numbers a loop variable takes, when a for loop in fn unpacks it from literal tuples."""
    for loop in (n for n in ast.walk(fn) if isinstance(n, ast.For)):
        if not isinstance(loop.target, ast.Tuple):
            continue
        names = [t.id if isinstance(t, ast.Name) else None for t in loop.target.elts]
        if name not in names:
            continue
        index = names.index(name)
        source = loop.iter
        if isinstance(source, ast.Name) and source.id in constants:
            source = constants[source.id][3][-1]
        if not isinstance(source, (ast.Tuple, ast.List)):
            return []
        values = []
        for element in source.elts:
            if isinstance(element, ast.Tuple) and len(element.elts) > index:
                number = _number(element.elts[index])
                if number is not None:
                    values.append(number)
        return values
    return []


def _measure(text: str, node) -> str:
    segment = " ".join((ast.get_source_segment(text, node) or "").split()).replace("self.", "")
    if segment.startswith("float(") and segment.endswith(")"):
        segment = segment[len("float("):-1]
    return segment if len(segment) <= 60 else segment[:57] + "..."


def _exact_checks(fn) -> int:
    """Assertions of equality: assertEqual, assertAlmostEqual, assert_allclose, assert_array_equal and the like."""
    names = ("assertEqual", "assertAlmostEqual", "assertListEqual", "assertDictEqual", "assertTupleEqual",
             "assert_allclose", "assert_array_equal", "assert_array_almost_equal", "assert_equal")
    return sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
               and (n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id) in names)


def _limit(values: list) -> str:
    values = sorted(set(values))
    return _show(values[0]) if len(values) == 1 else f"{_show(values[0])} to {_show(values[-1])}"


def _asserts(fn) -> list:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in OPS and len(n.args) >= 2]


def _bound_through_helpers(fn, text: str, helpers: dict) -> list:
    """Bounds a test passes to a helper that asserts them: self.check(LOF, 0.98) with check(self, raw, threshold)
    asserting `auroc(...) >= threshold`."""
    out = []
    for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        helper = helpers.get(name)
        if helper is None or helper is fn:
            continue
        params = [a.arg for a in helper.args.args]
        if params and params[0] in ("self", "cls") and isinstance(func, ast.Attribute):
            params = params[1:]
        given = dict(zip(params, call.args))
        given.update({k.arg: k.value for k in call.keywords if k.arg})
        for inner in _asserts(helper):
            bound = inner.args[1]
            if isinstance(bound, ast.Name) and bound.id in given and _number(given[bound.id]) is not None:
                out.append((inner, OPS[inner.func.attr], _show(_number(given[bound.id]))))
    return out


def _bounds(fn, text: str, constants: dict, helpers: dict) -> list:
    out = []
    for call in _asserts(fn):
        op, bound = OPS[call.func.attr], call.args[1]
        number = _number(bound)
        if number is not None:
            limit = _show(number)
        elif isinstance(bound, ast.Name):
            values = _loop_values(fn, bound.id, constants)
            if not values:
                continue
            limit = _limit(values)
        else:
            continue
        shown = f"`{_measure(text, call.args[0])}` {op} {limit}"
        if shown not in out:
            out.append(shown)
    for inner, op, limit in _bound_through_helpers(fn, text, helpers):
        shown = f"`{_measure(text, inner.args[0])}` {op} {limit}"
        if shown not in out:
            out.append(shown)
    return out


def test_index(include=None, example_uses=None) -> list:
    """One entry per test method of the template's test files: {"id", "file", "class", "method", "line", "names",
    "strings", "configured", "bounds", "exact", "gpu"}. `include` limits the files (for tests of this module);
    `example_uses` maps an example sweep's name to the components it uses, so that a test which runs an example by
    name (example_base("image_cnn")) names what that example configures."""
    example_uses = example_uses or {}
    out = []
    files = sorted((paths.repo_root() / "tests").glob("test_*.py"))
    for path in files:
        if path.name.startswith("test_project_") or (include is not None and path.name not in include):
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        module_constants = _closed(_assigned(tree.body))
        module_helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith("Test")):
            class_constants = _closed(_assigned(cls.body))
            methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            constants = {**module_constants, **class_constants}
            def reach(fn) -> tuple:
                """What fn names itself, through the constants it uses and through the helpers it calls: (names,
                strings, configured)."""
                names, strings, configured = _name_like(fn), _strings(fn), _configured(fn)
                for ref in _referenced(fn):
                    if ref in constants:
                        names |= constants[ref][0]
                        strings |= constants[ref][1]
                        configured |= constants[ref][2]
                    helper = methods.get(ref) if ref in methods and not ref.startswith("test") else None
                    helper = helper or module_helpers.get(ref)
                    if helper is not None:
                        names |= _name_like(helper)
                        strings |= _strings(helper)
                        configured |= _configured(helper)
                return names, strings, configured

            # what the class's fixture builds belongs to every test of the class
            fixture = (set(), set(), set())
            for setup in ("setUpClass", "setUp"):
                if setup in methods:
                    more = reach(methods[setup])
                    fixture = tuple(a | b for a, b in zip(fixture, more))
            for name, fn in methods.items():
                if not name.startswith("test"):
                    continue
                names, strings, configured = (a | b for a, b in zip(reach(fn), fixture))
                for example in strings & set(example_uses):
                    names, configured = names | example_uses[example], configured | example_uses[example]
                where = f"{path.stem}.{cls.name}.{name}"
                gpu = any(word in part.lower() for part in (path.stem, cls.name, name)
                          for word in ("gpu", "cuda", "cuml", "cugraph"))
                out.append({"id": where, "file": path.name, "class": cls.name, "method": name, "line": fn.lineno,
                            "names": names, "strings": strings, "configured": configured,
                            "bounds": _bounds(fn, text, constants, {**module_helpers, **{
                                k: v for k, v in methods.items() if not k.startswith("test")}}),
                            "exact": _exact_checks(fn), "gpu": gpu})
    return out


# ---------------------------------------------------------------- the components

def _template_entries(kind: str) -> list:
    from sweepkit.registry import REGISTRIES, from_template

    return [e for e in REGISTRIES[kind].entries() if from_template(e.target)]


def _methods(entry) -> list:
    """The values of a preparation step's `method` setting, when it is a fixed list of names."""
    hints = typing.get_type_hints(entry.settings_class)
    hint = hints.get("method")
    if hint is None or typing.get_origin(hint) is not typing.Literal:
        return []
    return list(typing.get_args(hint))


def _named_for(test: dict, word: str) -> bool:
    """The test's method name has the word as whole parts (test_kmeans_... for kmeans, not for minibatch_kmeans), or
    its class name has it (TestHDBSCAN for hdbscan)."""
    parts, wanted = test["method"].lower().split("_"), word.lower().split("_")
    in_method = any(parts[i:i + len(wanted)] == wanted for i in range(len(parts) - len(wanted) + 1))
    return in_method or word.replace("_", "").lower() in test["class"].lower()


def _specific_first(tests: list, word: str, rivals: set) -> list:
    def key(t):
        return (0 if _named_for(t, word) else 1, 0 if word in t["configured"] else 1, 0 if t["bounds"] else 1,
                len(t["names"] & rivals), t["file"], t["line"])

    return sorted(tests, key=key)


def _cell(tests: list, shown: int) -> str:
    if not tests:
        return "none"
    parts = []
    for t in tests[:shown]:
        if t["bounds"]:
            bounds = "; ".join(t["bounds"][:2])
        elif t["exact"]:
            bounds = f"{t['exact']} exact check{'s' if t['exact'] > 1 else ''}"
        else:
            bounds = "no numeric check"
        parts.append(f"{t['id']}: {bounds}")
    if len(tests) > shown:
        parts.append("and more")            # not counted, so a new test elsewhere does not change this line
    return "<br>".join(p.replace("|", "\\|") for p in parts)


def _example_for(kind: str, name: str, meta: dict, examples: dict, method=None) -> str:
    from sweepkit.scaffold.additions import smoke_job

    chosen = smoke_job(kind, name, meta, examples, method=method)
    return f"`{chosen[0]}`" if chosen else "none"


def render() -> str:
    from sweepkit import catalog
    from sweepkit.registry import METRICS, from_template
    from sweepkit.scaffold.additions import expanded_examples

    catalog.load_all()
    examples = {k: v for k, v in expanded_examples().items() if v["path"].parent.name == "examples"}
    index = test_index(example_uses={k: {name for _, name in v["uses"]} for k, v in examples.items()})
    lines = [START, "", "## The testing matrix", "",
             "Every component of the template, and every method of a preparation step, with: the tests that name it "
             "(the most specific first; `file.Class.method`) and the bounds each asserts; its smoke entry (the example "
             "sweep whose first configuration using it `smoke-all` runs); and, for a GPU path that is a separate "
             "implementation (cuML, cuGraph, rapids-singlecell, XGBoost's CUDA), the tests that run the CPU and GPU "
             "versions on the same data and compare them. PyTorch models run the same code on either device. A bound "
             "is shown as the test asserts it, so its left side is code: `ari(truth, labels)` is the adjusted Rand "
             "index with the known clusters.", ""]
    gaps_bound, gaps_gpu, gaps_method = [], [], []

    models = _template_entries("model")
    rivals_of = {}
    for family in FAMILY_ORDER:
        rivals_of[family] = {e.name for e in models if e.meta["family"] == family}
    for family in FAMILY_ORDER:
        entries = [e for e in models if e.meta["family"] == family]
        if not entries:
            continue
        lines += [f"### {FAMILY_TITLE[family]}", "",
                  "| Model | Profile | GPU path | Smoke entry | Tests and their bounds | CPU and GPU agreement |",
                  "|---|---|---|---|---|---|"]
        for e in entries:
            named = [t for t in index if e.name in t["names"]]
            ordinary = _specific_first([t for t in named if not t["gpu"]], e.name, rivals_of[family])
            gpu_path = e.meta.get("gpu", "none")
            if gpu_path in SEPARATE_GPU_PATHS:
                agreement_tests = _specific_first([t for t in named if t["gpu"]], e.name, rivals_of[family])
                agreement = _cell(agreement_tests, 1)
                if not agreement_tests:
                    gaps_gpu.append(e.name)
            elif gpu_path == "torch":
                agreement = "the same code on both devices"
            else:
                agreement = "no GPU path"
            if not any(t["bounds"] for t in ordinary):
                gaps_bound.append(e.name)
            lines.append(f"| `{e.name}` | {e.meta.get('profile', 'core')} | {gpu_path} | "
                         f"{_example_for('model', e.name, e.meta, examples)} | {_cell(ordinary, 3)} | {agreement} |")
        lines.append("")

    steps = _template_entries("prep step")
    rivals = {e.name for e in steps}
    lines += ["### Preparation steps", "",
              "| Step | Method | Profile | Smoke entry | Tests and their bounds |", "|---|---|---|---|---|"]
    for e in steps:
        named = [t for t in index if e.name in t["names"] and not t["gpu"]]
        methods = _methods(e)
        rows = [(None, named)] if not methods else [(m, [t for t in named if m in t["strings"]]) for m in methods]
        for method, tests in rows:
            ordered = _specific_first(tests, method or e.name, rivals)
            if not tests:
                gaps_method.append(f"{e.name} {method}" if method else e.name)
            lines.append(f"| `{e.name}` | {f'`{method}`' if method else 'all'} | {e.meta.get('profile', 'core')} | "
                         f"{_example_for('prep step', e.name, e.meta, examples, method)} | {_cell(ordered, 3)} |")
    lines.append("")

    for kind, title in OTHER_KINDS:
        entries = _template_entries(kind)
        if not entries:
            continue
        rivals = {e.name for e in entries}
        lines += [f"### {title}", "", "| Name | Smoke entry | Tests and their bounds |", "|---|---|---|"]
        for e in entries:
            tests = _specific_first([t for t in index if e.name in t["names"]], e.name, rivals)
            lines.append(f"| `{e.name}` | {_example_for(kind, e.name, e.meta, examples)} | {_cell(tests, 2)} |")
        lines.append("")

    metric_names = [n for n in METRICS.names() if from_template(METRICS.get(n)["fn"])]
    rivals = set(metric_names)
    lines += ["### Metrics and criteria", "",
              "Metrics have no smoke entry of their own: every run scores the metrics and criteria its task names.", "",
              "| Metric | Tasks | Tests and their bounds |", "|---|---|---|"]
    for name in metric_names:
        tests = _specific_first([t for t in index if name in t["names"]], name, rivals)
        lines.append(f"| `{name}` | {', '.join(METRICS.get(name)['tasks'])} | {_cell(tests, 2)} |")
    lines.append("")

    lines += ["### Gaps", "",
              "Models whose tests assert no numeric bound (they are run and checked in other ways, such as shapes, "
              "exact values or refusals): " + (", ".join(f"`{n}`" for n in gaps_bound) or "none") + ".", "",
              "Preparation methods that no test names: " + (", ".join(f"`{n}`" for n in gaps_method) or "none") + ".",
              "", "GPU paths without a test that compares the CPU and GPU versions: "
              + (", ".join(f"`{n}`" for n in gaps_gpu) or "none") + ".", "", END]
    return "\n".join(lines) + "\n"


def testing_md_with(matrix: str) -> str:
    """TESTING.md with its matrix section replaced by `matrix` (or appended when the markers are missing)."""
    path = paths.repo_root() / "TESTING.md"
    text = path.read_text(encoding="utf-8")
    if START in text and END in text:
        before = text[:text.index(START)]
        after = text[text.index(END) + len(END):]
        return before + matrix.rstrip("\n") + after
    return text.rstrip("\n") + "\n\n" + matrix


def current_section() -> str:
    text = (paths.repo_root() / "TESTING.md").read_text(encoding="utf-8")
    if START not in text or END not in text:
        return ""
    return text[text.index(START):text.index(END) + len(END)] + "\n"
