"""docs/EXAMPLE_ROUND.md, written by running the template: two complete rounds on synthetic data, an unsupervised one
(how many clusters, and which method) and a supervised one (does a change beat the baseline), each from round-new to
the reading, with every file and output shown as the commands wrote it.

`python -m sweepkit.cli example-round --write yes` copies the code into a temporary folder, runs every command there as
a separate process (as from a terminal), and writes the page. Times, the temporary folder's path and SHA-256 values
(the code's tree digest changes with every edit) are replaced by <utc>, <repo> and <sha256>; everything else is the
commands' own output, so the page cannot drift from the code: tests/test_docs.py writes it again and compares, byte
for byte, where the installed numpy, SciPy, pandas and scikit-learn are the pinned versions (other versions may round
a number differently).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sweepkit import paths

DATE = "20261101"
END = "2026-11-02 08:00"

INTRO = """# An example round

Written by `python -m sweepkit.cli example-round --write yes`, which runs every command below in a temporary copy of
the template on synthetic data and records what it printed and wrote. Times are shown as `<utc>`, the temporary
folder as `<repo>`, and SHA-256 values (the code's tree digest changes with every edit) as `<sha256>`; every other
character is the commands' own output.

A round is one night on a shared node or one batch on one machine (docs/HOW_WE_WORK.md). On a node, the person pastes
the round's lines one at a time (`lines/`, below) and the workers run the sweeps; here `run` does the same work in one
process. The order never changes: prepare, fix the RULE before any result exists, run, read the choosing files,
record the choice, and only then read the files that judge it.
"""

ROUND1_SWEEP = {
    "sweep_id": "round1_screen",
    "description": "How many clusters do the data hold, and does a mixture model or Ward's hierarchical clustering "
                   "find them better than k-means? Eight subsamples, three seeds, chosen by silhouette.",
    "purpose": "screen", "follows": [], "folds": [f"sub{i}" for i in range(1, 9)],
    "base": {
        "data": {"name": "synthetic", "generator": "blobs", "n_rows": 800, "n_features": 6, "n_groups": 4,
                 "noise": 1.6, "seed": 21},
        "split": {"name": "subsamples", "k": 8, "fraction": 0.8, "salt": "round1"},
        "task": {"name": "clustering", "criteria": ["silhouette", "calinski_harabasz", "n_clusters"],
                 "select": "silhouette", "sample_rows": 500, "sample_salt": "round1", "reference": "truth",
                 "reveal_metrics": ["ari", "ami"], "criteria_options": {}},
        "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
        "features": {"name": "none"},
        "model": {"name": "kmeans", "n_clusters": 4, "init": "k-means++", "n_init": 4, "max_iter": 300, "tol": 1e-4,
                  "extra": {}},
        "fit": {"name": "clusterer", "device": "auto", "assign": "fit_rows"}},
    "stages": [
        {"name": "base", "kind": "base", "priority": 9, "seeds": [1, 2, 3]},
        {"name": "methods", "kind": "variants", "priority": 8, "seeds": [1, 2, 3],
         "variants": {"gmm": {"model": {"name": "gmm", "n_components": 4, "covariance_type": "full", "n_init": 2,
                                        "max_iter": 200, "reg_covar": 1e-6, "extra": {}}},
                      "ward": {"model": {"name": "agglomerative", "linkage": "ward", "metric": "euclidean",
                                         "cut": "n_clusters", "n_clusters": 4, "distance": None, "extra": {}}}}},
        {"name": "k", "kind": "one_factor_at_a_time", "priority": 7, "seeds": [1, 2, 3],
         "factors": {"model.n_clusters": [3, 5]}}]}

ROUND1_RULE = {
    "what_runs": "Sweep round1_screen (sweeps/project/round1_screen.json): k-means with 4 clusters (the base), a\n"
                 "   Gaussian mixture and Ward's agglomerative clustering with 4, and k-means with 3 and 5 clusters, each\n"
                 "   on 8 fixed subsamples of 80% of the rows with seeds 1, 2 and 3, z-scored, on the CPU. No earlier\n"
                 "   result chose any of these settings.",
    "reading": "Label-free files only first: analyze ranks the configurations by silhouette (larger is better) and\n"
               "   compares each with the base. select takes the highest, read with its stability across seeds and\n"
               "   subsamples and the number of clusters it finds: among the configurations whose intervals overlap\n"
               "   the highest's, the most stable; if a configuration more stable by over 0.05 ranks below the\n"
               "   choice, the round stops for a look at the data before any choice. Only then reveal-report reads\n"
               "   that one configuration's agreement with the planted groups (ARI, AMI). No comparison family is\n"
               "   read on the reveal files. Runs not finished at the reading are left out and named.",
    "stop_condition": "The choice goes to a final run on every row with five seeds. If its stability is below 0.9, or\n"
                      "   the number of clusters it finds differs between seeds, stop and look at the data before any\n"
                      "   final run.",
    "limits": "One synthetic dataset with round clusters, which favours silhouette and k-means; settings chosen on\n"
              "   the same subsamples they are judged on (optimistic until the final run); no other criterion read.",
}

ROUND2_SWEEP = {
    "sweep_id": "round2_screen",
    "description": "Two classes that no straight line separates (XOR): does a random forest or k-nearest neighbours "
                   "beat the baseline logistic regression on balanced accuracy? Five folds plus the identity check "
                   "fold, three seeds, with the shuffled-label control.",
    "purpose": "screen", "follows": [], "folds": ["fold1", "fold2", "fold3", "fold4", "fold5", "identity"],
    "base": {
        "data": {"name": "synthetic", "generator": "xor", "n_rows": 600, "n_features": 4, "n_groups": 2,
                 "noise": 0.25, "seed": 22},
        "split": {"name": "kfold", "k": 5, "stratify": "truth", "salt": "round2", "identity_fold": True},
        "task": {"name": "classification", "target": "truth", "classes": ["c0", "c1"],
                 "metrics": ["balanced_accuracy", "accuracy", "f1_macro", "roc_auc"], "select": "balanced_accuracy",
                 "groups": []},
        "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
        "features": {"name": "none"},
        "model": {"name": "logistic_regression", "c": 1.0, "l1_ratio": 0.0, "class_weight": "none", "max_iter": 1000,
                  "extra": {}},
        "fit": {"name": "estimator", "device": "auto", "control": "none", "refit": "train", "eval_train": False}},
    "stages": [
        {"name": "base", "kind": "base", "priority": 9, "seeds": [1, 2, 3]},
        {"name": "changes", "kind": "variants", "priority": 8, "seeds": [1, 2, 3],
         "variants": {"random_forest": {"model": {"name": "random_forest", "n_estimators": 200, "max_depth": None,
                                                  "min_samples_leaf": 2, "max_features": 0.5, "bootstrap": True,
                                                  "class_weight": "none", "n_jobs": 1, "extra": {}}},
                      "knn": {"model": {"name": "knn", "n_neighbors": 15, "weights": "uniform",
                                        "metric": "euclidean", "extra": {}}}}},
        {"name": "control", "kind": "variants", "priority": 7, "seeds": [1, 2, 3],
         "variants": {"shuffled_labels": {"fit.control": "shuffle_train_labels"}}}]}

ROUND2_RULE = {
    "what_runs": "Sweep round2_screen (sweeps/project/round2_screen.json): logistic regression (the base), a random\n"
                 "   forest of 200 trees, k-nearest neighbours with 15 neighbours, and the base with shuffled training\n"
                 "   labels (the leak control), on 5 stratified folds and the identity check fold, seeds 1, 2 and 3,\n"
                 "   on the CPU. No earlier result chose any of these settings.",
    "reading": "Validation files only first: analyze ranks by balanced accuracy and compares each configuration with\n"
               "   the base (paired bootstrap over seeds and rows, Holm). The highest is selected with select. Then\n"
               "   test-report reads that configuration's test files, and test-compare runs family main of\n"
               "   round2_families.json on the test files: the selection against the base, and against k-nearest\n"
               "   neighbours, paired bootstrap with 2,000 draws and Holm's correction within the family.",
    "stop_condition": "The control must sit near chance (balanced accuracy near 0.5); if it is above 0.6, stop and\n"
                      "   check the split before reading any test file. The identity fold's validation and test numbers\n"
                      "   must agree. A selection that beats the base on test goes to confirmation on five new seeds.",
    "limits": "One synthetic dataset; three seeds (a screen, not a confirmation); the selection is optimistic until\n"
              "   confirmed on new seeds.",
}


class Runner:
    """Runs commands in the temporary copy, and keeps their output normalized for the page."""

    def __init__(self, root: Path, log):
        self.root = root
        self.log = log
        self.env = dict(os.environ, PYTHONPATH=str(root / "src"))
        self.env.pop("SWEEPKIT_REQUIRE", None)

    def normal(self, text: str) -> str:
        text = text.replace(str(self.root), "<repo>")
        text = re.sub(r"\b[0-9a-f]{64}\b", "<sha256>", text)
        text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?", "<utc>", text)
        text = re.sub(r"\d{8}T\d{6}(_\d{6})?Z", "<utc>", text)
        return text.rstrip("\n")

    def cli(self, *argv, expect: int = 0) -> str:
        self.log("  sweepkit " + " ".join(argv[:3]))
        done = subprocess.run([sys.executable, "-m", "sweepkit.cli", *argv], cwd=self.root, env=self.env,
                              capture_output=True, text=True, timeout=3600)
        if done.returncode != expect:
            raise RuntimeError(f"{argv[0]} exited {done.returncode} (expected {expect}):\n{done.stdout[-3000:]}\n"
                               f"{done.stderr[-3000:]}")
        return self.normal(done.stdout)

    def file(self, relative: str) -> str:
        return self.normal((self.root / relative).read_text(encoding="utf-8"))


def _block(text: str, language: str = "") -> str:
    return f"```{language}\n{text}\n```\n"


def _command(argv: str, output: str) -> str:
    return _block(f"$ python -m sweepkit.cli {argv}\n{output}")


def _copy_code(root: Path) -> None:
    from sweepkit.hashing import CODE_ITEMS

    source = paths.repo_root()
    for item in CODE_ITEMS:
        here = source / item
        if here.is_dir():
            shutil.copytree(here, root / item, ignore=shutil.ignore_patterns("__pycache__", "project"))
        elif here.is_file():
            shutil.copy(here, root / item)
    (root / "sweeps" / "project").mkdir(parents=True, exist_ok=True)
    # the example is the template's: a project's own name and discipline would change the page (a copy named
    # tmpltest failed the comparison on the node, 2026-10-27)
    (root / "project.json").write_text('{"project": "myproject", "discipline": "enforced"}\n', encoding="utf-8")


def _fill_rule(r: Runner, round_name: str, values: dict) -> None:
    path = r.root / "artifacts" / "analyses" / f"{round_name}_prep_{DATE}" / f"RULE_{round_name}.txt"
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    path.write_text(text, encoding="utf-8", newline="\n")


def _latest(r: Runner, parent: str, prefix: str) -> Path:
    found = sorted(p for p in (r.root / parent).iterdir() if p.name.startswith(prefix))
    return found[-1]


def _round(r: Runner, name: str, sweep: dict, rule: dict, question: str, families) -> list:
    out = []
    sweep_file = f"sweeps/project/{sweep['sweep_id']}.json"
    (r.root / sweep_file).write_text(json.dumps(sweep, indent=1) + "\n", encoding="utf-8", newline="\n")
    prep = f"artifacts/analyses/{name}_prep_{DATE}"
    out += ["### 1. Prepare", "",
            f"The sweep file `{sweep_file}` holds the base configuration and the stages that vary it. `check` "
            "expands it without queueing anything:", ""]
    checked = json.loads(r.cli("check", "--sweep-file", sweep_file))
    out.append(_block(json.dumps({k: checked[k] for k in ("sweep_id", "purpose", "configs", "jobs", "seeds")},
                                 indent=1), "json"))
    out += ["`smoke` runs every configuration once on a few rows, through the real runner:", ""]
    smoke = r.cli("smoke", "--sweep-file", sweep_file, "--device", "cpu", "--rows", "60")
    out.append(_command(f"smoke --sweep-file {sweep_file} --device cpu --rows 60",
                        "\n".join(line for line in smoke.splitlines()
                                  if line.startswith(("sweep ", "  stage ", "SMOKE", "NOTE")))))
    argv = (f'round-new --round {name} --date {DATE} --question "{question}" --sweep-files {sweep_file} '
            f'--workers-per-gpu 3 --gpus 0,1 --end-time "{END}" --environment main --profiles core unsupervised '
            f'--min-free-gb 15')
    made = r.cli("round-new", "--round", name, "--date", DATE, "--question", question, "--sweep-files", sweep_file,
                 "--workers-per-gpu", "3", "--gpus", "0,1", "--end-time", END, "--environment", "main",
                 "--profiles", "core", "unsupervised", "--min-free-gb", "15")
    fam_arg = "none"
    if families is not None:
        # the comparison families go in the preparation folder, and its README names every file of the folder
        fam_arg = f"{prep}/{name}_families.json"
        (r.root / fam_arg).write_text(json.dumps(families, indent=1) + "\n", encoding="utf-8", newline="\n")
        readme = r.root / prep / "README.txt"
        text = readme.read_text(encoding="utf-8")
        marker = "  sweep_files.sha256: the SHA-256 of each sweep file and of the plan at preparation.\n"
        readme.write_text(text.replace(marker, marker + f"  {name}_families.json: the comparison families "
                                       "test-compare runs, fixed with the RULE.\n", 1), encoding="utf-8",
                          newline="\n")
    out += ["`round-new` makes the round's preparation folder:", "", _command(argv, made),
            f"`{prep}/README.txt`" + (" (with the families file added to its list)" if families else "") + ":", "",
            _block(r.file(f"{prep}/README.txt")),
            "The lines the person pastes on a node, one file per step; the start line, "
            f"`{prep}/lines/03_start.txt`, for example:", "", _block(r.file(f"{prep}/lines/03_start.txt")),
            "### 2. Fix the RULE before any result exists", "",
            "The RULE's four sections are filled in (by Claude, approved by the person), then fixed:", ""]
    _fill_rule(r, name, rule)
    out.append(_block(r.file(f"{prep}/RULE_{name}.txt")))
    if families is not None:
        out += [f"The comparison families test-compare will run, `{fam_arg}`:", "", _block(r.file(fam_arg), "json")]
    fixed = r.cli("rule-fix", "--name", name, "--rule-file", f"{prep}/RULE_{name}.txt", "--sweeps",
                  sweep["sweep_id"], "--families", fam_arg, "--late-reason", "none")
    out.append(_command(f"rule-fix --name {name} --rule-file {prep}/RULE_{name}.txt --sweeps {sweep['sweep_id']} "
                        f"--families {fam_arg} --late-reason none", fixed))
    checked = r.cli("round-check", "--round", name, "--date", DATE)
    out += ["`round-check` passes before the first line is handed over:", "",
            _command(f"round-check --round {name} --date {DATE}", checked)]
    run = r.cli("run", "--sweep-file", sweep_file, "--device", "cpu", "--max-consecutive-failures", "3")
    out += ["### 3. Run", "",
            "On a node the lines smoke-test and enqueue the sweep and start the workers; here `run` does the same work "
            "in one process. Its last line:", "",
            _command(f"run --sweep-file {sweep_file} --device cpu --max-consecutive-failures 3",
                     run.splitlines()[-1])]
    return out


def _read_round1(r: Runner) -> list:
    out = ["### 4. Read the label-free files, then choose", ""]
    analyzed = r.cli("analyze", "--sweep", "round1_screen", "--confidence", "0.95", "--n-boot", "2000", "--seed", "1",
                     "--meaningful", "0.75", "--top", "5")
    out.append(_command("analyze --sweep round1_screen --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 "
                        "--top 5", analyzed))
    out += ["A difference can be detectable and still too small to matter: read the difference and its interval, "
            "not only the verdict.", ""]
    report = json.loads((_latest(r, "artifacts/analysis", "round1_screen_") / "report.json").read_text())
    ranked = [s for s in report["ranked"] if s.get("value") is not None]

    def stability(s):
        return (s.get("stability") or {}).get("value") or 0.0

    def overlap(a, b):
        return bool(a.get("interval") and b.get("interval") and a["interval"]["low"] <= b["interval"]["high"]
                    and b["interval"]["low"] <= a["interval"]["high"])

    # the RULE's reading: among the configurations whose intervals overlap the highest's, the most stable; one more
    # stable by over 0.05 that ranks below the choice stops the round (the page would then contradict its RULE, so
    # the example stops too)
    if not ranked:
        raise RuntimeError("the example round's analysis ranked no configuration")
    tied = [s for s in ranked if s is ranked[0] or overlap(ranked[0], s)]
    chosen = max(tied, key=stability)
    below = ranked[ranked.index(chosen) + 1:]
    if any(stability(s) > stability(chosen) + 0.05 for s in below):
        raise RuntimeError("the example round's RULE stops here: a configuration more stable by over 0.05 ranks below "
                           "its choice")
    clusters = (chosen.get("metrics") or {}).get("n_clusters")
    reason = "highest silhouette, read with its stability and number of clusters, as the RULE says"
    selected = r.cli("select", "--sweep", "round1_screen", "--config", chosen["config_hash"][:10], "--name",
                     "round1_choice", "--reason", reason)
    out += [f"The RULE says: the highest silhouette, read with its stability and the number of clusters it finds. "
            f"That is `{chosen['config_hash'][:12]}` ({chosen['stage']}: {chosen['label']}), stability "
            f"{stability(chosen):.3f}, "
            + (f"{clusters:.1f} clusters on average" if clusters is not None else "its number of clusters not recorded")
            + "; no configuration below it is more stable by over 0.05. `select` records it:", "",
            _command(f'select --sweep round1_screen --config {chosen["config_hash"][:10]} --name round1_choice '
                     f'--reason "{reason}"', selected),
            "### 5. Only now, the files that judge the choice", ""]
    revealed = r.cli("reveal-report", "--selection", "round1_choice", "--confidence", "0.95")
    out.append(_command("reveal-report --selection round1_choice --confidence 0.95", revealed))
    exported = r.cli("export", "--selection", "round1_choice", "--runs", "all", "--reveal", "no", "--plots", "no",
                     "--top-features", "3", "--formats", "csv")
    folder = _latest(r, "artifacts/exports", "round1_choice_")
    listing = "\n".join(sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()))
    out += ["`export` writes what the person takes home (plots left out here); its files:", "",
            _command("export --selection round1_choice --runs all --reveal no --plots no --top-features 3 "
                     "--formats csv", exported.splitlines()[-1]), _block(listing)]
    return out


def _read_round2(r: Runner) -> list:
    out = ["### 4. Read the validation files, then choose", ""]
    analyzed = r.cli("analyze", "--sweep", "round2_screen", "--confidence", "0.95", "--n-boot", "2000", "--seed", "1",
                     "--meaningful", "0.75", "--top", "5")
    out.append(_command("analyze --sweep round2_screen --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 "
                        "--top 5", analyzed))
    report = json.loads((_latest(r, "artifacts/analysis", "round2_screen_") / "report.json").read_text())
    chosen = report["ranked"][0]
    reason = "highest validation balanced accuracy, as the RULE says"
    selected = r.cli("select", "--sweep", "round2_screen", "--config", chosen["config_hash"][:10], "--name",
                     "round2_choice", "--reason", reason)
    out += [f"The highest validation balanced accuracy is `{chosen['config_hash'][:12]}` ({chosen['stage']}: "
            f"{chosen['label']}). `select` records it:", "",
            _command(f'select --sweep round2_screen --config {chosen["config_hash"][:10]} --name round2_choice '
                     f'--reason "{reason}"', selected),
            "### 5. Only now, the test files", ""]
    tested = r.cli("test-report", "--selection", "round2_choice", "--confidence", "0.95")
    out.append(_command("test-report --selection round2_choice --confidence 0.95", tested))
    rule = _latest(r, "artifacts/rules", "round2_").name
    compared = r.cli("test-compare", "--rule", rule, "--confidence", "0.95", "--n-boot", "2000", "--seed", "1",
                     "--meaningful", "0.75")
    out += ["`test-compare` runs exactly the families the RULE fixed:", "",
            _command(f"test-compare --rule {r.normal(rule)} --confidence 0.95 --n-boot 2000 --seed 1 "
                     "--meaningful 0.75", compared)]
    return out


def render(log) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        _copy_code(root)
        r = Runner(root, log)
        lines = [INTRO.rstrip("\n"), "",
                 "## Round 1: how many clusters, and which method (unsupervised)", ""]
        lines += _round(r, "round1", ROUND1_SWEEP, ROUND1_RULE,
                        "How many clusters do the data hold, and which method finds them best?", None)
        lines += _read_round1(r)
        lines += ["## Round 2: does a change beat the baseline (supervised)", ""]
        from sweepkit.sweep.design import expand

        jobs, summary = expand(ROUND2_SWEEP)
        knn = next(j["config_hash"] for j in jobs if j["label"] == "knn")[:10]
        families = {"families": [{"name": "main", "files": "test", "metric": "balanced_accuracy", "comparisons": [
            {"a": {"selection": "round2_choice"}, "b": {"sweep": "round2_screen",
                                                        "config": summary["base_config_hash"][:10]}},
            {"a": {"selection": "round2_choice"}, "b": {"sweep": "round2_screen", "config": knn}}]}]}
        lines += _round(r, "round2", ROUND2_SWEEP, ROUND2_RULE,
                        "Does a more regularized model or a random forest beat the baseline?", families)
        lines += _read_round2(r)
        lines += ["## After the reading", "",
                  "The reading folder `artifacts/analyses/<round>_<date>/` gets a README with what ran, what was read "
                  "when and what it found; findings.csv, VERIFICATION_LOG.md and STATE.md are updated; and a results "
                  "doc is written from `templates/results_doc.md` (docs/REPORTING.md). Round 1's choice goes to a "
                  "final run on every row (`sweeps/examples/cluster_final.json` shows one); round 2's to a "
                  "confirmation on five new seeds.", ""]
    return "\n".join(lines).rstrip("\n") + "\n"
