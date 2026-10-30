"""RULE files: which results will be read, and how, fixed before they exist.

rule-fix stores a RULE text once, with the UTC time it was fixed and its SHA-256:

    artifacts/rules/<name>_<utc>/RULE.txt        the text, byte for byte
    artifacts/rules/<name>_<utc>/record.json     name, fixed_utc, rule_sha256, sweeps it covers, comparison families,
                                                 runs_at_fix (complete runs of each covered sweep when it was fixed),
                                                 late_reason, warning
    artifacts/rules/<name>_<utc>/families.json   the families file as given (when there is one)
    artifacts/rules/<name>_<utc>/provenance.json

The comparison families (optional) are machine-readable, so test-compare runs exactly them:

    {"families": [
       {"name": "main", "files": "test", "metric": "balanced_accuracy",
        "comparisons": [{"a": {"selection": "final_cnn"}, "b": {"sweep": "night5_screen", "config": "3fa9c2d1"}}]}
    ]}

files is "test" (supervised: metrics_test.json and predictions_test.csv) or "reveal" (unsupervised: reveal.json).
A side is a recorded selection, or a sweep and the first 8 or more characters of a configuration hash.

The discipline (project.json "discipline"):
  - rule-fix records how many complete runs of each covered sweep exist when the RULE is fixed. A RULE is fixed
    before the results it governs, so when some exist it refuses unless a reason is given (--late-reason), which is
    recorded, and every report that uses the RULE prints "RULE <name> was fixed after <n> runs of <sweep> existed"
    with it. A confirmation sweep with its own RULE fixed before its runs is not affected;
  - test-report and reveal-report read the judging files of a selection only when a RULE that covers the selection's
    sweep was fixed before the selection was recorded;
  - test-compare runs a RULE's families only when no configuration they name had its judging files read (by any
    report or comparison) before the RULE was fixed.
Every judging read writes reading.json beside its report, so these checks look at the record, not at memory.
"enforced" refuses when a check fails; "advisory" runs anyway and puts a warning at the top of the report.
"""
import datetime
import json
from pathlib import Path

from sweepkit import paths, project
from sweepkit.hashing import sha256_file
from sweepkit.registry import METRICS

FAMILY_KEYS = ("name", "files", "metric", "comparisons")


class DisciplineError(PermissionError):
    pass


def now_utc() -> str:
    """UTC time to the microsecond, so two records made in the same second still have an order."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def _read(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def check_ref(ref, where: str) -> None:
    if type(ref) is not dict:
        raise ValueError(f"{where}: a side is {{\"selection\": name}} or {{\"sweep\": id, \"config\": hash prefix}}")
    if set(ref) == {"selection"}:
        if type(ref["selection"]) is not str or not ref["selection"]:
            raise ValueError(f"{where}: selection must be a name")
        return
    if set(ref) == {"sweep", "config"}:
        if type(ref["sweep"]) is not str or not ref["sweep"]:
            raise ValueError(f"{where}: sweep must be a sweep ID")
        config = ref["config"]
        if type(config) is not str or len(config) < 8 or any(c not in "0123456789abcdef" for c in config):
            raise ValueError(f"{where}: config must be at least 8 characters of a configuration hash")
        return
    raise ValueError(f"{where}: a side is {{\"selection\": name}} or {{\"sweep\": id, \"config\": hash prefix}}, "
                     f"got keys {sorted(ref)}")


def check_families(data) -> list:
    if type(data) is not dict or set(data) != {"families"} or type(data["families"]) is not list or not data["families"]:
        raise ValueError('a families file holds {"families": [...]} with at least one family')
    names = []
    for i, family in enumerate(data["families"]):
        where = f"families[{i}]"
        if type(family) is not dict or set(family) != set(FAMILY_KEYS):
            raise ValueError(f"{where}: needs exactly {list(FAMILY_KEYS)}")
        if type(family["name"]) is not str or not family["name"] or family["name"] in names:
            raise ValueError(f"{where}: a name, different from every other family's")
        names.append(family["name"])
        if family["files"] not in ("test", "reveal"):
            raise ValueError(f"{where}: files is test (supervised) or reveal (unsupervised)")
        entry = METRICS.get(family["metric"])
        if family["files"] == "reveal" and entry["label_free"]:
            raise ValueError(f"{where}: {family['metric']} is a label-free criterion; reveal families compare "
                             "agreement with the reference labels")
        if type(family["comparisons"]) is not list or not family["comparisons"]:
            raise ValueError(f"{where}: comparisons is a non-empty list")
        for k, pair in enumerate(family["comparisons"]):
            if type(pair) is not dict or set(pair) != {"a", "b"}:
                raise ValueError(f"{where}.comparisons[{k}]: needs exactly a and b")
            check_ref(pair["a"], f"{where}.comparisons[{k}].a")
            check_ref(pair["b"], f"{where}.comparisons[{k}].b")
            if pair["a"] == pair["b"]:
                raise ValueError(f"{where}.comparisons[{k}]: a and b are the same")
    return data["families"]


def late_note(record: dict) -> str | None:
    """The line every report that uses this RULE prints when it was fixed after runs of its sweeps existed."""
    late = {s: n for s, n in (record.get("runs_at_fix") or {}).items() if n}
    if not late:
        return None
    what = ", ".join(f"{n} runs of {s}" for s, n in late.items())
    why = f": {record['late_reason']}" if record.get("late_reason") else " (no reason given; advisory discipline)"
    return f"NOTE RULE {record['name']} was fixed after {what} existed{why}"


def rule_fix(name: str, rule_file: Path, sweeps: list, families_file: Path | None, late_reason: str | None) -> Path:
    """Store a RULE. Returns its folder. late_reason: why it is fixed after runs of its sweeps exist, or None."""
    from sweepkit.analysis.records import run_folders
    from sweepkit.runs.provenance import collect
    from sweepkit.runs.store import OutputFolder

    if not name or not all(c.isalnum() or c in "_-" for c in name) or "__" in name:
        raise ValueError(f"rule name {name!r}: letters, digits, _ and - only")
    if not sweeps or len(set(sweeps)) != len(sweeps) or any(not s for s in sweeps):
        raise ValueError("name every sweep the RULE covers, once each")
    text = Path(rule_file).read_bytes()
    if not text.strip():
        raise ValueError(f"{rule_file} is empty")
    if b"{{" in text:
        blanks = [f"line {n}: {line.strip()[:60]}" for n, line in
                  enumerate(text.decode("utf-8", errors="replace").splitlines(), 1) if "{{" in line]
        raise DisciplineError(f"{rule_file} still has blanks of the template, {{{{...}}}}: fill in every one (the "
                              f"round's and the project's names and the four sections), then fix it. "
                              + "; ".join(blanks))
    families, families_bytes = [], None
    if families_file is not None:
        families_bytes = Path(families_file).read_bytes()
        families = check_families(json.loads(families_bytes.decode("utf-8")))
    existing = {s: len(run_folders(s)["complete"]) for s in sweeps}
    late = {s: n for s, n in existing.items() if n}
    reason = (late_reason or "").strip() or None
    warning = None
    if late and reason is None:
        what = ", ".join(f"{n} runs of {s}" for s, n in late.items())
        warnings = enforce([f"RULE {name} would be fixed after {what} exist; a RULE is fixed before the results it "
                            "governs. If it is fixed late on purpose (a RULE for test-compare that also names an "
                            "earlier sweep's configurations, for example), give the reason with --late-reason: it is "
                            "recorded and printed with every report that uses this RULE"], "rule-fix")
        warning = warnings[0]
    fixed = now_utc()
    stamp = fixed.replace("-", "").replace(":", "").replace(".", "_").replace("+0000", "Z")
    folder = OutputFolder(paths.rules_dir(), f"{name}_{stamp}").begin()
    try:
        folder.write_bytes("RULE.txt", text)
        if families_bytes is not None:
            folder.write_bytes("families.json", families_bytes)
        _, rule_sha = sha256_file(folder.tmp / "RULE.txt")
        record = {"name": name, "fixed_utc": fixed, "rule_sha256": rule_sha, "sweeps": list(sweeps),
                  "families": families,
                  "families_sha256": sha256_file(folder.tmp / "families.json")[1] if families_bytes else None,
                  "runs_at_fix": existing, "late_reason": reason if late else None, "warning": warning}
        folder.write_json("record.json", record)
        folder.write_json("provenance.json", collect("sweepkit.cli rule-fix"))
    except BaseException as err:
        folder.fail(err)
        raise
    return folder.complete()


def all_rules() -> list:
    """(folder, record) of every complete RULE, oldest first."""
    root = paths.rules_dir()
    if not root.is_dir():
        return []
    out = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and not folder.name.endswith(".tmp") and ".failed-" not in folder.name \
                and (folder / "record.json").is_file():
            out.append((folder, _read(folder / "record.json")))
    out.sort(key=lambda fr: fr[1]["fixed_utc"])
    return out


def find_rule(folder_name: str) -> tuple:
    folder = paths.rules_dir() / folder_name
    if not (folder / "record.json").is_file():
        raise FileNotFoundError(f"no RULE folder {folder_name!r} under artifacts/rules/ (give the folder name "
                                "rule-fix printed)")
    record = _read(folder / "record.json")
    _, sha = sha256_file(folder / "RULE.txt")
    if sha != record["rule_sha256"]:
        raise ValueError(f"{folder_name}: RULE.txt no longer matches the SHA-256 recorded when it was fixed")
    return folder, record


def covering_rule(sweep_id: str, before_utc: str):
    """(folder, record) of the newest RULE that covers the sweep and was fixed before the time, or (None, None)."""
    found = [(f, r) for f, r in all_rules() if sweep_id in r["sweeps"] and r["fixed_utc"] < before_utc]
    return found[-1] if found else (None, None)


def readings() -> list:
    """Every recorded reading of judging files: (folder, reading.json). test-report, reveal-report and test-compare
    record one beside their report; export and cluster-report with --reveal yes beside their export."""
    out = []
    for root in (paths.test_reports_dir(), paths.reveal_reports_dir(), paths.comparisons_dir(), paths.exports_dir()):
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and not folder.name.endswith(".tmp") and (folder / "reading.json").is_file():
                out.append((folder, _read(folder / "reading.json")))
    return out


def earlier_readings(configs: set, before_utc: str) -> list:
    """Readings of any of the configurations ((sweep_id, config_hash) pairs) made before the time."""
    out = []
    for folder, reading in readings():
        read = {(c["sweep_id"], c["config_hash"]) for c in reading["configs"]}
        if reading["utc"] < before_utc and read & configs:
            out.append(f"{folder.parent.name}/{folder.name} ({reading['utc']})")
    return out


def enforce(problems: list, what: str) -> list:
    """Raise under the enforced discipline; return warnings under the advisory one."""
    if not problems:
        return []
    if project.discipline() == "enforced":
        raise DisciplineError(f"{what} refused: " + "; ".join(problems) + ". (A project that decided so can set "
                              "\"discipline\": \"advisory\" in project.json instead.)")
    return [f"WARNING (advisory discipline): {p}" for p in problems]


def reading_record(kind: str, configs: list, rule: str | None, selection: str | None, utc: str) -> dict:
    return {"kind": kind, "utc": utc, "configs": configs, "rule": rule, "selection": selection}
