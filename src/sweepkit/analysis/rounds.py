"""A round's preparation folder, written from the templates, and the check of it before anything runs.

round-new makes artifacts/analyses/<round>_prep_<date>/ once:
    README.txt            the question, the sweeps (configurations and jobs from expand), the node's environment and
                          schedule, the plan's path, the cost (from the newest estimate of each sweep, when one
                          exists), the tree digest, and every file of the folder
    RULE_<round>.txt      the RULE, from templates/RULE.txt, with its four sections to fill in before rule-fix
    sweep_files.sha256    the SHA-256 of each sweep file and of the plan at preparation
    lines/                one file per step for the person who runs the node, from templates/lines/, with the
                          project, the round, the environment and its profiles, the tree digest, the sweeps and the
                          plan filled in
and the plan, sweeps/rounds/<round>_prep_<date>/plan.txt: the steps node/chain.sh runs on the node (smoke and enqueue,
then the GPU schedule), in the round's environment. The sweep files and the plan stay under sweeps/ because they
travel to the node with the code (artifacts/ does not), and the tree digest that lines 02 and 03 check covers them,
so the node runs exactly the plan that was prepared. The digest in the lines is taken after the plan is written.

round-check confirms, before the first line is handed over and again before reading:
    - the RULE was fixed with rule-fix (a record with this RULE's SHA-256 exists) and holds no unfilled section;
    - it was fixed before the first result of the round's sweeps (the earliest time in their runs' provenance);
    - every line names its shell, says what it should print and when to stop, is one command, and, when it is a
      bash line (the node), passes bash -n and logs its own output under logs/<round>/ (`| tee -a .../logs/...`);
    - the plan exists, passes bash -n line by line and has no carriage returns;
    - the sweep files and the plan are unchanged since preparation;
    - while the round has no result yet, the code's tree digest is still the one in the lines (after results exist,
      a change is only noted: the runs record the code they ran);
    - README.txt names every file of the folder.
"""
import datetime
import json
import re
import shutil
import subprocess
from pathlib import Path

from sweepkit import paths, project
from sweepkit.hashing import sha256_file, tree_digest
from sweepkit.runs.store import OutputFolder

PLACEHOLDER = re.compile(r"\{\{[a-z_]+\}\}")


class RoundError(ValueError):
    """A round's argument or file to correct; the command line prints it as one line."""


def prep_folder(round_name: str, date: str) -> Path:
    return paths.analyses_dir() / f"{round_name}_prep_{date}"


def plan_path(round_name: str, date: str) -> str:
    """The plan's path from the repository root. Its folder has the preparation folder's name, from which
    node/chain.sh takes the round's name for its logs."""
    return f"sweeps/rounds/{round_name}_prep_{date}/plan.txt"


END_TIME = re.compile(r"(?P<day>\d{4}-\d{2}-\d{2}) (?P<clock>\d{2}:\d{2})|now \+ (?P<n>\d{1,3}) (?P<unit>minutes|hours)")


def _check_end_time(end_time: str, date: str) -> None:
    """The schedule's end, which the node reads with `date -d` when the workers start: a date and a time on the
    round's date or at most 6 days after it, or 30 minutes to 7 days from then ("now + 45 minutes", "now + 10
    hours"). Other forms are
    refused: `date -d` reads relative words such as "tomorrow" when the plan's step runs, not when the round was
    prepared, and a form it cannot read would leave the workers with no stop time. The limit catches a mistyped month
    or year, which would leave the workers running for weeks; node/schedule.sh refuses an end more than 7 days after
    the start, so a date ends at most 6 days after the round's. The plan checks the schedule on the node before it
    enqueues anything (schedule.sh --check), which also refuses a time the node's clock skips."""
    max_days = 7                        # node/schedule.sh's limit too (SCHEDULE_MAX_DAYS, 7 unless set on the node)
    found = END_TIME.fullmatch(end_time.strip())
    if found is None:
        raise RoundError(f"--end-time {end_time!r}: \"YYYY-MM-DD HH:MM\" in the node's local time, for example "
                         f"\"2026-10-29 08:00\", or \"now + <n> minutes\" or \"now + <n> hours\" from the start")
    if found.group("day"):
        try:
            end = datetime.datetime.strptime(f"{found.group('day')} {found.group('clock')}", "%Y-%m-%d %H:%M")
        except ValueError:
            raise RoundError(f"--end-time {end_time!r}: not a date and time that exists") from None
        days = (end.date() - datetime.datetime.strptime(date, "%Y%m%d").date()).days
        if not 0 <= days < max_days:
            raise RoundError(f"--end-time {end_time!r}: on the round's date ({date}) or at most {max_days - 1} days "
                             f"after it (the node's schedule refuses an end more than {max_days} days after its start)")
    elif found.group("unit") == "minutes" and int(found.group("n")) < 30:
        raise RoundError(f"--end-time {end_time!r}: at least 30 minutes (the plan checks the schedule before the smoke "
                         f"tests, and node/schedule.sh --check wants the end that far ahead)")
    elif int(found.group("n")) < 1:
        raise RoundError(f"--end-time {end_time!r}: at least 1 hour")
    elif found.group("unit") == "hours" and int(found.group("n")) > max_days * 24:
        raise RoundError(f"--end-time {end_time!r}: at most {max_days * 24} hours ({max_days} days)")


def _end_phrase(end_time: str) -> str:
    """The end as the round's README says it, after "stops them": a time is the node's local time; a relative end
    counts from the plan's step 3, which reads it with `date -d` when it starts the workers (the dry run's README read
    "stops them at now + 45 minutes, the node's local time", 2026-10-30)."""
    found = END_TIME.fullmatch(end_time.strip())
    if found is not None and found.group("unit"):
        n, unit = int(found.group("n")), found.group("unit")
        return f"{n} {unit[:-1] if n == 1 else unit} after they start (the plan's step 3, once the smoke tests pass)"
    return f"at {end_time.strip()}, the node's local time"


def _check_names(round_name: str, date: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", round_name) or "__" in round_name:
        raise RoundError(f"round {round_name!r}: letters, digits, _ and - only")
    try:
        datetime.datetime.strptime(date, "%Y%m%d")
    except ValueError:
        raise RoundError(f"date {date!r}: YYYYMMDD, the round's date") from None


def render(text: str, values: dict) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def _latest_estimate(sweep_id: str):
    from sweepkit.analysis.records import latest

    folder, report = latest(paths.estimates_dir(), sweep_id, "report.json")
    return report


def round_new(round_name: str, date: str, question: str, sweep_files: list, workers_per_gpu: int, gpus: str,
              end_time: str, environment: str, profiles: list, min_free_gb: int) -> Path:
    from sweepkit.sweep.design import expand
    from sweepkit.sweep.enqueue import load_sweep_file

    _check_names(round_name, date)
    if not question.strip():
        raise RoundError("--question: say in one or two sentences what this round asks")
    if workers_per_gpu < 1 or not re.fullmatch(r"\d+(,\d+)*", gpus):
        raise RoundError("--workers-per-gpu at least 1; --gpus a list such as 0,1,2,3")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,23}", environment):
        raise RoundError(f"--environment {environment!r}: the name the environment was built with on the node "
                         f"(node/setup_env.sh <environment> ...), for example main or rapids")
    root = paths.repo_root()
    if not profiles:
        raise RoundError("--profiles: the profiles installed in that environment, for example core unsupervised")
    for profile in profiles:
        if not (root / "env" / f"requirements-{profile}.txt").is_file():
            raise RoundError(f"--profiles: no env/requirements-{profile}.txt (docs/PROFILES.md lists the profiles)")
    if min_free_gb < 1:
        raise RoundError("--min-free-gb: the free disk in GB the round keeps in the home folder, at least 1 (the floor "
                         "agreed for the project, PROJECT.md Compute)")
    _check_end_time(end_time, date)
    if prep_folder(round_name, date).exists():
        raise RoundError(f"{prep_folder(round_name, date)} exists: round-new makes a round once (a new round name "
                         f"or date makes a new preparation)")
    plan_rel = plan_path(round_name, date)
    if (root / plan_rel).exists():
        raise RoundError(f"{plan_rel} exists: round-new makes a round once (a new round name or date makes a new "
                         f"preparation)")
    rows, ids, digests, cost_lines, total_jobs = [], [], [], [], 0
    for f in sweep_files:
        path = (root / f).resolve() if not Path(f).is_absolute() else Path(f).resolve()
        if root.resolve() / "sweeps" not in path.parents:
            raise RoundError(f"{f}: a round's sweep files live under sweeps/ (they travel with the code to the node)")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", path.relative_to(root.resolve()).as_posix()):
            raise RoundError(f"{f}: letters, digits, _ . / and - only in a round's sweep file paths (the plan passes "
                             f"them to bash on the node)")
        sweep = load_sweep_file(path)
        jobs, summary = expand(sweep)
        rel = path.relative_to(root.resolve()).as_posix()
        # the plan starts workers on GPUs only, and a GPU worker takes the jobs whose fit.device is cuda or auto
        # (auto: the method's GPU path when it has one, otherwise the CPU); a job on cpu or mps would wait all night
        stuck = sorted({j["settings"]["fit"].get("device", "cpu") for j in jobs} - {"cuda", "auto"})
        if stuck:
            raise RoundError(f"{rel}: fit.device {stuck} in some jobs; a round's workers run on the node's GPUs and "
                             f"take jobs on cuda or auto only (auto runs a method without a GPU path on the CPU). "
                             f"Set fit.device to auto or cuda")
        ids.append(summary["sweep_id"])
        total_jobs += summary["jobs"]
        rows.append(f"  {rel}: sweep {summary['sweep_id']} ({summary['purpose']}), {summary['configs']} configurations, "
                    f"{summary['jobs']} jobs, seeds {summary['seeds']}")
        digests.append(f"{sha256_file(path)[1]}  {rel}")
        estimate = _latest_estimate(summary["sweep_id"])
        if estimate is None:
            cost_lines.append(f"{summary['sweep_id']}: not estimated yet (run estimate)")
        else:
            hours = estimate["per_job"]["seconds"] * summary["jobs"] / 3600
            cost_lines.append(f"{summary['sweep_id']}: about {hours:.1f} job-hours on {estimate['device']} "
                              f"({estimate['per_job']['seconds'] / 60:.1f} min a job from estimate)")
    if len(set(ids)) != len(ids):
        raise RoundError(f"two sweep files share a sweep_id: {ids}")
    name = project.name()
    folder_rel = f"artifacts/analyses/{round_name}_prep_{date}"
    templates = root / "templates"
    line_files = sorted((templates / "lines").glob("*.txt"))
    values = {"project": name, "round": round_name, "date": date, "question": question.strip(),
              "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "sweeps": "\n".join(rows), "cost": "; ".join(cost_lines) + f" ({total_jobs} jobs in all)",
              "prep_folder": folder_rel, "plan": plan_rel, "environment": environment,
              "profiles": " ".join(profiles), "sweep_files": " ".join(d.split("  ", 1)[1] for d in digests),
              "sweep_ids": " ".join(ids), "pack_paths": " ".join(f"runs/{i} sweeps/{i}" for i in ids),
              "workers_per_gpu": workers_per_gpu, "gpus": gpus, "gpus_pattern": gpus.replace(",", "|"),
              "end_time": end_time.strip(), "end_phrase": _end_phrase(end_time), "min_free_gb": min_free_gb,
              "lines": ", ".join(p.name for p in line_files), "rule_fixed": "not fixed yet"}
    # the plan first, under sweeps/, so that the tree digest written into the lines covers it
    plan_file = root / plan_rel
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_file, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(render((templates / "plan.txt").read_text(encoding="utf-8"), values))
    try:
        values["digest"] = tree_digest(root)
        digests.append(f"{sha256_file(plan_file)[1]}  {plan_rel}")
        folder = OutputFolder(paths.analyses_dir(), f"{round_name}_prep_{date}").begin()
    except BaseException:
        _remove_plan(plan_file)
        raise
    try:
        folder.write_text(f"RULE_{round_name}.txt", render((templates / "RULE.txt").read_text(encoding="utf-8"), values))
        folder.write_text("README.txt", render((templates / "round_README.txt").read_text(encoding="utf-8"), values))
        folder.write_text("sweep_files.sha256", "\n".join(digests) + "\n")
        for p in line_files:
            folder.write_text(f"lines/{p.name}", render(p.read_text(encoding="utf-8"), values))
    except BaseException as err:
        folder.fail(err)
        _remove_plan(plan_file)
        raise
    return folder.complete()


def _remove_plan(plan_file: Path) -> None:
    """A preparation that failed leaves no plan behind, so the same round and date can be prepared again."""
    plan_file.unlink(missing_ok=True)
    try:
        plan_file.parent.rmdir()
    except OSError:
        pass


def _bash_ok(command: str) -> tuple:
    if shutil.which("bash") is None:
        return None, "bash is not installed here"
    done = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
    return done.returncode == 0, done.stderr.strip()


def _first_result_utc(sweep_ids: list):
    from sweepkit.analysis.records import read_json, run_folders

    times = []
    for sweep_id in sweep_ids:
        for folder in run_folders(sweep_id)["complete"]:
            times.append(read_json(folder, "provenance.json", "choose")["utc"])
    return min(times) if times else None


def round_check(round_name: str, date: str) -> dict:
    from sweepkit.analysis.rules import all_rules

    _check_names(round_name, date)
    folder = prep_folder(round_name, date)
    if not folder.is_dir():
        raise FileNotFoundError(f"no folder {folder}; round-new makes it")
    problems, notes = [], []
    rule_path = folder / f"RULE_{round_name}.txt"
    rule_text = rule_path.read_text(encoding="utf-8")
    if PLACEHOLDER.search(rule_text):
        problems.append(f"{rule_path.name} still holds unfilled sections: {sorted(set(PLACEHOLDER.findall(rule_text)))}")
    _, rule_sha = sha256_file(rule_path)
    records = [r for _, r in all_rules() if r["rule_sha256"] == rule_sha]
    root = paths.repo_root()
    listed = [line.split("  ", 1) for line in (folder / "sweep_files.sha256").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    plan_rel = plan_path(round_name, date)
    sweep_ids = []
    for digest, rel in listed:
        what = "the plan" if rel == plan_rel else "sweep file"
        path = root / rel
        if not path.is_file():
            problems.append(f"{what} {rel} is missing")
            continue
        if sha256_file(path)[1] != digest:
            problems.append(f"{what} {rel} changed since preparation")
        if rel != plan_rel:
            sweep_ids.append(json.loads(path.read_text(encoding="utf-8"))["sweep_id"])
    first = _first_result_utc(sweep_ids)
    if plan_rel not in [rel for _, rel in listed]:
        problems.append(f"sweep_files.sha256 does not list the plan {plan_rel}: prepare the round again with round-new")
    elif (root / plan_rel).is_file():
        plan_bytes = (root / plan_rel).read_bytes()
        if b"\r" in plan_bytes:
            problems.append(f"{plan_rel} holds carriage returns (saved with Windows line endings); bash on the node "
                            f"would read them as part of each command")
        for line in plan_bytes.decode("utf-8", errors="replace").splitlines():
            if line.strip() and not line.startswith("#"):
                ok, err = _bash_ok(line)
                if ok is False:
                    problems.append(f"{plan_rel} fails bash -n: {err}")
    readme_text = (folder / "README.txt").read_text(encoding="utf-8")
    expected = re.search(r"tree digest ([0-9a-f]{64})", readme_text)
    if expected is None:
        problems.append("README.txt names no tree digest")
    else:
        now = tree_digest(root)
        if now != expected.group(1):
            if first is None:
                problems.append(f"the code changed since preparation: its tree digest is {now[:12]}..., the lines "
                                f"expect {expected.group(1)[:12]}... Undo the change, or prepare the round again "
                                f"(round-new with a new round name or date)")
            else:
                notes.append(f"the code changed since preparation (tree digest {now[:12]}..., the lines named "
                             f"{expected.group(1)[:12]}...); the runs record the code they ran")
    if not records:
        problems.append(f"{rule_path.name} was not fixed with rule-fix (no record has its SHA-256 {rule_sha[:12]})")
    else:
        fixed = min(r["fixed_utc"] for r in records)
        covered = set().union(*(set(r["sweeps"]) for r in records))
        missing = sorted(set(sweep_ids) - covered)
        if missing:
            problems.append(f"the RULE's record does not cover sweeps {missing}")
        if first is not None and not fixed < first:
            reasons = sorted({r["late_reason"] for r in records if r.get("late_reason")})
            problems.append(f"the RULE was fixed at {fixed}, after the first result of the round ({first})"
                            + (f"; the reason recorded: {'; '.join(reasons)}" if reasons else ""))
        notes.append(f"RULE fixed {fixed}" + (f", first result {first}" if first else ", no result yet"))
    for p in sorted((folder / "lines").glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        comments = [line for line in text.splitlines() if line.startswith("#")]
        commands = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
        if not any(c.startswith("# expect:") for c in comments):
            problems.append(f"lines/{p.name} does not say what it should print (# expect:)")
        if not any(c.startswith("# stop if:") for c in comments):
            problems.append(f"lines/{p.name} does not say when to stop (# stop if:)")
        shell = next((c for c in comments if c.startswith("# shell:")), None)
        if shell is None:
            problems.append(f"lines/{p.name} does not name its shell (# shell:)")
        if len(commands) != 1:
            problems.append(f"lines/{p.name} holds {len(commands)} command lines; a step is one line")
            continue
        if PLACEHOLDER.search(text):
            problems.append(f"lines/{p.name} still holds {sorted(set(PLACEHOLDER.findall(text)))}")
        if shell and "bash" in shell.split(";")[0]:
            if not re.search(r"\|\s*tee -a \S*logs/", commands[0]):
                problems.append(f"lines/{p.name} does not log its output under logs/<round>/ (end it with "
                                f"2>&1 | tee -a logs/<round>/<its name>.txt)")
            ok, err = _bash_ok(commands[0])
            if ok is False:
                problems.append(f"lines/{p.name} fails bash -n: {err}")
            elif ok is None:
                notes.append(f"lines/{p.name} not checked: {err}")
        if "<" in commands[0] and ">" in commands[0] and re.search(r"<[a-z][^>]*>", commands[0]):
            notes.append(f"lines/{p.name} holds a part to fill in when the line is handed over")
    readme = readme_text
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name != "README.txt" and p.parent.name != "lines" and p.name not in readme:
            problems.append(f"README.txt does not name {p.relative_to(folder).as_posix()}")
    return {"round": round_name, "date": date, "folder": str(folder), "ok": not problems, "problems": problems,
            "notes": notes}


def render_check(report: dict) -> str:
    lines = [f"round-check {report['round']} ({report['folder']}):"]
    lines += [f"  PROBLEM {p}" for p in report["problems"]]
    lines += [f"  note: {n}" for n in report["notes"]]
    lines.append("ROUND OK" if report["ok"] else f"ROUND NOT READY ({len(report['problems'])} problems)")
    return "\n".join(lines)
