"""round-new and round-check: the preparation folder from the templates, the plan that travels with the code, and
every way round-check refuses one."""
import contextlib
import io
import json
import shutil
import time
import unittest
from unittest import mock

from tests.support import REPO, TempRepo, load_catalog
from tests.sweep_fixtures import cluster_base, sweep


def cli(*argv) -> int:
    from sweepkit import cli as module

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return module.main(list(argv))


FILLED = {"{{what_runs}}": "sweep r1_k, kmeans on blobs, seeds 1 to 3.",
          "{{reading}}": "analyze on label-free criteria, select, then reveal-report.",
          "{{stop_condition}}": "a configuration below the base on silhouette is dropped.",
          "{{limits}}": "one synthetic dataset."}


class TestRounds(unittest.TestCase):
    def setUp(self):
        load_catalog()
        self.repo = TempRepo()
        self.root = self.repo.__enter__()
        shutil.copytree(REPO / "templates", self.root / "templates")
        shutil.copytree(REPO / "env", self.root / "env")
        (self.root / "sweeps").mkdir()
        base = cluster_base()
        base["fit"]["device"] = "auto"                       # a node round's GPU workers take cuda and auto jobs
        s = sweep(base, [{"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]}], sweep_id="r1_k",
                  purpose="learning")
        (self.root / "sweeps" / "r1_k.json").write_text(json.dumps(s), encoding="utf-8")
        self.assertEqual(self.new("r1"), 0)
        self.folder = self.root / "artifacts" / "analyses" / "r1_prep_20261028"
        self.plan = self.root / "sweeps" / "rounds" / "r1_prep_20261028" / "plan.txt"

    def new(self, name, environment="main", profiles=("core", "unsupervised"), date="20261028"):
        return cli("round-new", "--round", name, "--date", date, "--question", "Does k-means find the blobs?",
                   "--sweep-files", "sweeps/r1_k.json", "--workers-per-gpu", "3", "--gpus", "0,1", "--end-time",
                   "2026-10-29 08:00", "--environment", environment, "--profiles", *profiles, "--min-free-gb", "15")

    def tearDown(self):
        self.repo.__exit__(None, None, None)

    def check(self):
        from sweepkit.analysis.rounds import round_check

        return round_check("r1", "20261028")

    def test_an_unfilled_rule_is_refused(self):
        rule = self.folder / "RULE_r1.txt"
        self.assertIn("{{what_runs}}", rule.read_text(encoding="utf-8"))
        from sweepkit.analysis.rules import DisciplineError, rule_fix

        # through the command: a refusal (REFUSED and exit 1, no traceback)
        self.assertEqual(cli("rule-fix", "--name", "r1", "--rule-file", str(rule), "--sweeps", "r1_k", "--families",
                             "none", "--late-reason", "none"), 1)
        with self.assertRaisesRegex(DisciplineError, "still has blanks of the template.*line 1"):
            rule_fix("r1", rule, ["r1_k"], None, None)
        self.assertFalse((self.root / "artifacts" / "rules").is_dir() and any(
            (self.root / "artifacts" / "rules").iterdir()))

    def fill_and_fix(self, late_reason="none", expect=0):
        rule = self.folder / "RULE_r1.txt"
        text = rule.read_text(encoding="utf-8")
        for key, value in FILLED.items():
            text = text.replace(key, value)
        rule.write_text(text, encoding="utf-8")
        self.assertEqual(cli("rule-fix", "--name", "r1", "--rule-file", str(rule), "--sweeps", "r1_k", "--families",
                             "none", "--late-reason", late_reason), expect)

    def test_files_from_the_templates(self):
        names = sorted(p.relative_to(self.folder).as_posix() for p in self.folder.rglob("*") if p.is_file())
        self.assertEqual(names, ["README.txt", "RULE_r1.txt", "lines/01_push.txt", "lines/02_install_test.txt",
                                 "lines/03_start.txt", "lines/04_status.txt", "lines/05_pack.txt", "lines/06_pull.txt",
                                 "sweep_files.sha256"])
        readme = (self.folder / "README.txt").read_text(encoding="utf-8")
        self.assertIn("sweeps/r1_k.json: sweep r1_k (learning), 1 configurations, 4 jobs", readme)
        self.assertIn("the environment main (~/venvs/testproject-main, profiles core unsupervised)", readme)
        start = (self.folder / "lines" / "03_start.txt").read_text(encoding="utf-8")
        self.assertIn("testproject_r1", start)
        self.assertNotIn("{{", start)
        self.assertEqual(self.new("r1"), 1)                  # a round is prepared once

    def test_the_plan_travels_with_the_code_and_the_lines_name_its_environment(self):
        from sweepkit.analysis.rounds import plan_path
        from sweepkit.hashing import sha256_file, tree_digest

        self.assertEqual(plan_path("r1", "20261028"), "sweeps/rounds/r1_prep_20261028/plan.txt")
        plan = self.plan.read_bytes().decode("utf-8")
        self.assertNotIn("\r", plan)
        self.assertNotIn("{{", plan)
        steps = [line for line in plan.splitlines() if line and not line.startswith("#")]
        schedule = '3 "$(date +%s):0,1" "$(date -d \'2026-10-29 08:00\' +%s):end"'
        # the schedule is checked before anything is enqueued, so a refused end leaves no jobs without workers
        self.assertEqual(steps, [f"env ENVIRONMENT=main MIN_FREE_GB=15 bash node/schedule.sh --check {schedule}",
                                 "env ENVIRONMENT=main SMOKE_DEVICE=cuda SMOKE_GPUS=0,1 bash node/smoke_enqueue.sh "
                                 "sweeps/r1_k.json",
                                 f"env ENVIRONMENT=main MIN_FREE_GB=15 bash node/schedule.sh {schedule}"])
        self.assertIn("bash node/disk_check.sh 15 > /dev/null", (self.folder / "lines" / "03_start.txt").read_text())
        lines = {p.name: p.read_text(encoding="utf-8") for p in (self.folder / "lines").glob("*.txt")}
        self.assertIn('"bash node/chain.sh sweeps/rounds/r1_prep_20261028/plan.txt"', lines["03_start.txt"])
        self.assertIn("env -u CUDA_VISIBLE_DEVICES CHECK_GPUS=0,1 bash ~/testproject/node/check.sh main core unsupervised "
                      "2>&1", lines["02_install_test.txt"])
        self.assertIn('C=${PIPESTATUS[0]}; echo "check.sh exit $C"', lines["02_install_test.txt"])
        self.assertIn("G=$(bash node/free_gpus.sh 2>/dev/null | grep -xE '0|1' | paste -sd, -)", lines["03_start.txt"])
        self.assertIn('env ENVIRONMENT=main bash node/run.sh status --sweep "$s"', lines["04_status.txt"])
        # the digest in the lines is the code's with the plan in it, and the plan's SHA-256 is listed
        digest = tree_digest(self.root)
        for name in ("01_push.txt", "02_install_test.txt", "03_start.txt"):
            self.assertIn(digest, lines[name], name)
        listed = (self.folder / "sweep_files.sha256").read_text(encoding="utf-8")
        self.assertIn(f"{sha256_file(self.plan)[1]}  sweeps/rounds/r1_prep_20261028/plan.txt", listed)
        self.plan.unlink()
        self.assertNotEqual(tree_digest(self.root), digest)

    def test_round_new_refuses_a_bad_environment_or_profile_and_leaves_nothing(self):
        self.assertEqual(self.new("r2", environment="Main!"), 1)
        self.assertEqual(self.new("r2", profiles=("core", "nosuchprofile")), 1)
        self.assertFalse((self.root / "artifacts" / "analyses" / "r2_prep_20261028").exists())
        self.assertFalse((self.root / "sweeps" / "rounds" / "r2_prep_20261028").exists())
        self.assertEqual(self.new("r2", environment="rapids", profiles=("rapids",)), 0)
        self.assertIn("node/check.sh rapids rapids 2>&1", (self.root / "artifacts" / "analyses" / "r2_prep_20261028" /
                                                          "lines" / "02_install_test.txt").read_text(encoding="utf-8"))

    def test_end_times_and_paths_the_node_could_misread_are_refused(self):
        from sweepkit.analysis.rounds import RoundError, round_new

        def new(end, files=("sweeps/r1_k.json",), name="r5"):
            return round_new(name, "20261028", "q", list(files), 3, "0", end, "main", ["core"], 15)

        for bad in ("tomorrow 08:00", "29/10/2026 08:00", "2026-02-30 08:00", "now + 0 minutes", "08:00",
                    "now + 20 minutes", "now + 0 hours",
                    "2026-10-29 08:00'; rm -rf ~; '", "now + 2 days",
                    # a mistyped month or year: before the round's date, or more than 6 days after it (the node's
                    # schedule refuses an end more than 7 days after its start)
                    "2025-10-29 08:00", "2026-10-27 23:00", "2026-11-29 08:00", "2026-11-04 00:00", "now + 169 hours"):
            with self.subTest(end=bad), self.assertRaises(RoundError):
                new(bad)
        for good in ("2026-10-28 23:00", "2026-11-03 23:59", "now + 168 hours", "now + 999 minutes", "now + 30 minutes",
                     "now + 1 hours"):
            with self.subTest(end=good):
                from sweepkit.analysis.rounds import _check_end_time
                _check_end_time(good, "20261028")
        (self.root / "sweeps" / "round one.json").write_text(
            (self.root / "sweeps" / "r1_k.json").read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaisesRegex(RoundError, "letters, digits"):
            new("2026-10-29 08:00", files=("sweeps/round one.json",))
        self.assertFalse((self.root / "sweeps" / "rounds" / "r5_prep_20261028").exists())
        new("now + 45 minutes")
        new("2026-10-29 08:00", name="r6")
        plan = (self.root / "sweeps" / "rounds" / "r5_prep_20261028" / "plan.txt").read_text()
        self.assertIn("\"$(date -d 'now + 45 minutes' +%s):end\"", plan)

    def test_a_sweep_with_jobs_only_a_cpu_worker_would_take_is_refused(self):
        s = json.loads((self.root / "sweeps" / "r1_k.json").read_text(encoding="utf-8"))
        s["base"]["fit"]["device"] = "cpu"
        (self.root / "sweeps" / "r1_k.json").write_text(json.dumps(s), encoding="utf-8")
        from sweepkit.analysis.rounds import RoundError, round_new

        with self.assertRaisesRegex(RoundError, r"fit.device \['cpu'\].*auto or cuda"):
            round_new("r4", "20261028", "q", ["sweeps/r1_k.json"], 3, "0", "2026-10-29 08:00", "main", ["core"],
                      15)
        self.assertFalse((self.root / "sweeps" / "rounds" / "r4_prep_20261028").exists())

    def test_a_preparation_that_fails_leaves_no_plan(self):
        from sweepkit.analysis.rounds import round_new
        from sweepkit.runs.store import OutputFolder

        with mock.patch.object(OutputFolder, "write_text", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                round_new("r3", "20261028", "Does k-means find the blobs?", ["sweeps/r1_k.json"], 3, "0,1",
                          "2026-10-29 08:00", "main", ["core", "unsupervised"], 15)
        self.assertFalse((self.root / "sweeps" / "rounds" / "r3_prep_20261028").exists())
        self.assertFalse((self.root / "artifacts" / "analyses" / "r3_prep_20261028").exists())
        self.assertEqual(self.new("r3"), 0)                  # so the same round can be prepared again

    def test_a_changed_plan_or_code_before_any_result_is_refused(self):
        self.fill_and_fix()
        self.plan.write_bytes(self.plan.read_bytes().replace(b"\n", b"\r\n"))
        problems = "\n".join(self.check()["problems"])
        self.assertIn("the plan sweeps/rounds/r1_prep_20261028/plan.txt changed since preparation", problems)
        self.assertIn("holds carriage returns", problems)
        self.assertIn("the code changed since preparation", problems)

    def test_code_changed_after_the_first_result_is_only_noted(self):
        self.fill_and_fix()
        # a run records its start to the second and the RULE its fixing to the microsecond, and a run in the same
        # second as the fixing counts as not after it; a real round's runs come long after
        time.sleep(1.1)
        self.assertEqual(cli("run", "--sweep-file", str(self.root / "sweeps" / "r1_k.json"), "--device", "cpu",
                             "--max-consecutive-failures", "1"), 0)
        (self.root / "sweeps" / "r2_next.json").write_text("{}", encoding="utf-8")     # the next round's sweep
        report = self.check()
        self.assertEqual(report["problems"], [])
        self.assertTrue(any("the code changed since preparation" in n for n in report["notes"]))

    def test_unfilled_and_unfixed_rule_is_not_ready(self):
        report = self.check()
        self.assertFalse(report["ok"])
        self.assertTrue(any("unfilled" in p for p in report["problems"]))
        self.assertTrue(any("not fixed with rule-fix" in p for p in report["problems"]))

    def test_ready_after_filling_and_fixing(self):
        self.fill_and_fix()
        report = self.check()
        self.assertEqual(report["problems"], [])
        self.assertTrue(report["ok"])
        self.assertEqual(cli("round-check", "--round", "r1", "--date", "20261028"), 0)

    def test_rule_fixed_after_the_first_result_is_refused(self):
        self.assertEqual(cli("run", "--sweep-file", str(self.root / "sweeps" / "r1_k.json"), "--device", "cpu",
                             "--max-consecutive-failures", "1"), 0)
        self.fill_and_fix(expect=1)               # runs of r1_k exist: refused without a reason
        self.fill_and_fix("written after a first look at the screen")
        report = self.check()
        self.assertTrue(any("after the first result" in p and "written after a first look" in p
                            for p in report["problems"]))

    def test_bad_line_changed_sweep_and_unlisted_file(self):
        self.fill_and_fix()
        line = self.folder / "lines" / "04_status.txt"
        text = line.read_text(encoding="utf-8").replace("# expect:", "# hope:")
        line.write_text(text + "echo a second command\n", encoding="utf-8")
        (self.folder / "lines" / "07_broken.txt").write_text(
            "# 7: broken\n# shell: bash on the node\n# expect: nothing\n# stop if: anything\ncd ~ && if [ 1 = 1 ]; then echo x\n",
            encoding="utf-8")
        (self.folder / "notes.txt").write_text("not named in the README", encoding="utf-8")
        s = json.loads((self.root / "sweeps" / "r1_k.json").read_text())
        s["description"] = "changed after preparation"
        (self.root / "sweeps" / "r1_k.json").write_text(json.dumps(s), encoding="utf-8")
        problems = "\n".join(self.check()["problems"])
        self.assertIn("lines/04_status.txt does not say what it should print", problems)
        self.assertIn("lines/04_status.txt holds 2 command lines", problems)
        if shutil.which("bash"):
            self.assertIn("lines/07_broken.txt fails bash -n", problems)
        self.assertIn("lines/07_broken.txt does not log its output under logs/<round>/", problems)
        self.assertNotIn("lines/03_start.txt does not log", problems)       # the template's own lines log theirs
        self.assertIn("sweep file sweeps/r1_k.json changed since preparation", problems)
        self.assertIn("README.txt does not name notes.txt", problems)


if __name__ == "__main__":
    unittest.main()
