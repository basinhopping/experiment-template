"""A whole round through its own lines, the way a person runs one, on a stand-in node: a temporary home folder for the
node and a copy of the repository for the computer, real tmux (a server of its own), and stand-ins for nvidia-smi (two
GPUs numbered 98 and 99, 98 busy) and systemctl (no slurmd). No machine has a GPU 99, so the workers this test starts
(CUDA_VISIBLE_DEVICES=99) see no GPU, even on a shared node where line 02 runs this suite; their own check that the
GPU is visible is stood in for, since torch would see no GPU 99. The environment is this Python, except that the
unit tests and env-check inside line 02 are stood in for (line 02 runs the whole suite on a node; here that would
run this test again).

The computer makes the data and prepares the round (round-new, the RULE, rule-fix, round-check); push sends the code
and the data (packing only: PUSH_DRY_RUN); lines 02 to 05 run on the node exactly as the preparation folder holds
them; pull fetches the archive from the stand-in's outgoing folder (PULL_FROM) and checks every file; the reading
runs on the computer in its fixed order. Nothing touches a real node or a real GPU."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from tests.support import REPO

# Linux only, as the node: its scripts use GNU sed, find and tar (a Mac with tmux and sha256sum would fail for that)
TOOLS = sys.platform.startswith("linux") and all(shutil.which(t) for t in ("bash", "tar", "sha256sum", "awk", "sed",
                                                                              "tmux"))
CODE = ["src", "tests", "sweeps", "node", "env", "docs", "templates", "notebooks", "README.md", "START_HERE.md",
        "CLAUDE.md", "TESTING.md", "LICENSE", "project.json", ".gitignore", ".gitattributes"]
DATE = "20261028"


def run(command, env, cwd=None, check=True, timeout=600):
    done = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and done.returncode != 0:
        raise AssertionError(f"{command} exited {done.returncode}:\n{done.stdout}\n{done.stderr}")
    return done


@unittest.skipUnless(TOOLS and (REPO / "node").is_dir(),
                     "needs Linux (the node's scripts use GNU tools), bash, tar, sha256sum, tmux and the node/ folder")
class TestARoundThroughItsLines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        base = Path(self.tmp.name)
        self.home, self.pc, self.bin = base / "home", base / "pc", base / "bin"
        for folder in (self.home, self.pc, self.bin, base / "tmux"):
            folder.mkdir()
        for item in CODE:
            source = REPO / item
            if source.is_dir():
                shutil.copytree(source, self.pc / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif source.is_file():
                shutil.copy2(source, self.pc / item)
        self.project = json.loads((self.pc / "project.json").read_text())["project"]
        (self.bin / "nvidia-smi").write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            case "$*" in
                -L) printf 'GPU 98: stand-in\\nGPU 99: stand-in\\n' ;;
                *index,uuid*) printf '98, GPU-a\\n99, GPU-b\\n' ;;
                *compute-apps*) ;;
                *index,memory.used*) printf '98, 50000\\n99, 100\\n' ;;
                *query-gpu=index\\ *|*query-gpu=index) printf '98\\n99\\n' ;;
            esac
            """))
        (self.bin / "systemctl").write_text("#!/usr/bin/env bash\nexit 3\n")
        # the node's environment: this Python, with the unit tests and env-check of line 02 stood in for. A worker
        # checks that its GPU is visible, asking torch (or CuPy) when it is installed, and neither sees GPU 99: with
        # torch installed, every worker stopped at once and no job ran (the node, 2026-10-29). So the workers count
        # the stand-in GPU as visible, as the stand-in nvidia-smi does on a machine without torch or CuPy. No job may
        # take a GPU path, though: where cuML is installed (check.sh rapids rapids runs this suite in the rapids
        # environment), a job on auto would choose cuML's path for GPU 99 and fail (the node, 2026-10-30). So every
        # command of the template run here finds no GPU path ready, and every job runs on the CPU.
        venv = self.home / "venvs" / f"{self.project}-main" / "bin"
        venv.mkdir(parents=True)
        no_gpu_path = "sweepkit.devices.gpu_path_ready = lambda gpu: False; "
        cli = 'runpy.run_module("sweepkit.cli", run_name="__main__", alter_sys=True)'
        visible = f"import runpy, sweepkit.devices; sweepkit.devices.gpu_visible = lambda: True; {no_gpu_path}{cli}"
        cpu_only = f"import runpy, sweepkit.devices; {no_gpu_path}{cli}"
        (venv / "python").write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            case "$*" in
                *"-m unittest"*) echo "Ran 1 tests in 0.0s"; echo; echo "OK"; exit 0 ;;
                *env-check*) echo "ENV OK"; exit 0 ;;
                "-m sweepkit.cli worker "*) exec {shlex.quote(sys.executable)} -c {shlex.quote(visible)} "${{@:3}}" ;;
                "-m sweepkit.cli "*) exec {shlex.quote(sys.executable)} -c {shlex.quote(cpu_only)} "${{@:3}}" ;;
            esac
            exec {shlex.quote(sys.executable)} "$@"
            """))
        for f in (self.bin / "nvidia-smi", self.bin / "systemctl", venv / "python"):
            f.chmod(0o755)
        # SCHEDULE_POLL_SECONDS: how often the round's schedule looks whether its queue is empty (60 on a node); the
        # tmux server this test starts passes it to the round's session
        self.env = dict(os.environ, HOME=str(self.home), PATH=f"{self.bin}:{os.environ['PATH']}",
                        TMUX_TMPDIR=str(base / "tmux"), NVIDIA_SMI_CMD=str(self.bin / "nvidia-smi"),
                        SCHEDULE_POLL_SECONDS="2")
        for name in ("TMUX", "CUDA_VISIBLE_DEVICES", "CHECK_GPUS", "ENVIRONMENT", "MIN_FREE_GB", "PYTHONPATH"):
            self.env.pop(name, None)
        self.pc_env = dict(self.env, PYTHONPATH=str(self.pc / "src"))

    def tearDown(self):
        run(["tmux", "kill-server"], self.env, check=False)
        self.tmp.cleanup()

    def cli(self, *args):
        return run([sys.executable, "-m", "sweepkit.cli", *args], self.pc_env, cwd=self.pc).stdout

    def line(self, name: str) -> str:
        """The command of a line file, as the person pastes it on the node (from the home folder)."""
        text = (self.pc / "artifacts" / "analyses" / f"rf_prep_{DATE}" / "lines" / name).read_text()
        (command,) = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
        return run(["bash", "-c", command], self.env, cwd=self.home).stdout

    def prepare_push_and_install(self) -> str:
        """The computer's part up to the push, then line 02 on the node; returns line 02's output."""
        # the computer: the data, as TESTING.md's flow check makes it, and the round
        testing = (self.pc / "TESTING.md").read_text(encoding="utf-8")
        (made,) = [line for line in testing.splitlines() if "make-synthetic --generator regression_nonlinear" in line]
        self.cli(*shlex.split(made.split("sweepkit.cli ", 1)[1]))
        (self.pc / "data" / "flow").mkdir()
        for f in (self.pc / "data" / "synthetic").glob("regress*"):
            shutil.move(str(f), str(self.pc / "data" / "flow" / f.name))
        self.cli("round-new", "--round", "rf", "--date", DATE, "--question", "Does a round run through its lines?",
                 "--sweep-files", "sweeps/checks/flow_regression.json", "--workers-per-gpu", "2", "--gpus", "99",
                 "--end-time", "now + 30 minutes", "--environment", "main", "--profiles", "core", "--min-free-gb", "1")
        prep = self.pc / "artifacts" / "analyses" / f"rf_prep_{DATE}"
        rule = prep / "RULE_rf.txt"
        text = rule.read_text()
        for key in ("{{what_runs}}", "{{reading}}", "{{stop_condition}}", "{{limits}}"):
            text = text.replace(key, "as the round's files say.")
        rule.write_text(text)
        jobs = json.loads(self.cli("check", "--sweep-file", "sweeps/checks/flow_regression.json"))
        control = next(c["config"] for c in jobs["configurations"] if c["stage"] == "control")
        families = {"families": [{"name": "against_control", "files": "test", "metric": "rmse",
                                  "comparisons": [{"a": {"selection": "rf_choice"},
                                                   "b": {"sweep": "check_flow_regression", "config": control}}]}]}
        (prep / "rf_families.json").write_text(json.dumps(families))
        readme = prep / "README.txt"
        readme.write_text(readme.read_text().replace("  sweep_files.sha256:", "  rf_families.json: families.\n"
                                                     "  sweep_files.sha256:", 1))
        self.cli("rule-fix", "--name", "rf", "--rule-file", str(rule), "--sweeps", "check_flow_regression",
                 "--families", str(prep / "rf_families.json"), "--late-reason", "none")
        self.assertIn("ROUND OK", self.cli("round-check", "--round", "rf", "--date", DATE))
        # push, packing only, with the data; the copy goes to the node's incoming folder
        pushed = run(["bash", "node/pc/push.sh", "--data", "flow"], dict(self.env, PUSH_DRY_RUN="1"),
                     cwd=self.pc).stdout
        stage = next(line for line in pushed.splitlines() if line.startswith("dry run")).split("is in ")[1]
        shutil.move(stage, str(self.home / f"{self.project}_incoming"))
        # line 02: install the code, check its digest, test on a free GPU of the round's
        return self.line("02_install_test.txt")

    def test_line_03_starts_nothing_when_the_rounds_gpus_are_busy(self):
        self.prepare_push_and_install()
        busy = (self.bin / "nvidia-smi").read_text().replace("98, 50000\\n99, 100", "98, 100\\n99, 50000")
        (self.bin / "nvidia-smi").write_text(busy)
        out = self.line("03_start.txt")
        self.assertIn("no free GPU among the round's (99): stop here and paste this", out)
        self.assertNotEqual(run(["tmux", "has-session", "-t", f"={self.project}_rf"], self.env,
                                check=False).returncode, 0)

    def test_prepare_push_run_pack_pull_and_read(self):
        out = self.prepare_push_and_install()
        digest = run([sys.executable, "-m", "sweepkit.cli", "tree-digest", "--root", str(self.pc)], self.pc_env,
                     cwd=self.pc).stdout.split()[-1]
        self.assertIn(f"TREE OK {digest}", out)
        self.assertIn("tests on GPU 99 (CUDA_VISIBLE_DEVICES)", out)
        self.assertIn("check: both passed", out)
        self.assertIn("check.sh exit 0", out)
        # the data's unpack line, as push prints it
        incoming = self.home / f"{self.project}_incoming"
        (archive,) = sorted(p.name for p in incoming.glob(f"{self.project}_data_flow_*.tar"))
        run(["bash", str(incoming / "unpack.sh"), archive], self.env, cwd=self.home)
        # line 03: the chain in its own session; its plan came with the code
        out = self.line("03_start.txt")
        self.assertIn(f"{self.project}_rf started; free GPUs now: 99", out)
        workers = f"={self.project}_gpu99"
        deadline = time.time() + 240
        started = False
        while time.time() < deadline:
            alive = run(["tmux", "has-session", "-t", workers], self.env, check=False).returncode == 0
            started = started or alive
            done = len(list((self.home / self.project / "queue" / "done").glob("*")))
            if started and not alive or done == 45:
                break
            time.sleep(1)
        while run(["tmux", "has-session", "-t", workers], self.env, check=False).returncode == 0:
            if time.time() > deadline:
                self.fail("the workers did not finish")
            time.sleep(1)
        logs = self.home / self.project / "logs" / "rf"
        chain = (logs / "chain.txt").read_text()
        self.assertIn("SCHEDULE OK", (logs / "1_schedule.txt").read_text())
        smoked = (logs / "2_smoke_enqueue.txt").read_text()
        self.assertIn("smoke tests on GPU 99", smoked)
        self.assertIn("SMOKE OK", smoked)
        self.assertIn("ALL DONE", smoked)
        self.assertIn("step 3: env ENVIRONMENT=main MIN_FREE_GB=1 bash node/schedule.sh --end-when-empty 2", chain)
        self.assertIn("GPU 99 started: 2 workers", (logs / "3_schedule.txt").read_text())
        # line 04: the status, then line 05: pack once the workers are gone
        status = self.line("04_status.txt")
        self.assertIn("runs: 45 complete of 45, 0 failed", status)
        # each step's log in short, so the smoke tests and the GPUs show without opening the logs (the dry run,
        # 2026-10-29: line 03 named these logs and no line showed them)
        self.assertRegex(status, r"--- logs/rf/2_smoke_enqueue.txt\n[1-9][0-9]* SMOKE OK, 0 SMOKE FAILED\n")
        for shown in ("--- logs/rf/1_schedule.txt", "SCHEDULE OK", "ALL DONE", "GPU 99 started: 2 workers"):
            self.assertIn(shown, status)
        # with no job left, the schedule ends long before its end time (30 minutes away), and the round's session with
        # it: the next round's install is not refused for a session that only waits
        deadline = time.time() + 60
        while run(["tmux", "has-session", "-t", f"={self.project}_rf"], self.env, check=False).returncode == 0:
            if time.time() > deadline:
                self.fail("the round's session did not end when its queue was empty")
            time.sleep(1)
        self.assertIn("CHAIN DONE", (logs / "chain.txt").read_text())
        self.assertIn("so the schedule ends before its end", (logs / "3_schedule.txt").read_text())
        packed = self.line("05_pack.txt").splitlines()
        # line 06: the line pack printed to paste on the computer, run as pasted (from any folder: it starts with the
        # cd that push recorded), fetching from the stand-in's outgoing folder, checked file by file
        paste = packed[packed.index("paste this on your computer (a terminal):") + 1]
        self.assertTrue(paste.startswith(f"cd '{self.pc}' && bash node/pc/pull.sh {self.project}_results_"), paste)
        pulled = run(["bash", "-c", paste], dict(self.env, PULL_FROM=str(self.home / f"{self.project}_outgoing")),
                     cwd=self.home).stdout
        self.assertIn("checked against the node's hashes", pulled)
        # the reading, on the computer, in its fixed order
        analysis = self.cli("analyze", "--sweep", "check_flow_regression", "--confidence", "0.95", "--n-boot", "200",
                            "--seed", "1", "--meaningful", "0.75", "--top", "5")
        self.assertIn("45 of 45 runs complete", analysis)
        ranked = [line for line in analysis.splitlines() if line.startswith("  ") and "| " in line]
        best = ranked[0].split("| ")[1].split()[0][:10]
        self.cli("select", "--sweep", "check_flow_regression", "--config", best, "--name", "rf_choice", "--reason",
                 "the lowest validation rmse, as the RULE says")
        self.assertIn("test results of selection rf_choice", self.cli("test-report", "--selection", "rf_choice",
                                                                      "--confidence", "0.95"))
        rule_name = next(p.name for p in (self.pc / "artifacts" / "rules").iterdir() if p.name.startswith("rf_"))
        compared = self.cli("test-compare", "--rule", rule_name, "--confidence", "0.95", "--n-boot", "200", "--seed",
                            "1", "--meaningful", "0.75")
        self.assertIn("better", compared)


if __name__ == "__main__":
    unittest.main()
