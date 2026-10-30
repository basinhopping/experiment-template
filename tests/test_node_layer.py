"""The node layer, run here with a temporary home folder standing in for the node and for the user's computer:
push (packing only), unpack, the tree digest, pack, pull, the GPU schedule with stand-ins for tmux and nvidia-smi,
the chain of a round, and the repository working with node/ removed.

Nothing touches a real node: push runs with PUSH_DRY_RUN=1 and pull with PULL_FROM, and every script runs from a copy
of the code installed into the temporary home."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

from tests.support import REPO

TOOLS = all(shutil.which(t) for t in ("bash", "tar", "sha256sum", "awk", "sed"))


def run(command, env, cwd=None, check=True):
    done = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, timeout=600)
    if check and done.returncode != 0:
        raise AssertionError(f"{command} exited {done.returncode}:\n{done.stdout}\n{done.stderr}")
    return done


@unittest.skipUnless(TOOLS and (REPO / "node").is_dir(), "needs bash, tar, sha256sum and the node/ folder")
class TestNodeLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.tmp.name) / "home"
        cls.home.mkdir()
        (Path(cls.tmp.name) / "t").mkdir()
        cls.project = json.loads((REPO / "project.json").read_text())["project"]
        # The scripts' own tmux calls (unpack's check for running sessions) go to a tmux server of the test's own, so
        # a real session of the project (the node test runs these tests inside one) does not make them refuse.
        (Path(cls.tmp.name) / "tmux").mkdir()
        cls.env = dict(os.environ, HOME=str(cls.home), TMPDIR=str(Path(cls.tmp.name) / "t"), PUSH_DRY_RUN="1",
                       TMUX_TMPDIR=str(Path(cls.tmp.name) / "tmux"))
        cls.env.pop("TMUX", None)
        out = run(["bash", str(REPO / "node" / "pc" / "push.sh")], cls.env).stdout
        cls.code = next(line.split()[2] for line in out.splitlines() if line.startswith("code archive "))
        incoming = Path(next(line for line in out.splitlines() if line.startswith("dry run")).split("is in ")[1])
        shutil.move(str(incoming), str(cls.home / f"{cls.project}_incoming"))
        cls.unpack = run(["bash", str(cls.home / f"{cls.project}_incoming" / "unpack.sh"), cls.code], cls.env).stdout
        cls.node = cls.home / cls.project
        run(["bash", str(cls.home / f"{cls.project}_incoming" / "unpack.sh"), cls.code, str(cls.home / "pc")], cls.env)
        cls.pc = cls.home / "pc"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unpack_installs_the_code_items_and_the_digests_agree(self):
        from sweepkit.hashing import tree_digest

        shell = self.unpack.strip().splitlines()[-1].rsplit(" ", 1)[1]
        self.assertEqual(shell, tree_digest(self.node))
        self.assertTrue((self.node / "src" / "sweepkit" / "__init__.py").is_file())
        self.assertFalse((self.node / "node.local.txt").exists())
        self.assertIn(self.code, (self.node / "CODE_VERSIONS.log").read_text())
        self.assertEqual(tree_digest(self.pc), tree_digest(self.node))

    def test_pack_then_pull_checks_every_file(self):
        runs = self.node / "artifacts" / "runs" / "s1"
        for k in range(3):
            (runs / f"r{k}").mkdir(parents=True)
            (runs / f"r{k}" / "metrics_val.json").write_text(json.dumps({"k": k}))
        out = run(["bash", str(self.node / "node" / "pack.sh"), "runs/s1"], self.env).stdout
        name = out.splitlines()[0].rsplit(": ", 1)[1]
        self.assertTrue(name.startswith(f"{self.project}_results_runs_s1_"))
        # push recorded the computer's folder, so pack ends with the whole line to paste there
        folder = str(REPO).replace("'", "'\\''")
        self.assertEqual(out.splitlines()[-2:], ["paste this on your computer (a terminal):",
                                                 f"cd '{folder}' && bash node/pc/pull.sh {name}"])
        env = dict(self.env, PULL_FROM=str(self.home / f"{self.project}_outgoing"))
        pulled = run(["bash", str(self.pc / "node" / "pc" / "pull.sh"), name], env).stdout
        self.assertIn("3 files unpacked", pulled)
        self.assertEqual(json.loads((self.pc / "artifacts" / "runs" / "s1" / "r2" / "metrics_val.json").read_text()),
                         {"k": 2})
        again = run(["bash", str(self.pc / "node" / "pc" / "pull.sh"), name], env, check=False)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("REFUSING", again.stdout)
        (runs / "r3").mkdir()
        (runs / "r3" / "metrics_val.json").write_text("{}")
        out = run(["bash", str(self.node / "node" / "pack.sh"), "--after", name, "runs/s1"], self.env).stdout
        second = out.splitlines()[0].rsplit(": ", 1)[1]
        pulled = run(["bash", str(self.pc / "node" / "pc" / "pull.sh"), second], env).stdout
        self.assertIn("1 files unpacked", pulled)
        (runs / "r4.tmp").mkdir()
        refused = run(["bash", str(self.node / "node" / "pack.sh"), "runs/s1"], self.env, check=False)
        self.assertIn("REFUSING: unfinished folders", refused.stdout)

    def test_pack_prints_the_pull_line_for_the_computer_push_recorded(self):
        recorded = self.home / f"{self.project}_incoming" / "pc_folder.txt"
        kept = recorded.read_text()
        runs = self.node / "artifacts" / "runs" / "s_line"
        (runs / "r0").mkdir(parents=True)
        (runs / "r0" / "metrics_val.json").write_text("{}")
        try:
            recorded.write_text("windows C:\\Some Folder\\my repo\n")      # as push.ps1 writes it
            out = run(["bash", str(self.node / "node" / "pack.sh"), "runs/s_line"], self.env).stdout
            name = out.splitlines()[0].rsplit(": ", 1)[1]
            self.assertEqual(out.splitlines()[-2:], [
                "paste this on your computer (Command Prompt):",
                'cd /d "C:\\Some Folder\\my repo" && powershell -ExecutionPolicy Bypass -File node\\pc\\pull.ps1 '
                + name])
            recorded.unlink()                                                 # no push recorded a folder
            out = run(["bash", str(self.node / "node" / "pack.sh"), "runs/s_line"], self.env).stdout
            name = out.splitlines()[0].rsplit(": ", 1)[1]
            self.assertIn("paste this on your computer, in the repository folder:", out)
            self.assertIn(f"    Mac, Linux:  bash node/pc/pull.sh {name}", out)
        finally:
            recorded.write_text(kept)

    def _fakes(self, busy_mib: int, slurm: bool, tag: str = "") -> dict:
        bin_dir = Path(self.tmp.name) / f"bin_{busy_mib}_{int(slurm)}{tag}"
        bin_dir.mkdir(exist_ok=True)
        log = bin_dir / "tmux.log"
        (bin_dir / "tmux").write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            echo "$@" >> {log}
            if [ "$1" = has-session ]; then exit 1; fi
            exit 0
            """))
        (bin_dir / "nvidia-smi").write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            case "$*" in
                *index,uuid*) printf '0, GPU-a\\n1, GPU-b\\n' ;;
                *compute-apps*) ;;
                *index,memory.used*) printf '0, 100\\n1, {busy_mib}\\n' ;;
                *query-gpu=index\\ *|*query-gpu=index) printf '0\\n1\\n' ;;
            esac
            """))
        (bin_dir / "systemctl").write_text(f"#!/usr/bin/env bash\nexit {0 if slurm else 3}\n")
        for f in ("tmux", "nvidia-smi", "systemctl"):
            (bin_dir / f).chmod(0o755)
        return dict(self.env, PATH=f"{bin_dir}:{os.environ['PATH']}", TMUX_CMD=str(bin_dir / "tmux"),
                    NVIDIA_SMI_CMD=str(bin_dir / "nvidia-smi")), log

    def test_test_all_refuses_a_busy_gpu_slurm_and_too_little_disk_and_ends_by_itself(self):
        script = str(self.node / "node" / "test_all.sh")
        env, _ = self._fakes(5000, False)
        env = dict(env, MIN_FREE_GB="0")
        busy = run(["bash", script, "1", "2", "main"], env, check=False)
        self.assertNotEqual(busy.returncode, 0)
        self.assertIn("GPU 1 is not free (memory in use: 5000 MiB)", busy.stdout)
        slurm_env, _ = self._fakes(100, True)
        refused = run(["bash", script, "0", "2", "main"], dict(slurm_env, MIN_FREE_GB="0"), check=False)
        self.assertIn("slurmd is active", refused.stdout)
        full = run(["bash", script, "0", "2", "main"], dict(env, MIN_FREE_GB="100000000"), check=False)
        self.assertIn("NOT ENOUGH DISK", full.stdout)
        for wrong in (["0"], ["0", "2"], ["0", "two", "main"], ["0", "2", "everything"]):
            self.assertNotEqual(run(["bash", script, *wrong], env, check=False).returncode, 0, wrong)
        # nothing built in this home: the environment is recorded as not built, and the run ends with its folder
        done = run(["bash", script, "0", "2", "rapids"], env).stdout
        self.assertIn("environment rapids: not built (node/setup_env.sh builds it), skipped", done)
        self.assertIn("ends by", done)
        self.assertIn("TEST-ALL DONE: 0 steps: 0 ok, 0 failed, 0 timed out, 0 not run; not built: rapids", done)
        folders = sorted((self.node / "artifacts" / "test_all").iterdir())
        self.assertFalse(any(f.name.endswith(".tmp") for f in folders))
        lines = (folders[-1] / "summary.csv").read_text().splitlines()
        self.assertEqual(lines[0], "step,environment,status,exit_code,seconds,last_line")
        self.assertTrue(lines[1].startswith("environment,rapids,not built,"))

    def test_test_all_stops_at_its_time_limit(self):
        """A stand-in environment whose Python answers at once, except the unit tests, which outlast the limit: the
        tests step is timed out and the steps after it are recorded as not run."""
        env, _ = self._fakes(100, False)
        venv = self.home / "venvs" / f"{self.project}-main"
        (venv / "bin").mkdir(parents=True)
        python = venv / "bin" / "python"
        python.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            case "$*" in
                *required-modules*) echo numpy ;;
                *env-check*) echo "ENV OK" ;;
                *unittest*) sleep 60 ;;
                -c*) exit 1 ;;
                *) echo "SMOKE-ALL OK" ;;
            esac
            """))
        python.chmod(0o755)
        (self.node / "artifacts" / "env").mkdir(parents=True, exist_ok=True)
        (self.node / "artifacts" / "env" / "main_20261028T000000Z.txt").write_text("# profiles: core\nnumpy==1\n")
        try:
            started = time.time()
            done = run(["bash", str(self.node / "node" / "test_all.sh"), "0", "0.004", "main"],
                       dict(env, MIN_FREE_GB="0", MIN_STEP_SECONDS="3")).stdout
            self.assertLess(time.time() - started, 40)               # 0.004 hours is 14 s, plus the summary
        finally:
            shutil.rmtree(self.home / "venvs")
        self.assertIn("TEST-ALL DONE: 5 steps: 1 ok, 0 failed, 1 timed out, 3 not run", done)
        folder = sorted((self.node / "artifacts" / "test_all").iterdir())[-1]
        rows = [line.split(",")[:4] for line in (folder / "summary.csv").read_text().splitlines()[1:]]
        self.assertEqual(rows, [["env_check_main", "main", "ok", "0"], ["tests_main", "main", "timed out", "124"],
                                ["smoke_cuda_main", "main", "not run: time", ""],
                                ["smoke_cpu_main", "main", "not run: time", ""],
                                ["large_data_main", "main", "not run: time", ""]])
        self.assertIn(" in " + str(venv), (folder / "summary.txt").read_text())    # the environment's size

    def test_check_runs_its_gpu_tests_on_a_free_gpu_only(self):
        """GPU 1 is busy (5000 MiB in use): check.sh puts the tests on GPU 0, refuses when CHECK_GPUS names only GPU 1,
        and keeps a CUDA_VISIBLE_DEVICES set before it (empty: no GPU, and then no GPU is required). A stand-in
        environment prints what the tests would see."""
        env, _ = self._fakes(5000, False)
        python = self.home / "venvs" / f"{self.project}-main" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            case "$*" in
                *required-modules*) echo numpy ;;
                *env-check*) echo "ENV OK" ;;
                *unittest*) echo "tests see GPU [${CUDA_VISIBLE_DEVICES-unset}] require [$SWEEPKIT_REQUIRE]"; echo OK ;;
            esac
            """))
        python.chmod(0o755)
        script = str(self.node / "node" / "check.sh")
        # run inside node/check.sh (line 02 of a round), these tests inherit its CHECK_GPUS and CUDA_VISIBLE_DEVICES
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("CHECK_GPUS", None)
        try:
            out = run(["bash", script, "main", "core"], env).stdout
            self.assertIn("tests on GPU 0 (CUDA_VISIBLE_DEVICES)", out)
            self.assertIn("tests see GPU [0] require [numpy,gpu]", out)
            self.assertIn("check: both passed", out)
            refused = run(["bash", script, "main", "core"], dict(env, CHECK_GPUS="1"), check=False)
            self.assertEqual(refused.returncode, 1)
            self.assertIn("no free GPU among 1 for the tests", refused.stdout)
            self.assertNotIn("tests see GPU", refused.stdout)
            out = run(["bash", script, "main", "core"], dict(env, CHECK_GPUS="1,0")).stdout
            self.assertIn("tests see GPU [0]", out)
            out = run(["bash", script, "main", "core"], dict(env, CUDA_VISIBLE_DEVICES="1")).stdout
            self.assertIn("tests see GPU [1] require [numpy,gpu]", out)
            out = run(["bash", script, "main", "core"], dict(env, CUDA_VISIBLE_DEVICES="")).stdout
            self.assertIn("tests on GPU none", out)
            self.assertIn("tests see GPU [] require [numpy]", out)
        finally:
            shutil.rmtree(self.home / "venvs")

    def test_rapids_run_builds_each_command_line(self):
        """cuml: cuml.accel with -v; cuml+cudf: cuDF's runner, which refuses -v (the first node run, 2026-10-27), so
        the log level goes through CUML_ACCEL_LOG_LEVEL. A stand-in Python prints what it was given."""
        python = self.home / "venvs" / f"{self.project}-rapids" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text('#!/usr/bin/env bash\necho "args: $*"\necho "level: ${CUML_ACCEL_LOG_LEVEL:-unset}"\n'
                          'echo "[cuml.accel] KMeans.fit ran on GPU"\n')
        python.chmod(0o755)
        script = self.home / "standin.py"
        script.write_text("print('a script')\n")
        try:
            out = run(["bash", str(self.node / "node" / "rapids_run.sh"), "cuml+cudf", str(script), "a.csv"],
                      self.env).stdout
            self.assertIn(f"args: -m cudf.pandas -m cuml.accel {script} a.csv", out)
            self.assertIn("level: info", out)
            self.assertIn("calls that ran on the GPU: 1", out)
            out = run(["bash", str(self.node / "node" / "rapids_run.sh"), "cuml", str(script)], self.env).stdout
            self.assertIn(f"args: -m cuml.accel -v {script}", out)
        finally:
            shutil.rmtree(self.home / "venvs" / f"{self.project}-rapids")

    def _venv_standin(self, environment: str = "main") -> None:
        """An environment whose Python runs `check` with this Python and answers every other command with exit 0:
        schedule.sh refuses to start workers whose environment has no Python, and the swap expands sweeps with check."""
        python = self.home / "venvs" / f"{self.project}-{environment}" / "bin" / "python"
        if not python.exists():
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text(f'#!/usr/bin/env bash\ncase "$*" in *" check "*) exec {shlex.quote(sys.executable)} "$@" ;; esac\n'
                              f"exit 0\n")
            python.chmod(0o755)
        self.addCleanup(shutil.rmtree, self.home / "venvs", ignore_errors=True)

    def _swap_env(self, tag: str, sessions: tuple):
        """Stand-ins for swap_and_restart.sh. tmux keeps its sessions as files in a folder: <name>.static for a session
        that only exists (the names given), <name>.pid for a <project>_schedule session, whose command really runs in
        the background, and <name>.panes for the pane processes of a session. Workers' sessions are only logged.
        nvidia-smi shows GPUs 0 and 1 free; no slurmd. The environment and disk floor are set, as the swap's tmux
        command sets them, and no lead time is asked for (these swaps end within minutes)."""
        env, log = self._fakes(100, False, tag=tag)
        state = Path(self.tmp.name) / f"tmux_state{tag}"
        state.mkdir()
        for name in sessions:
            (state / f"{self.project}_{name}.static").write_text("")
        Path(env["TMUX_CMD"]).write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            echo "$@" >> {log}
            S={state}
            alive() {{ [ -e "$S/$1.static" ] || {{ [ -f "$S/$1.pid" ] && kill -0 "$(cat "$S/$1.pid")" 2>/dev/null; }}; }}
            case "$1" in
                ls) for f in "$S"/*.static "$S"/*.pid; do [ -e "$f" ] || continue; n="${{f##*/}}"; n="${{n%.*}}"
                        if alive "$n"; then echo "$n"; fi; done | sort -u; exit 0 ;;
                has-session) alive "${{3#=}}"; exit $? ;;
                kill-session) n="${{3#=}}"; rm -f "${{S:?}}/${{n:?}}.static"
                    if [ -f "$S/$n.pid" ]; then kill -- "-$(cat "$S/$n.pid")" 2>/dev/null; rm -f "${{S:?}}/${{n:?}}.pid"; fi; exit 0 ;;
                new) case "$4" in *_schedule) setsid bash -c "$5" > /dev/null 2>&1 & echo $! > "$S/$4.pid" ;; esac; exit 0 ;;
                list-panes) n="${{4#=}}"; if [ -f "$S/$n.panes" ]; then cat "$S/$n.panes"; fi; exit 0 ;;
            esac
            exit 0
            """))
        env = dict(env, SWAP_HOME=str(self.home), SWAP_POLL_SECONDS="1", ENVIRONMENT="main", MIN_FREE_GB="15",
                   SWAP_LEAD_MINUTES="0")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        (self.node / "queue").mkdir(exist_ok=True)
        self._venv_standin()

        def clean():
            for pid_file in state.glob("*.pid"):
                subprocess.run(["kill", "--", f"-{pid_file.read_text().strip()}"], capture_output=True)
            for f in ("STOP", "SWAP_FAILED"):
                (self.node / "queue" / f).unlink(missing_ok=True)
        self.addCleanup(clean)
        return env, log, state

    def _swap(self, *extra, end_in=600, digest=None):
        if digest is None:
            digest = run(["bash", str(self.node / "node" / "tree_digest.sh"), str(self.node)], self.env).stdout.split()[-1]
        return ["bash", str(self.node / "node" / "swap_and_restart.sh"), self.code, digest, "1", *extra[:1],
                str(int(time.time()) + end_in), *extra[1:]]

    def _started(self, log: Path, gpu: int, count: int = 1, timeout: float = 20) -> list:
        """The workers' sessions the swap's schedule started on a GPU, once it has started `count` of them."""
        deadline = time.time() + timeout
        while True:
            started = [line for line in (log.read_text().splitlines() if log.exists() else [])
                       if f"new -d -s {self.project}_gpu{gpu} " in line]
            if len(started) >= count or time.time() > deadline:
                return started
            time.sleep(0.2)

    def _python_of_the_project(self, seconds: int) -> subprocess.Popen:
        """A Python of the project's environment, as an activated environment shows it (VIRTUAL_ENV)."""
        busy = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"],
                                env=dict(os.environ, VIRTUAL_ENV=f"{self.home}/venvs/{self.project}-main"))
        self.addCleanup(lambda: (busy.kill(), busy.wait()))
        return busy

    def test_swap_installs_then_replaces_the_schedule(self):
        """Beside a schedule (bash only), the code is installed and the session <project>_schedule replaced by one
        that runs the workers, with the environment and floor given, until the swap's end; the swap's own session
        ends. A queue/SWAP_FAILED from an earlier swap is moved into logs/."""
        env, log, state = self._swap_env("_swap1", ("swap", "schedule"))
        (self.node / "queue" / "SWAP_FAILED").write_text("an earlier swap's failure\n")
        done = run(self._swap("0"), env, check=False)
        self.assertEqual(done.returncode, 0, done.stdout)
        out = done.stdout
        self.assertIn("TREE OK", out)
        self.assertIn("swap finished: new code, no sweeps given", out)
        self.assertIn(f"the running {self.project}_schedule is ended", out)
        self.assertIn(f"in tmux session {self.project}_schedule", out)
        self.assertFalse((self.node / "queue" / "SWAP_FAILED").exists())
        moved = sorted((self.node / "logs").glob("swap_failed_before_*.txt"))
        self.assertEqual(moved[-1].read_text(), "an earlier swap's failure\n")
        started = self._started(log, 0)
        self.assertEqual(len(started), 1)
        self.assertIn("ENVIRONMENT=main MIN_FREE_GB=15 ", started[0])
        schedule = (state / f"{self.project}_schedule.pid").read_text().strip()
        self.assertEqual(subprocess.run(["kill", "-0", schedule], capture_output=True).returncode, 0)
        # sweeps given: every one smoke-tested on a free GPU of the swap's before the workers start, and enqueued
        # only when all pass; a failure is recorded in queue/SWAP_FAILED and the workers start all the same
        folder = self._smoke_standin(env)
        good, bad = str(folder / "good.json"), str(folder / "bad.json")
        out = run(self._swap("0", good), env, check=False).stdout
        self.assertIn("smoke tests on GPU 0", out)
        self.assertIn(f"sweep {good} on cuda, GPU [0]", out)
        self.assertIn('"sweep_id"', out)
        self.assertIn("swap finished: new code, sweeps enqueued", out)
        self.assertLess(out.index("smoke tests on GPU 0"), out.index("workers on GPUs 0 until"))
        self.assertFalse((self.node / "queue" / "SWAP_FAILED").exists())
        self.assertEqual(len(self._started(log, 0, 2)), 2)
        out = run(self._swap("0", good, bad), env, check=False).stdout
        self.assertIn("NOT ENQUEUED: 1 of 2 sweeps failed their smoke test, so none is enqueued", out)
        self.assertNotIn('"sweep_id"', out)
        self.assertIn("swap finished: new code, SWEEPS NOT ENQUEUED", out)
        self.assertIn("not enqueued", (self.node / "queue" / "SWAP_FAILED").read_text())
        self.assertEqual(len(self._started(log, 0, 3)), 3)
        # without the allowance, a schedule's session makes a plain unpack refuse
        refused = run(["bash", str(self.home / f"{self.project}_incoming" / "unpack.sh"), self.code], env,
                      check=False)
        self.assertIn("REFUSING: tmux sessions", refused.stdout)

    def test_swap_refuses_before_any_change(self):
        """Nothing is stopped, and every refusal is added to queue/SWAP_FAILED: a missing or mistyped setting, a
        schedule schedule.sh would refuse, an archive or a tree that is not the one named, code in place that could
        not be put back as it is, a sweep the GPU workers would not take, CPU workers, a round's chain (it uses the
        code), or a Python of the project outside the workers."""
        env, log, state = self._swap_env("_swap2", ("swap",))
        round_env, round_log, _ = self._swap_env("_swap2r", ("swap", "schedule", "r1"))
        cpu_env, cpu_log, _ = self._swap_env("_swap2c", ("swap", "cpu"))
        folder = self._smoke_standin(env)
        incoming = self.home / f"{self.project}_incoming"
        corrupt = self.code.replace("_code_", "_code_corrupt_")
        shutil.copy(incoming / self.code, incoming / corrupt)
        (incoming / f"{corrupt}.sha256").write_text(f"{'0' * 64}  {corrupt}\n")
        self.addCleanup((incoming / corrupt).unlink, missing_ok=True)
        self.addCleanup((incoming / f"{corrupt}.sha256").unlink, missing_ok=True)
        other = self.code.replace("_code_", "_code_other_")
        shutil.copy(incoming / self.code, incoming / other)
        (incoming / f"{other}.sha256").write_text(run(["sha256sum", self.code], self.env, cwd=incoming).stdout
                                                  .replace(self.code, other))
        self.addCleanup((incoming / other).unlink, missing_ok=True)
        self.addCleanup((incoming / f"{other}.sha256").unlink, missing_ok=True)
        corrupt_swap, other_swap, no_end = self._swap("0"), self._swap("0"), self._swap("0")
        corrupt_swap[2], other_swap[2], no_end[6] = corrupt, other, ""
        cases = [(env, {"ENVIRONMENT": ""}, self._swap("0"), "set ENVIRONMENT and MIN_FREE_GB"),
                 (env, {"ENVIRONMENT": "mian"}, self._swap("0"), "no environment"),
                 (env, {"MIN_FREE_GB": "15G"}, self._swap("0"), "MIN_FREE_GB='15G'"),
                 (env, {"WAIT_MINUTES": "1.5"}, self._swap("0"), "WAIT_MINUTES=1.5: a whole number"),
                 (env, {}, no_end, "an argument is missing or empty"),
                 (env, {}, self._swap("0", end_in=-10), "has passed"),
                 (env, {}, self._swap("7"), "nvidia-smi lists no GPU 7"),
                 (env, {}, self._swap("0", end_in=8 * 86400), "more than 7 days away"),
                 (env, {"SWAP_LEAD_MINUTES": "30"}, self._swap("0", str(folder / "good.json"), end_in=600),
                  "less than 30 minutes away"),
                 (env, {}, self._swap("0", str(folder / "on_cpu.json")), "has jobs on cpu or mps"),
                 (env, {}, corrupt_swap, "does not match its SHA-256"),
                 (env, {}, other_swap, "was installed from"),
                 (env, {}, self._swap("0", digest="0" * 64), "is not " + "0" * 64),
                 (cpu_env, {}, self._swap("0"), "CPU workers are running"),
                 (round_env, {}, self._swap("0"), f"sessions of the project that use the code are running: "
                                                  f"{self.project}_r1")]
        for test_env, changed, swap, reason in cases:
            with self.subTest(reason=reason):
                done = run(swap, dict(test_env, **changed), check=False)
                self.assertEqual(done.returncode, 1)
                self.assertIn("REFUSED before any change", done.stdout)
                self.assertIn(reason, done.stdout)
                self.assertFalse((self.node / "queue" / "STOP").exists())
                self.assertIn(reason, (self.node / "queue" / "SWAP_FAILED").read_text().splitlines()[-1])
        self.assertEqual(len((self.node / "queue" / "SWAP_FAILED").read_text().splitlines()), len(cases))
        # the code in place is not the tree its archive installed (a file added on the node), or came from no
        # archive with its tree recorded: it could not be put back as it is
        added = self.node / "sweeps" / "zz_added_on_the_node.json"
        added.write_text("{}")
        try:
            self.assertIn("could not be put back as it is", run(self._swap("0"), env, check=False).stdout)
        finally:
            added.unlink()
        versions = self.node / "CODE_VERSIONS.log"
        kept = versions.read_text()
        self.addCleanup(versions.write_text, kept)
        versions.write_text(kept + f"2026-10-28T00:00:00Z\t{'0' * 64}\t{self.code}\n")
        self.assertIn("with its tree digest recorded", run(self._swap("0"), env, check=False).stdout)
        versions.write_text(kept)
        # a Python of the project outside the workers: a notebook, a test
        self._python_of_the_project(60)
        time.sleep(0.5)
        done = run(self._swap("0"), env, check=False)
        self.assertIn("a Python of the project runs outside the workers", done.stdout)
        self.assertFalse((self.node / "queue" / "STOP").exists())
        for fake_log in (log, round_log, cpu_log):
            self.assertFalse(fake_log.exists() and "new -d" in fake_log.read_text())

    def test_swap_timeout_puts_nothing_back_and_starts_a_schedule_with_its_end(self):
        """A worker still on a long job when the wait ends: its job stays in queue/running (it is still running), the
        workers go on, and GPU 0 gets workers again from a schedule that ends them all at the swap's end."""
        env, log, state = self._swap_env("_swap3", ("swap", "gpu1"))
        running = self.node / "queue" / "running"
        running.mkdir(parents=True, exist_ok=True)
        job = running / "zz_long_job.json"
        job.write_text("{}")
        self.addCleanup(job.unlink, missing_ok=True)
        swap = self._swap("0,1")
        done = run(swap, dict(env, WAIT_MINUTES="0"), check=False)
        self.assertEqual(done.returncode, 1)
        self.assertIn("timeout: 1 worker sessions", done.stdout)
        self.assertTrue(job.is_file())
        self.assertEqual(len(self._started(log, 0)), 1)
        self.assertEqual(sum(f"new -d -s {self.project}_gpu1 " in line for line in log.read_text().splitlines()), 0)
        self.assertIn(f"{swap[6]}:end", next(line for line in log.read_text().splitlines()
                                             if line.startswith(f"new -d -s {self.project}_schedule ")))
        self.assertIn("timeout", (self.node / "queue" / "SWAP_FAILED").read_text())

    def _worker_with_a_python(self, state: Path, seconds: int, ends_after: float) -> None:
        """A worker session on GPU 1 whose pane runs a Python of the project for `seconds`; the session itself is
        gone after `ends_after` seconds, the Python still running (a job's process outliving its worker)."""
        busy = self._python_of_the_project(seconds)
        (state / f"{self.project}_gpu1.static").write_text("")
        (state / f"{self.project}_gpu1.panes").write_text(f"{busy.pid}\n")
        timer = threading.Timer(ends_after, lambda: (state / f"{self.project}_gpu1.static").unlink(missing_ok=True))
        timer.start()
        self.addCleanup(timer.cancel)

    def test_swap_waits_for_the_projects_python_to_end(self):
        env, _, state = self._swap_env("_swap4", ("swap",))
        # the Python outlasts the swap's first checks, however slow the disk: the swap must wait for it
        self._worker_with_a_python(state, seconds=12, ends_after=1)
        out = run(self._swap("0"), env, check=False).stdout
        waited = int(out.split("every worker has stopped (after ")[1].split("s)")[0])
        self.assertGreaterEqual(waited, 2)
        self.assertIn("TREE OK", out)

    def test_swap_enqueues_nothing_when_its_end_passed_during_the_wait(self):
        """The schedule is checked again after the install: past the end, the new sweeps are not enqueued and the
        workers stay stopped."""
        env, log, state = self._swap_env("_swap5", ("swap",))
        folder = self._smoke_standin(env)
        # the end comes after the swap's first checks, however slow the disk, and before the Python ends
        self._worker_with_a_python(state, seconds=30, ends_after=1)
        done = run(self._swap("0", str(folder / "good.json"), end_in=15), env, check=False)
        self.assertEqual(done.returncode, 1)
        self.assertIn("TREE OK", done.stdout)
        self.assertIn("after the install, the workers' schedule would be refused", done.stdout)
        self.assertIn("has passed", done.stdout)
        self.assertNotIn("smoke tests on GPU", done.stdout)
        self.assertNotIn('"sweep_id"', done.stdout)
        self.assertTrue((self.node / "queue" / "STOP").is_file())
        self.assertFalse("new -d" in log.read_text())

    def test_swap_keeps_the_code_when_unpack_changes_nothing(self):
        """When unpack refuses before it touches the code, the code stays as it was and the workers start again on
        it, with the failure in queue/SWAP_FAILED (a stand-in unpack.sh that refuses)."""
        env, log, state = self._swap_env("_swap6", ("swap",))
        unpack = self.home / f"{self.project}_incoming" / "unpack.sh"
        kept = unpack.read_text()
        self.addCleanup(unpack.write_text, kept)
        unpack.write_text('#!/usr/bin/env bash\necho "REFUSING: a stand-in that changes nothing"\nexit 1\n')
        before = run(["bash", str(self.node / "node" / "tree_digest.sh"), str(self.node)], self.env).stdout.split()[-1]
        done = run(self._swap("0"), env, check=False)
        self.assertIn("unpack.sh FAILED", done.stdout)
        self.assertIn("swap finished: OLD CODE, nothing enqueued", done.stdout)
        self.assertIn("is in place", (self.node / "queue" / "SWAP_FAILED").read_text())
        after = run(["bash", str(self.node / "node" / "tree_digest.sh"), str(self.node)], self.env).stdout.split()[-1]
        self.assertEqual(after, before)
        self.assertEqual(len(self._started(log, 0)), 1)

    def test_schedule_starts_free_gpus_only_and_ends(self):
        self._venv_standin()
        env, log = self._fakes(5000, False)
        now = int(time.time())
        out = run(["bash", str(self.node / "node" / "schedule.sh"), "3", f"{now}:0,1", f"{now + 2}:end"], env).stdout
        self.assertIn("GPU 0 started: 3 workers", out)
        self.assertIn("GPU 1 NOT started: memory in use: 5000 MiB", out)
        self.assertIn("end: queue/STOP written", out)
        self.assertTrue((self.node / "queue" / "STOP").is_file())
        started = [line for line in log.read_text().splitlines() if f"new -d -s {self.project}_gpu0 " in line]
        self.assertEqual(len(started), 1)
        self.assertIn("ENVIRONMENT=main MIN_FREE_GB=", started[0])
        self.assertIn(" MAX_FAILS=", started[0])
        self.assertIn("/node/worker.sh gpu 0 3", started[0])
        (self.node / "queue" / "STOP").unlink()

    def test_schedule_passes_the_environment_to_its_workers(self):
        """tmux starts a session with the tmux server's environment, so the workers' settings go in the command."""
        self._venv_standin("rapids")
        env, log = self._fakes(100, False)
        env = dict(env, ENVIRONMENT="rapids", MIN_FREE_GB="15", MAX_FAILS="1", THREADS="4")
        now = int(time.time())
        out = run(["bash", str(self.node / "node" / "schedule.sh"), "2", f"{now}:1", f"{now + 1}:end"], env).stdout
        self.assertIn("GPU 1 started: 2 workers", out)
        started = [line for line in log.read_text().splitlines() if f"new -d -s {self.project}_gpu1 " in line]
        self.assertIn("ENVIRONMENT=rapids MIN_FREE_GB=15 MAX_FAILS=1 THREADS=4 bash", started[-1])
        (self.node / "queue" / "STOP").unlink()

    def test_schedule_checks_every_step_before_starting_anything(self):
        """A step whose time `date -d` could not read arrives empty; without an end, or with an end that has passed,
        the workers would have no stop time. Each is refused before any worker starts."""
        self._venv_standin()
        env, log = self._fakes(100, False)
        now = int(time.time())
        script = str(self.node / "node" / "schedule.sh")
        # settings that would make every worker die at once
        for changed, reason in [({"ENVIRONMENT": "mian"}, "no environment"), ({"MIN_FREE_GB": "15G"}, "MIN_FREE_GB='15G'"),
                                ({"MAX_FAILS": "x"}, "MAX_FAILS=x"), ({"THREADS": "0"}, "THREADS=0")]:
            with self.subTest(changed=changed):
                done = run(["bash", script, "3", f"{now}:0", f"{now + 60}:end"], dict(env, **changed), check=False)
                self.assertEqual(done.returncode, 1)
                self.assertIn(reason, done.stdout)
                self.assertIn("nothing started", done.stdout)
        for steps, reason in [([f"{now}:0"], "the last step must be <time>:end"),
                              ([f"{now}:0", ":end"], "the time must be seconds since 1970"),
                              ([f"{now - 100}:0", f"{now - 10}:end"], "has passed"),
                              ([f"{now + 100}:0", f"{now + 50}:end"], "comes before the step ahead of it"),
                              ([f"{now}:0;1", f"{now + 60}:end"], "give GPU indices"),
                              ([f"{now}:0", f"{now + 60}:end", f"{now + 120}:1"], "an end must be the last step"),
                              ([f"{now}:0", f"{now + 8 * 86400}:end"], "is more than 7 days away"),
                              ([f"{now}:0,7", f"{now + 60}:end"], "nvidia-smi lists no GPU 7")]:
            for check in ([], ["--check"]):
                with self.subTest(steps=steps, check=check):
                    done = run(["bash", script, *check, "3", *steps], env, check=False)
                    self.assertEqual(done.returncode, 1)
                    self.assertIn("REFUSING", done.stdout)
                    self.assertIn(reason, done.stdout)
                    self.assertIn("nothing started", done.stdout)
        self.assertIn("a whole number from 1", run(["bash", script, "0", f"{now}:0", f"{now + 60}:end"], env,
                                                   check=False).stdout)
        self.assertFalse(log.exists() and "new -d" in log.read_text())
        # --check: the same checks, nothing started and nothing written
        (self.node / "queue" / "STOP").unlink(missing_ok=True)
        logs_before = sorted((self.node / "logs").glob("schedule_*"))
        done = run(["bash", script, "--check", "3", f"{now}:0,1", f"{now + 3600}:end"], env)
        self.assertIn("SCHEDULE OK: 2 steps; the workers stop at", done.stdout)
        # --check wants the end far enough ahead for the work before the workers start (30 minutes unless set)
        near = run(["bash", script, "--check", "3", f"{now}:0", f"{now + 600}:end"], env, check=False)
        self.assertEqual(near.returncode, 1)
        self.assertIn("is less than 30 minutes away", near.stdout)
        self.assertIn("SCHEDULE OK", run(["bash", script, "--check", "3", f"{now}:0", f"{now + 600}:end"],
                                         dict(env, SCHEDULE_MIN_LEAD_MINUTES="5")).stdout)
        self.assertEqual(sorted((self.node / "logs").glob("schedule_*")), logs_before)
        self.assertFalse(log.exists() and "new -d" in log.read_text())
        self.assertFalse((self.node / "queue" / "STOP").exists())

    def _smoke_standin(self, env):
        """A stand-in environment for smoke_enqueue.sh: `check` runs with this Python; its smoke test prints the device
        and GPU it was given and fails for a sweep file whose name says bad; enqueue refuses one whose name says
        refused. The sweep files are the flow check's regression sweep under other sweep IDs (fit.device auto), and one
        with a stage that puts fit.device on cpu."""
        python = self.home / "venvs" / f"{self.project}-main" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            case "$*" in
                *" check "*) exec {shlex.quote(sys.executable)} "$@" ;;
                *" smoke "*) echo "sweep $5 on $7, GPU [${{CUDA_VISIBLE_DEVICES-unset}}]"
                             case "$5" in *bad*) echo "SMOKE FAILED (1 of 1 configurations)"; exit 1 ;; esac
                             echo "SMOKE OK" ;;
                *" enqueue "*) case "$*" in *refused*) echo "SweepError: follows 'nowhere'"; exit 1 ;; esac
                               shift 4; for f in "$@"; do echo " \\"sweep_id\\": \\"$(basename "$f" .json)\\","; done ;;
            esac
            """))
        python.chmod(0o755)
        self.addCleanup(shutil.rmtree, self.home / "venvs", ignore_errors=True)
        folder = Path(self.tmp.name) / "smoke_sweeps"
        folder.mkdir(exist_ok=True)
        source = json.loads((REPO / "sweeps" / "checks" / "flow_regression.json").read_text())
        for name in ("good", "good2", "bad", "refused", "on_cpu"):
            sweep = dict(source, sweep_id=f"zz_smoke_{name}")
            if name == "on_cpu":
                sweep["stages"] = source["stages"] + [{"name": "cpu", "kind": "one_factor_at_a_time", "priority": 1,
                                                       "seeds": [1], "factors": {"fit.device": ["cpu"]}}]
            (folder / f"{name}.json").write_text(json.dumps(sweep))
        return folder

    def test_smoke_enqueue_tests_on_a_free_gpu_of_the_project_and_fails_when_a_sweep_is_not_enqueued(self):
        env, _ = self._fakes(5000, False)           # GPU 1 busy, GPU 0 free
        env.pop("CUDA_VISIBLE_DEVICES", None)
        folder = self._smoke_standin(env)
        script = str(self.node / "node" / "smoke_enqueue.sh")
        good, bad = str(folder / "good.json"), str(folder / "bad.json")
        out = run(["bash", script, good, bad], dict(env, SMOKE_DEVICE="cuda", SMOKE_GPUS="1,0"), check=False)
        self.assertEqual(out.returncode, 1)
        self.assertIn("smoke tests on GPU 0", out.stdout)
        self.assertIn(f"sweep {good} on cuda, GPU [0]", out.stdout)
        self.assertIn(f"sweep {bad} on cuda, GPU [0]", out.stdout)
        self.assertIn("NOT ENQUEUED: zz_smoke_bad", out.stdout)
        self.assertIn("NOT ENQUEUED: 1 of 2 sweeps failed their smoke test, so none is enqueued", out.stdout)
        self.assertNotIn('"sweep_id"', out.stdout)          # the good sweep waits too: a round runs whole or not at all
        self.assertNotIn("ALL DONE", out.stdout)
        on_cpu = run(["bash", script, str(folder / "on_cpu.json")], dict(env, SMOKE_DEVICE="cuda", SMOKE_GPUS="0"),
                     check=False)
        self.assertEqual(on_cpu.returncode, 1)
        self.assertIn("has jobs on cpu|mps, which workers on cuda do not take", on_cpu.stdout)
        both = run(["bash", script, good, str(folder / "good2.json")], dict(env, SMOKE_DEVICE="cuda", SMOKE_GPUS="0"))
        self.assertEqual(both.stdout.count('"sweep_id"'), 2)
        self.assertIn("ALL DONE", both.stdout)
        # both pass their smoke tests, and enqueue refuses one (it follows a sweep the node has no record of): one
        # enqueue for all, so none goes in
        refused = run(["bash", script, good, str(folder / "refused.json")], dict(env, SMOKE_DEVICE="cuda",
                                                                                   SMOKE_GPUS="0"), check=False)
        self.assertEqual(refused.returncode, 1)
        self.assertIn("SweepError: follows 'nowhere'", refused.stdout)
        self.assertIn("NOT ENQUEUED: enqueue refused the sweeps", refused.stdout)
        self.assertNotIn("ALL DONE", refused.stdout)
        refused = run(["bash", script, good], dict(env, SMOKE_DEVICE="cuda", SMOKE_GPUS="1"), check=False)
        self.assertEqual(refused.returncode, 1)
        self.assertIn("no free GPU among 1 for the smoke tests", refused.stdout)
        self.assertNotIn("sweep ", refused.stdout)
        cpu = run(["bash", script, good], dict(env, SMOKE_DEVICE="cpu"))
        self.assertIn(f"sweep {good} on cpu, GPU []", cpu.stdout)
        self.assertIn("ALL DONE", cpu.stdout)

    def test_schedule_refuses_while_slurm_runs(self):
        env, log = self._fakes(100, True)
        done = run(["bash", str(self.node / "node" / "schedule.sh"), "3", f"{int(time.time())}:0"], env, check=False)
        self.assertEqual(done.returncode, 1)
        self.assertIn("slurmd is active", done.stdout)
        self.assertFalse(log.exists())

    def _plan(self, name: str, content: bytes) -> Path:
        """A plan where round-new writes one, under sweeps/rounds/ (a code item); removed after the test, so the
        installed code keeps its digest for the other tests."""
        rounds = self.node / "sweeps" / "rounds"
        prep = rounds / f"{name}_prep_20261028"
        made = rounds if not rounds.exists() else prep
        prep.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, made)
        (prep / "plan.txt").write_bytes(content)
        return prep / "plan.txt"

    def test_chain_stops_at_the_first_failure(self):
        self._plan("c1", b"# a comment\necho one\nfalse\necho three\n")
        done = run(["bash", str(self.node / "node" / "chain.sh"), "sweeps/rounds/c1_prep_20261028/plan.txt"], self.env,
                   check=False)
        self.assertEqual(done.returncode, 1)
        self.assertIn("CHAIN STOPPED at step 2", done.stdout)
        logs = self.node / "logs" / "c1"                     # the round's name, without _prep_<date>
        self.assertEqual((logs / "1_echo.txt").read_text(), "one\n")
        self.assertFalse((logs / "3_echo.txt").exists())

    def test_chain_names_each_steps_log_after_the_script_it_runs(self):
        plan = self._plan("c3", b"bash node/tree_digest.sh . > /dev/null && echo hashed\n"
                                b"env A=1 bash node/disk_check.sh 0 > /dev/null; echo checked\necho done\n")
        done = run(["bash", str(self.node / "node" / "chain.sh"), str(plan)], self.env, check=False)
        self.assertEqual(done.returncode, 0, done.stdout)
        logs = self.node / "logs" / "c3"
        self.assertEqual((logs / "1_tree_digest.txt").read_text(), "hashed\n")
        self.assertEqual((logs / "2_disk_check.txt").read_text(), "checked\n")
        self.assertEqual((logs / "3_echo.txt").read_text(), "done\n")

    def test_chain_drops_windows_line_endings(self):
        plan = self._plan("c2", b"# saved on Windows\r\necho one\r\ntest one = one && echo same\r\n")
        done = run(["bash", str(self.node / "node" / "chain.sh"), str(plan)], self.env, check=False)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("CHAIN DONE, 2 steps", done.stdout)
        logs = self.node / "logs" / "c2"
        self.assertEqual((logs / "1_echo.txt").read_bytes(), b"one\n")
        self.assertEqual((logs / "2_test.txt").read_bytes(), b"same\n")


def first_line(doc: Path) -> str:
    """The read-only first line on a node: the fenced block of the doc that lists the tmux sessions."""
    blocks = [block.strip() for block in doc.read_text(encoding="utf-8").split("```")[1::2]]
    found = [block for block in blocks if "--- your tmux sessions" in block]
    if len(found) != 1:
        raise AssertionError(f"{doc.name}: {len(found)} blocks with the first line")
    return found[0]


@unittest.skipUnless(TOOLS and shutil.which("ps"), "needs bash and ps")
class TestFirstLine(unittest.TestCase):
    def test_the_first_line_is_the_same_in_both_docs_and_only_reads(self):
        """The line runs with stand-ins for nvidia-smi, systemctl, tmux and curl in a temporary home folder: it
        prints every part, names the user of a process on a GPU, and leaves the home folder and the folder it runs
        from exactly as they were."""
        line = first_line(REPO / "docs" / "NODE.md")
        self.assertEqual(line, first_line(REPO / "TESTING.md"))
        with tempfile.TemporaryDirectory() as d:
            home, here, bin_dir = Path(d) / "home", Path(d) / "here", Path(d) / "bin"
            for folder in (home / "project" / "data", here, bin_dir):
                folder.mkdir(parents=True)
            (home / "project" / "data" / "rows.csv").write_text("a,b\n" * 1000)
            (home / ".bashrc").write_text("# shell\n")
            pid = os.getpid()
            stand_ins = {
                "nvidia-smi": f"""case "$*" in
                    *query-compute-apps*) echo "GPU-aaaa, {pid}, 62 MiB" ;;
                    *query-gpu*) echo "0, GPU-aaaa, NVIDIA H100 NVL, 610.57.04, 70 MiB, 95830 MiB, 0 %" ;;
                esac""",
                "systemctl": 'echo inactive; exit 3',
                "tmux": 'echo "no server running" >&2; exit 1',
                "curl": 'printf 200',
            }
            for name, body in stand_ins.items():
                (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{textwrap.dedent(body)}\n")
                (bin_dir / name).chmod(0o755)

            def state():
                return sorted((p.relative_to(d).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                              for p in Path(d).rglob("*") if p.is_file() and bin_dir not in p.parents)

            before = state()
            env = dict(os.environ, HOME=str(home), PATH=f"{bin_dir}:{os.environ['PATH']}")
            done = run(["bash", "-c", line], env, cwd=here)
            out = done.stdout
            for part in ("--- the largest items in the home folder", "--- CPUs: ", "NVIDIA H100 NVL, 610.57.04",
                         "--- your tmux sessions\nnone", "--- your crontab\nnone", "--- slurmd: inactive",
                         "python3.12: ", "pypi.nvidia.com: 200", "huggingface.co: 200"):
                self.assertIn(part, out)
            # whole, not cut to 8 characters with a + (the first node run, 2026-10-27)
            user = run(["ps", "-o", "user:32=", "-p", str(pid)], env).stdout.strip()
            self.assertIn(f"GPU-aaaa, {pid}, 62 MiB: {user}", out)
            self.assertIn("project", out)
            self.assertEqual(state(), before)


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class TestWithoutNode(unittest.TestCase):
    def test_core_tests_and_a_local_sweep_run_without_node(self):
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "repo"
            copy.mkdir()
            for item in ("src", "tests", "sweeps", "templates", "project.json"):
                source = REPO / item
                if source.is_dir():
                    shutil.copytree(source, copy / item, ignore=shutil.ignore_patterns("__pycache__"))
                else:
                    shutil.copy(source, copy / item)
            env = dict(os.environ, PYTHONPATH=str(copy / "src"))
            env.pop("SWEEPKIT_REQUIRE", None)
            tests = run([sys.executable, "-m", "unittest", "tests.test_settings", "tests.test_sweep", "tests.test_store",
                         "tests.test_hashing", "tests.test_stats"], env, cwd=copy)
            self.assertIn("OK", tests.stderr)
            local = run([sys.executable, "-m", "sweepkit.cli", "run", "--sweep-file",
                         "sweeps/examples/cluster_kmeans.json", "--device", "cpu", "--max-consecutive-failures", "1"],
                        env, cwd=copy)
            self.assertIn("'failed': 0", local.stdout)
            self.assertTrue((copy / "artifacts" / "runs" / "example_cluster_kmeans").is_dir())


if __name__ == "__main__":
    unittest.main()
