# How we work

The rules a project built on this template follows, for people and for Claude, each with the reason it exists. Every
rule came from a mistake that cost time in the project the template grew out of: a month of nightly sweeps on a
shared GPU node, more than 37,000 runs, with results read by an advisor. Claude follows them without being asked
(CLAUDE.md) and says so when a request would break one.

## Who does what

- Claude prepares everything: sweep files, scripts, the exact command lines, analyses and write-ups, and checks each
  one before handing it over (a sweep file with `check` and `smoke`, a script on stand-in data with a known answer).
- The person runs every command on a shared node and pastes the output back. Claude compares it with what the line
  said it should print before giving the next line. A line that prints something unexpected stops the sequence until
  it is understood. Why: a pasted line cannot be taken back once it has run on a machine other people use.
- One step at a time, each with its shell named (Command Prompt, PowerShell, bash on the node), chained from its `cd`
  with `&&`, and with what it should print. Why: lines written for one shell and pasted into another once ran git in
  the wrong folder.

## A shared GPU node

- The node's rules are written down as the owner or administrator states them (hours, GPUs, disk, how much may stay
  queued), in PROJECT.md, and nothing stricter or looser is assumed. Why: a guessed rule was once stricter than the
  real one and wasted nights, and once looser.
- Only GPUs no one else is using: no process of another user and no more than a little memory in use (an idle model
  server can hold most of a GPU's memory at 0% use). `node/free_gpus.sh` decides; the schedule checks again before it
  starts workers.
- Work runs in tmux sessions named after the project, which end by themselves: an empty queue or the night's stop
  time (`node/schedule.sh` writes `queue/STOP`). Never start anything that restarts itself or runs forever, and never
  touch a session the project did not start. Why: a dropped SSH connection then costs nothing, and a forgotten run
  cannot hold the GPUs into the day.
- Never change the code while workers run: stop them, wait until no session of the project is left, then install
  (`node/unpack.sh` refuses otherwise). Why: a running worker imports some modules late and would mix old and new
  code.
- Install by moving the old code aside, never by deleting it in place (a network file system keeps a deleted file
  that is still open, and the delete fails half way); `node/unpack.sh` keeps the code it replaced last and deletes
  the copies moved aside before that. Check the installed tree's digest against the one computed where the code was
  written.
- Check the free disk before copying data or starting a night, counting what will be written (`estimate`,
  `node/disk_check.sh`), and keep free what anything else in the same home folder needs. Workers stop below a floor.
  Why: a home folder on a shared node often has its own limit, far smaller than the drive it sits on, and every run
  that writes there stops when it fills; on a node without such limits, everyone's runs stop.
- If Slurm's `slurmd` is active, the schedule refuses to start: Slurm may hand the same GPUs to batch jobs.

## What makes a number trustworthy

- Every number comes from a file the code wrote, rounded half up once from the stored value (0.6006 shows as 0.601).
  Nothing is estimated, typed by hand or remembered.
- Choices use validation files only, or label-free criteria for unsupervised work. `analyze` compares
  configurations on those files; `select` records the choice with its time; only then do `test-report` or
  `reveal-report` open the files that judge it. The code enforces the order: a choosing step that asks for a judging
  file is refused. Why: a test number used to choose is no longer a test number.
- Before results exist, a RULE file says which runs are made, which judging files will be read, which comparisons
  will be made (families of comparisons, paired bootstrap, Holm's correction within each family), what happens to
  unfinished runs, and what result would change the plan. `rule-fix` stores it with its UTC time and SHA-256, and
  the number of runs of its sweeps that exist already: a RULE fixed after some exist needs a recorded reason, which
  every report that uses it prints. Once a judging file has been read, the rule does not change; a change is a dated
  amendment, with both versions kept. Why: this is what makes a p-value mean what it says.
- Seeds: three to screen many configurations, five new seeds to confirm and report, one only for learning and smoke
  tests. The best of many configurations on the same seeds is optimistic; a gain is claimed only after it is
  confirmed on seeds no choice has used.
- Controls: a shuffled-label control (supervised) should sit at chance; above chance means something leaks. The
  identity fold, whose validation rows are its test rows, checks that the two evaluation paths give the same numbers;
  no choosing step reads it.
- A setting that uses the inputs of validation or test rows, even without their labels (for example normalization
  statistics per subject over every row), is transductive: it is marked, and every report that uses it says so.
- Unsupervised work: a label-free criterion chooses (for example DBCV for density-based clustering, silhouette for
  k-means), stability across seeds and subsamples is measured, and agreement with any known labels is read only after
  the choice is recorded, as a test set is.
- A comparison reports the difference, its interval and the Holm-adjusted p, with a fixed verdict word: better, worse,
  or no detectable difference. Not "slightly better".

## Records

- A round (one night on a node, or one batch on one machine) has a preparation folder,
  `artifacts/analyses/<round>_prep_<date>/` (README, RULE, the lines), and a reading folder,
  `artifacts/analyses/<round>_<date>/` (what ran, what was read when, what it found). `round-new` makes the first,
  with the plan the node runs in `sweeps/rounds/<round>_prep_<date>/plan.txt`, beside the round's sweep files: only
  the code travels to the node, and the tree digest the lines check covers the plan, so the node runs exactly the
  plan that was prepared. `round-check` checks it. docs/EXAMPLE_ROUND.md shows a complete one.
- Run folders are written once and never overwritten or deleted. A failed job's folder is kept as evidence.
- Every file that crosses between machines is checked by SHA-256, file by file (`node/pack.sh`, `node/pc/pull.*`).
- STATE.md says where the work stands and is read first by every session. VERIFICATION_LOG.md holds one entry per
  check of data or code, with the file each number came from. findings.csv lists each claim with the runs and test
  behind it. These and data, results, logs and logins stay out of git (`.gitignore`).

## Writing results

- A results doc per round, a meeting summary before each meeting, and a standing "how it works" page
  (`templates/`, docs/REPORTING.md). Each is checked before anyone else reads it: nothing in it the readers should not
  see, and every number checked against its file by a separate read.
- Plain prose in the first person, short sentences, a number with its source rather than a characterisation, tables
  for more than three comparable values. Say what a result shows and what it does not; if something was not tested,
  say so.

## Working with Claude

- Claude opens a file before saying what is in it, and names it. It keeps three states apart: verified against a
  named file; not found in the files checked (it names them); contradicted by a named file. Not found does not mean
  wrong.
- It says what it did not check. An output that reports something missing or broken is a hypothesis until a second
  check confirms it, and a checker is first tried on a case with a known answer.
- Before a long run, it states the cost (jobs, GPU hours, disk) and the stop condition.
- It changes only what was named, and a constraint stated once stands until it is lifted.
- It never runs git inside the project's repository from a sandbox that cannot delete git's lock files (the Cowork
  virtual machine); it gives the person the lines instead.
- Working from a cloud container with the person's folder linked, it edits files in place on the computer, checks
  every file it copies either way by its SHA-256 where it arrives, and brings work built in the container back as one
  payload that installs only onto the files it was built from (`src/sweepkit/payload.py`; CLAUDE.md, "From a cloud
  container to the person's computer"). Why: a copy tool once handed back an earlier version of a file that had just
  been replaced, and only the hash showed it; and work installed onto files changed in the meantime mixes two
  versions that nobody tested together.
- It measures a node before asking about it (the first line of docs/NODE.md) and asks the person only what the node
  cannot show: hours, cutoff, GPUs, VPN, and the other projects on the same account. Why: a guessed free-disk figure
  or GPU owner costs a night, and a question the output already answers costs the person's time.
