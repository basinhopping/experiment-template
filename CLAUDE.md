# CLAUDE.md

You are working in a copy of the experiment template: a small Python package (`sweepkit`, in `src/`) and the scripts
around it, for running sweeps of model configurations on any data, supervised or unsupervised, on one machine or a
shared GPU node, and for reading the results with a fixed discipline so that the numbers can be trusted. The person
you work with may be new to the code, to Claude, or to machine learning. Take them through each step, say briefly what
it does, and say what it should print.

## Read first

1. `STATE.md`, if it exists: where the work stands. Then `PROJECT.md` (the project's brief) and `STARTING_PLAN.md`.
2. This file and `docs/HOW_WE_WORK.md` (the rules, each with its reason). Follow them without being asked, and say
   so when a request would break one.
3. For the step at hand: `docs/CONCEPTS.md` (how it works), `docs/COMMANDS.md` (every command, with an example),
   `docs/DATA.md`, `docs/UNSUPERVISED.md`, `docs/EXTENDING.md`, `docs/PROFILES.md`, `docs/NODE.md`,
   `docs/REPORTING.md`, `docs/WALKTHROUGH.md` (one result end to end) and `docs/EXAMPLE_ROUND.md` (a complete round).

No `STATE.md` and no `PROJECT.md` means a new project: start with "A new project" below.

## How the repository works

- A job is one configuration on one fold with one seed. Its settings have seven sections: `data`, `split`, `task`,
  `prep` (a list of steps), `features`, `model`, `fit`. Every section names a registered component and gives every
  setting it takes; nothing has a default, so a settings file says everything a run did.
- A sweep file (JSON) holds a complete base configuration and stages that vary it (`base`, `variants`,
  `one_factor_at_a_time`, `grid`, `random`), with seeds, folds and priorities. `check` expands it into jobs, each
  named by the hash of its settings; `smoke` runs every configuration once on a few rows; `enqueue` puts the jobs in a
  file queue, or `run` runs them in one process.
- Each job writes one run folder, once, with its provenance (code hash, package versions, device actually used, the
  folds' hash). A runner per model family: `torch`, `estimator` (any scikit-learn-style model), `clusterer`,
  `embedder`, `detector`.
- Reading, in a fixed order: `analyze` compares configurations on validation files (or label-free criteria) only;
  `select` records the choice; `rule-fix` stores a RULE file, written before the results, that says what will be
  read; only then `test-report`, `reveal-report` and `test-compare` open the files that judge the choice. The code
  refuses a judging read of a selection without a RULE fixed before the selection was recorded, refuses
  `test-compare` on a configuration whose judging files were read before its RULE was fixed, and refuses a RULE fixed
  after runs of its sweeps exist unless `--late-reason` records why (every report that uses it then says so). Every
  judging read is recorded (`reading.json`).
- The project's own components (datasets, models, steps, metrics) go in `src/project/`, written with `scaffold`;
  the template's files stay untouched, so fixes from the template merge cleanly (`git pull upstream main`).

## Hard rules

1. No credentials, personal details, data or results in git. A node login lives in `node.local.txt` (one line,
   ignored by git). Never ask anyone for a password or a login, never use anyone else's, and never log in to a node:
   commands on a shared node are the person's to run.
2. Never run git inside the person's repository from the Cowork virtual machine: it cannot delete git's lock files,
   and a stale lock breaks the repository for the person. Give them the git lines for their own terminal, written for
   one named shell (Command Prompt, PowerShell or bash), each chained from its `cd` with `&&`.
3. Choices use validation files or label-free criteria only. Never open a test file, a reveal file or a judging
   column by any other route (a text editor, pandas, a notebook) before `select` has recorded the choice and a RULE
   covers it. When unsure whether a file judges, treat it as if it does.
4. Seeds: three to screen, five new seeds to confirm and report, one for learning and smoke tests only. A method
   that gives the same result for every seed (HDBSCAN, DBSCAN, agglomerative clustering) is screened on one seed and
   more subsamples instead (docs/UNSUPERVISED.md).
5. Run folders are written once. Never overwrite, edit or delete one; a failed run's folder is evidence.
6. Every number you report comes from a file the code wrote, rounded half up once from the stored value. Never
   estimate, recall or retype a result.
7. Test before handing anything over: `check` and `smoke` for a sweep file, a script on stand-in data with a known
   answer, SHA-256 for every file that crosses between machines.
8. On a shared node: only GPUs no one else is using, a stop time on every night's workers, tmux sessions that end by
   themselves, never another user's session, the free disk measured before data is copied, never a code change while
   workers run (docs/NODE.md).
9. Look up current package versions and APIs (the package's own documentation, its changelog, its source at the
   pinned version) instead of assuming them. The pins in `env/` are exact, except two that are written down: pip
   itself, which the setup scripts upgrade, and the ceiling `setuptools<82` in the automl profile (its reason is in
   that file). A new package gets an exact pin too.
10. Change only what was named. A constraint stated once stands until the person lifts it.

## A new project

Interview the person before writing anything. Ask, a few questions at a time, and write down each answer with who
gave it:

- the question the project asks, and what result would answer it;
- the data: where it is, its format, size (files, rows, columns, bytes), what one row is, labels or targets or none,
  groups that must stay together (subjects, patients, sites, batches), licences and privacy;
- the protocol: how rows are split, what is held out, what may never be used to choose;
- the number that chooses (a validation metric, or for unsupervised work a label-free criterion) and its direction,
  and the numbers that judge afterwards (test metrics, or agreement with reference labels revealed later);
- the models or algorithms wanted, and what the field uses;
- the compute: laptop CPU, home GPU, Mac, shared node, and a shared node's rules as its owner states them;
- existing code: where it is and how it is run;
- who reads the results and what they need.

For a shared node, in this order:

1. The login stays with the person. They write it (`user@host`, one line) into `node.local.txt` beside README.md
   themselves, and the push and pull scripts read it there. Never ask for it or for a password, and never write
   either into a line or a file.
2. Measure before asking. The person logs in and runs the read-only first line of docs/NODE.md ("The first time on a
   node", step 1). It shows the home folder's size, limit, free space and largest items; the CPUs and memory; each
   GPU's driver and memory in use, with every process on it and that process's user; the person's own tmux sessions
   and crontab; whether Slurm's `slurmd` is active; the Python versions; and whether the node reaches PyPI, NVIDIA's
   package index, GitHub and Hugging Face. Say back what it shows in a few lines and write it into PROJECT.md
   ("Compute") with its date. Any question the output answers is not asked.
3. Then ask only what the person knows and the node cannot show: the hours the project may use the GPUs, and any
   cutoff its work must stop by; which GPUs it may use; whether reaching the node needs a VPN; and the other projects
   on the same account, with the hours and GPUs each uses and the free disk each needs (a floor below which its
   workers stop, for example). The floor this project keeps free and its GPU window are set from these answers.

Then write `PROJECT.md`, `STARTING_PLAN.md` and `STATE.md` from `templates/`, set the project's name in
`project.json` (lowercase letters, digits and `_`), and look at the data before any model code: `profile-data`, with
`--hide` naming the reference labels and any other column that judges a choice, and the findings in
`VERIFICATION_LOG.md` entry 1 (`templates/VERIFICATION_LOG_entry.md`).

## Existing code

When the person has code (scripts, notebooks, a pipeline):

1. Read all of it before changing anything: what it reads, every setting and its default, what it writes, its random
   seeds, its library versions.
2. Run it unchanged on a small sample, and keep its output.
3. Keep its behaviour. Wrap it behind the template's interfaces: its data loading as a dataset (`table`, `store` or a
   `scaffold --kind dataset`), its method as a registered model (`scaffold --kind clusterer`, `estimator`, ...) with
   every setting it uses, library defaults included, as explicit settings.
4. Run the template's version on the same sample with the same settings and seed, and compare: for a deterministic
   clustering, adjusted Rand index 1.0 between the two label sets; for a model, the same predictions, or metrics within
   a tolerance you state with its reason. Record the comparison in `VERIFICATION_LOG.md`.
5. Only then vary it: the person's settings become the sweep's base configuration.

## From a laptop's CPU to a node's GPU

A common first request is a script too slow on a laptop, for example HDBSCAN on one large CSV. docs/NODE.md has the
steps: measure the data and the node's free disk first; send the data with SHA-256; run the script unchanged under
cuML's zero-code-change mode (`node/rapids_run.sh`) on a free GPU and read its fallback summary; bring the output
home with `pack` and `pull`; then build the template's version as a sweep and check it against the script's output.
For HDBSCAN on many columns, reduce first (docs/UNSUPERVISED.md).

## From a cloud container to the person's computer

Some sessions run in a cloud container with the person's folder linked from their computer (a Cowork session linked
to the computer, for example): a shell that runs on the computer and sees the folder, and tools that copy a file
between the computer and the container. The repository stays on the computer.

1. Edit in place on the computer, through its shell: change a file where it is (`sed -i`, or a short script that
   reads it, changes it and writes it back), never by retyping its contents from a tool's output, which can be cut
   short. Most work goes this way.
2. Work that needs the container (a package the computer's shell cannot install, a long run, many files at once)
   starts from a snapshot. On the computer, from the repository:
   `python3 src/sweepkit/payload.py snapshot . artifacts/payloads` packs the code items with every file's SHA-256
   (paths after the folder pack only those; `artifacts/` is outside the code and out of git). The tool needs only
   Python 3.8 or newer.
3. Every copy is checked by its SHA-256 where it arrives. In the container: `sha256sum -c` on the archive's
   `.sha256`, then unpack into a new empty folder and run `payload.py check <snapshot>.json <that folder>`. A copy
   that fails its check, or a copy tool that fails, is copied again: a copy tool can also hand back an earlier
   version of a file, which only the hash shows. After two failures, stop and say so.
4. Files built in the container go back as one payload:
   `payload.py make <snapshot>.json <worked folder> <out folder> [<paths of results made there>]` writes the changed
   and new files with MANIFEST.sha256, BEFORE.sha256 (each replaced file's hash in the snapshot), NEW.txt,
   REMOVED.sha256 and DIGEST.txt (the code-tree digest after the install), and packs them with the installer. Copy the
   archive and its `.sha256` into `artifacts/payloads/` on the computer, check it there, unpack it, and from the
   repository run `python3 artifacts/payloads/<payload>/payload.py install .`. It refuses, changing nothing, when a
   payload file differs from its manifest, a file it replaces or removes is not the snapshot's version, a new file
   exists with other content, or any other code changed since the snapshot (the digest would differ). Otherwise it
   writes each file in place, moves removed files aside (it never deletes) and checks the manifest and the digest.
   After a refusal, take a new snapshot and rebuild on it: what is installed is what was tested.
5. Data goes over file by file with its SHA-256. A test file, a reveal file or any file holding a judging column
   reaches the container only after `select` and `rule-fix` have written the selection and its RULE (hard rule 3).
6. Git stays on the computer and the person runs it (hard rule 2).

## Adapting the template to the project

1. Data: `ingest` a received folder (a verified copy with a manifest) or put the file under `data/`; `convert` a CSV
   larger than memory into a store; `profile-data` it (docs/DATA.md).
2. A dataset section: usually `table` or `store` with settings and no code; `items`, `graph` or `anndata` for those
   kinds; a new dataset module only for an unusual format.
3. The split and the task: built-in ones, chosen in the settings (docs/CONCEPTS.md).
4. Preparation and models: built-in ones (docs/PROFILES.md lists every component with its profile); anything missing
   through `scaffold` (docs/EXTENDING.md), which writes the module, its catalog line, an example sweep and a test.
5. The first sweep: the person's current settings as the base, on a sample or subsamples. `check`, `smoke`, run,
   `analyze`.
6. The tests, `check-additions` and `smoke-all` for the profiles in use (TESTING.md).
7. Then rounds: screens on three seeds, confirmations on five new seeds, a RULE before each reading.

Unsupervised work follows the same flow with its own choices (docs/UNSUPERVISED.md): subsamples choose the settings
and measure stability, a label-free criterion chooses, the chosen configuration then runs once on every row (split
`all`), reference labels are revealed only after the choice, and `export` and `cluster-report` give the person the
results to take home.

## A round

One night on a node or one batch on one machine (docs/HOW_WE_WORK.md, docs/EXAMPLE_ROUND.md):

1. Prepare: write the sweeps, `check` and `smoke` them, then `round-new` makes
   `artifacts/analyses/<round>_prep_<date>/` (README, RULE, numbered lines) and the plan the node runs,
   `sweeps/rounds/<round>_prep_<date>/plan.txt`, which travels with the code (push sends only the code; the tree
   digest in the lines covers the plan). A round's workers run on GPUs, so its sweeps use `fit.device` `auto` (a
   method without a GPU path then runs on the CPU), or `cuda` for one that must have the GPU. Name the node's
   environment and its profiles (`--environment`, `--profiles`), as built there, the GPUs the project may use
   (`--gpus`) and the floor of free disk agreed for it (`--min-free-gb`). Fill the RULE with the person, state the
   cost (`estimate`: jobs, GPU hours, disk) and the stop condition. A change to the code or the plan after this
   makes the lines' digest wrong: `round-check` says so; prepare the round again under a new round name or date.
2. Fix the rule: `rule-fix` before any result exists; `round-check` passes.
3. Run: the person pastes one line at a time; compare each output with the line's `# expect:` before the next.
4. Read, in the fixed order: pull (every file checked), `analyze`, `select`, then `test-report` or `reveal-report`
   and `test-compare` as the RULE says.
5. Record: the reading folder `artifacts/analyses/<round>_<date>/` with its README, `findings.csv`,
   `VERIFICATION_LOG.md`, `STATE.md`.
6. Report: a results doc (`templates/results_doc.md`), checked before anyone else reads it.

## Lines for the person

Each step is one line in its own file under the round's `lines/`: it names its shell and machine, starts from a `cd`
chained with `&&`, prints "stop here and paste this" on anything unexpected (a session of the project still running,
a tree digest that differs, a GPU someone else is using, too little disk, `slurmd` active), logs its own output under
`logs/<round>/` when it runs on the node, and lists what it should print as `# expect:` comments. Give one line, wait
for its output, compare, then give the next. `round-check` checks every line.

## Extending

A model, algorithm, transformation, metric, data reader, split or stage kind that is not here: `scaffold` writes it,
and docs/EXTENDING.md has one worked example of each kind. `check-additions` refuses it until a test names it and,
except for a metric, an example sweep uses it; `smoke-all` runs every component through the real runner (a metric
runs inside the runs that compute it, a stage kind when its example sweep is checked and smoked). Put the project's
own components in `src/project/` (`--where project`). Look up the library's current API before calling it, pass every
parameter explicitly, and give each new package an exact pin in the right profile.

## Reports

Results docs, meeting summaries and the "how it works" page come from `templates/`; docs/REPORTING.md has the
conventions. They go in `reports/` (kept out of git) and, when the session has a Claude Project, in the project as
well. Before anyone else reads one: nothing in it its readers should not see, and every number checked against its
file, for an important document by a separate agent that did not write it.

## How to answer

- Open a file before saying what is in it, and name the file and the version you read.
- Keep three states apart: verified against a named file; not found in the files checked (name them); contradicted
  by a named file. Not found does not mean wrong.
- Say what you did not check. An output that reports something missing or broken is a hypothesis until a second
  method confirms it; try a checker on a case with a known answer first.
- A correction needs more sourcing than the claim it corrects. A review from another session is a claim: check each
  item against the current file.
- Before a long run, state its cost in the unit that matters and write down the stop condition.
- Before running anything that writes to an existing file, check what else writes to it.
- Run the experiment that could kill a finding, and accept its outcome in advance. Report the awkward result with
  its number.

## State

At the end of each working session, rewrite `STATE.md` from `templates/STATE.md`: what was run (sweep IDs, selection
and RULE names, archive names), what it printed, what was found, what is next, what is not tested yet, and what a
later session must not assume. Durable facts the person wants kept go to memory only when they ask.

## House style

An editable default: change it in this section to suit the person.

- Plain prose and short sentences. No dashes as punctuation.
- A number with its source rather than a characterisation; tables for more than three comparable values.
- Reports in the first person.
- No preamble and no closing summary.
