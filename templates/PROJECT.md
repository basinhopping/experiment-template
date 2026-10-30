# {{project}}: the project's brief

Written by Claude from the first interview (CLAUDE.md, "A new project"), read and corrected by the person, and kept
current. It lives in the repository folder and stays out of git unless the person decides nothing in it is private.
Every later session reads it after STATE.md. Each answer says who gave it and when; "not known yet" is an answer.

## The question

What the project wants to learn or build, in two or three sentences, and what result would answer it.

## The data

- Where it is (a folder on this computer, a shared drive, a node), and who may see it. Licences or privacy rules.
- Format and size: files, rows, columns, bytes. What one row is.
- Labels or targets: which columns, how many classes or what range; or none (unsupervised work), and whether
  reference labels exist that may be revealed only after a choice is recorded.
- Groups that must stay together across roles (subjects, patients, sites, batches, sessions).
- Known problems: missing values, duplicates, batches, outliers, units.

## The protocol

- The split: k-fold, group k-fold, leave one group out, a published split, a holdout; or, unsupervised, subsamples to
  choose settings and measure stability, then one run on every row.
- What is held out and never used to choose.
- Seeds: 3 to screen, 5 to report (CLAUDE.md), unless the person decides otherwise.

## How a configuration is chosen and how the choice is judged

- The metric or label-free criterion that chooses, and its direction (larger or smaller is better).
- The metrics read on test files, or the reveal metrics read against reference labels, after the choice.
- What counts as a meaningful difference, and the comparisons planned for the RULE files.

## Models, algorithms and preparation

What the person wants tried, what the field uses, and what is already known to work or fail.

## Compute

- Machines: a laptop CPU, a home GPU, a Mac, a shared node. For a shared node: its usage rules as the person states
  them (hours, GPUs, disk, sessions), and the free disk measured before any data is copied.
- A shared node, measured (the first line of docs/NODE.md, with its date): the home folder's size, limit and free
  space; the CPUs and memory; each GPU's driver, memory and the other users' processes on it; the person's own tmux
  sessions and crontab; Slurm; the Python versions; which package indexes and sites the node reaches.
- A shared node, as the person states it (CLAUDE.md, "A shared node"): the hours and the cutoff, the GPUs this
  project may use, whether a VPN is needed, and the other projects on the same account with the hours, GPUs and free
  disk each needs. The floor this project keeps free, and its GPU window, follow from these.
- The environment profiles needed (docs/PROFILES.md).

## Existing code

Where it is, what it does, how it is run, and the output it gives on a sample, so the template's version can be
checked against it (CLAUDE.md, "Existing code").

## Who reads the results

The people, what they need, the form (a results doc per round, a meeting summary), and any deadlines.

## Decisions

| Date (UTC) | Decision | Reason | Who decided |
|---|---|---|---|
