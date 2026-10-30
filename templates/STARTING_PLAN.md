# {{project}}: a starting plan

Written by Claude after the interview and the data audit, agreed with the person. Each step ends with something that
is checked, and says who does it. The plan changes when a result changes it; each change is dated.

## Step 1: set up

- The repository: cloned, `project.json` names the project, `upstream` points at the template (START_HERE.md).
- Environments: the profiles the project needs (docs/PROFILES.md). Done when `env-check` passes for each and the
  test suite passes (TESTING.md).
- Done when: `python -m unittest discover -s tests -t .` passes and `smoke-all --profile core` prints SMOKE-ALL OK.

## Step 2: the data, looked at before any model runs

- `ingest` (a verified copy with a manifest) or the file in place under `data/`; `profile-data` on it.
- Done when: VERIFICATION_LOG.md entry 1 records the files, sizes, rows, columns, types, missing values and the join
  between labels and rows, each with the file it was read from.

## Step 3: existing code reproduced

- The person's script, unchanged, on a sample; the template's version of the same method on the same sample.
- Done when: the two outputs agree (for a deterministic clustering, adjusted Rand index 1.0, or the difference shown to
  be float32 against float64 as CLAUDE.md says; for a model, the same predictions or metrics within the stated
  tolerance), recorded in VERIFICATION_LOG.md. Skip when there is no code.

## Step 4: the first sweep

- The base configuration is the person's current settings (or a standard starting point), on a sample or subsamples.
- Done when: `check` expands it, `smoke` passes, it runs (`run`, or the node), and `analyze` reads it.

## Step 5: the first round

- A question, a RULE fixed before results (`round-new`, `rule-fix`, `round-check`), the runs, the reading in the fixed
  order, the records and a results doc (docs/HOW_WE_WORK.md).
- Done when: the round's README, numbers and results doc exist, and every number in the doc traces to a file.

## Later steps

Screens on three seeds, confirmations on five new seeds, the reading the RULE fixed; each as its own round.
