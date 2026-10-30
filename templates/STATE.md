# State of {{project}}, {{date_utc}} (UTC)

Read first by every session; rewritten at the end of each one (CLAUDE.md, "State"). Short and exact: names of
sweeps, selections, RULE folders, archives and files, never "the last run". Stays out of git.

## What this is

One paragraph: the project, its question, and where it stands.

## Where everything is

- Repository folder and branch; the code's tree digest at the last node install.
- Data: files under `data/` (and on the node), with their manifests.
- Results: sweeps run (IDs), analyses, selections, RULE folders, rounds (`artifacts/analyses/<round>_<date>/`),
  reports (`reports/`), exports pulled home.

## Done since the last state note

What ran (sweep IDs, job counts, failures), what was read (in the fixed order), what was found (with the file of
each number), and what changed in the code.

## Next

The next step, and the one after. The decision waiting on the person, if any.

## Not tested yet

Code written but not run, paths not run on a GPU, packages not installed where they will be used.

## Do not assume

Facts a later session could get wrong: a sweep still queued, a node session still running, a file that was changed
by hand, a RULE that covers some sweeps but not others.
