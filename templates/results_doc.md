# {{project}}, round {{round}}: results

{{date_utc}} (UTC). Written by Claude from the round's files and checked before anyone else reads it (CLAUDE.md,
"Reports"): every number here comes from a file named beside it, rounded half up once from the stored value. Kept in
`reports/{{round}}_{{date}}.md`, and in the Claude Project when the session has one.

## Findings

Each in one or two sentences, with its finding ID (findings.csv), its number and interval, and its status
(supported, not supported, pending). The comparison with the previous best comes first: "this round's y beat last
round's x by z (interval, Holm-adjusted p)", or "did not detectably differ".

## What was done

The question, the sweeps (IDs, configurations, seeds, folds), the RULE (folder, fixed time, SHA-256), the device, the
code's tree digest, and the round's run time (from the runs' efficiency.json files).

## How I found this

For each finding, the steps that produced it, in the order they happened: which files were read when (validation or
label-free files with analyze, the selection record, then the test or reveal files), the commands with their
arguments, and the output file of each number. Enough that someone else could repeat it and get the same numbers.

## Detail

The tables: every configuration compared with the baseline (difference, interval, P(better), Holm-adjusted p,
verdict), the controls (expect chance), stability where it applies, per-class or per-group results, figures with
their source files.

## What this does not show

From the RULE's section 4, and anything found since: optimistic choices, one dataset or split, transductive steps,
what was not run.

## Files

The round's folders (`artifacts/analyses/{{round}}_prep_{{date}}/`, `artifacts/analyses/{{round}}_{{date}}/`), the
numbers file, and the exports.
