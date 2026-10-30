# {{Round}} Results on {{data}}

{{date}} · {{author}}

<!-- A results doc, one per round, written by Claude from the round's files after the reading (docs/REPORTING.md).
When the session can make Claude artifacts it is a Claude Docs document (the Docs artifact type) with this title, and
these headings as its sections; otherwise this file as reports/{{round}}_{{date}}.md. Checked before anyone else
reads it (CLAUDE.md, "Reports"): every number here comes from a file named in it, rounded half up once from the
stored value. Remove this comment. -->

What the round did and its main result, in two or three sentences with the numbers: "Round 3 screened ... on ...;
the best of them, ..., beat the previous best by ... [interval]" or "no configuration was detectably better".

## Early findings

- **The claim in a few words.** Its number with its interval and verdict word (better, worse, no detectable
  difference), what it was compared with, and its finding ID (findings.csv). The comparison with the previous best
  comes first.
- **The next claim.** The same form: one bullet per finding, the awkward results included.

## What I did

Who asked for what, and when (the meeting, in the asker's words), and what ran. A table when the round has arms that
differ in more than one way (what each trains or clusters on, and what it is judged on):

| Arm | Fit on | Judged on |
|---|---|---|
| ... | ... | ... |

Then one paragraph: the sweeps by ID, the configurations, seeds, folds or subsamples, the number of runs, the device
and GPUs, the run time (from the runs' efficiency.json files) and the code's tree digest.

## How I found this

- **The rule came first.** The RULE's name, when rule-fix fixed it (before any run) and its SHA-256; what it set.
- **Validation (or label-free) files first.** Which files came into the analysis, each matched to the node's SHA-256
  by pull; what analyze ranked and by which criterion.
- **The record before the judging files.** select's record, with its reason as the RULE said it.
- **Test (or reveal) files last.** Only the selected configuration's, with test-report or reveal-report, and the
  families test-compare ran.
- **The method.** The comparison (paired by seed, fold and row), the bootstrap and its draws, Holm within each
  family; and any check of the method on a case with a known answer.

## Results

The answer in a sentence or two, then each chart: the document's own chart, made from the rows of the file it shows,
with a caption that names the file, the family and the seeds (artifacts/analyses/{{round}}_{{date}}/report.json ·
family F1, seeds 1 to 5), and a paragraph under it saying what it shows. Then the tables: every configuration against
the baseline (difference, interval, P(better), Holm-adjusted p, verdict), the controls (expect chance), stability
with its parts and the number of groups found where they apply, and per-class or per-group results.

## Against the previous best

The previous best by name, with its number and its document; this round's result against it, as tested; and what
this round does not test (from the RULE's section 4, and anything found since).

## Next

1. What runs next and when, on which GPUs, and what is read then.
2. What is still open from the meeting.

## Sources

Every number comes from these files of the project folder (under artifacts/ unless named otherwise), rounded half up
from the exact values.

- analyses/{{round}}_prep_{{date}}: the plan, RULE_{{round}}.txt (SHA-256 ...) and README.txt
- analysis/<sweep>_<utc>: analyze's reports, one per sweep
- selections/<name>: the selection records
- test_reports/ or reveal_reports/<selection>_<utc>, and the comparisons of test-compare
- exports/<selection>_<utc>: the rows and plots taken home
- VERIFICATION_LOG.md, the entry of this round
