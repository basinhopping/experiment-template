# Reporting

How results become documents other people read, so that every number in them can be traced to the file it came from
and nobody has to take a claim on trust. docs/HOW_WE_WORK.md has the rules behind it; `templates/` has the documents.

## Numbers

- Every number in a report is read from a file an analysis wrote: the reports of `analyze`, `test-report`,
  `reveal-report` and `test-compare` (each a `report.json` beside a `report.txt`), the tables of an `export`, or a
  file the project's own analysis code wrote. One writer per table or figure: the code that makes a table writes the
  file the table is read from.
- A number that is in no file does not go in. A number is never typed by hand from a screen.
- Values are rounded once, half up, from the stored exact value (`sweepkit.eval.stats.round_half_up`): 0.6005 shows as
  0.601, never 0.600, and a rounded value is never rounded again.
- Every count and value carries its unit or term: runs, configurations, seeds, folds, comparisons, families;
  validation files or test files; balanced or plain accuracy; a fraction or percentage points.
- An interval says its level and method (95% paired bootstrap over seeds and rows, 2,000 draws). A p-value says
  whether it is Holm-adjusted and within which family.

## Verdict words

A comparison gets exactly one of: **better**, **worse**, **no detectable difference** (the Holm-adjusted p at or
above the threshold, whatever the sign of the difference). Never "slightly better", "a trend", or "nearly
significant". A claim is limited to what was tested: the data, the split, the seeds, the configurations.

Each reported comparison also says when it was fixed: in a RULE before the results (planned), after the judging files
were read (post hoc), exploratory, or descriptive. Only planned comparisons carry a p-value in a summary.

## Documents

| Document | When | Template | Where it lives |
|---|---|---|---|
| Results doc | after each round's reading | `templates/results_doc.md` | `reports/<round>_<date>.md`, and the Claude Project |
| Meeting summary | before each meeting | `templates/meeting_summary.md` | `reports/meeting_<date>.md`, and the Claude Project |
| How it works | once the first round has run, kept current | `templates/how_it_works.md` | `reports/how_it_works.md` |
| Findings list | each finding, when it is made or changes | `templates/findings.csv` | `findings.csv` |
| Round README | at the reading | `templates/round_README.txt` (the preparation half) | `artifacts/analyses/<round>_<date>/README.txt` |

`reports/`, `findings.csv` and the rest stay out of git (`.gitignore`): results are not code.

- **Results doc**: the findings first (each with its number, interval and finding ID), then what was done, "How I
  found this" for each result (the steps and files, enough to repeat it), the detail, what it does not show, and the
  files. The comparison with the previous best comes first among the findings.
- **Meeting summary**: only what was asked, each item done fully under its own large header, with its tables and
  figures kept beside the text that explains them, then the questions for the meeting.
- **Findings list**: one row per claim: an ID, the claim in one sentence, the runs and tables behind it, the test
  used, the result, a status (supported, not supported, pending), the round, and when it was last checked. A claim in
  a paper cites a finding ID.

## Figures

A figure is drawn from data files identified by hash, by a script that refuses inputs that disagree (different
seeds, different folds), and is looked at as it renders before it goes into a document. The export command's plots
(`export --plots yes`) and `cluster-report` follow this.

## Before anyone else reads a document

1. Nothing in it that its readers should not see: no logins, host names, personal details, private data or raw rows.
2. Every number checked against its file by a separate read: for important documents, a separate Claude session or
   agent that did not write it recomputes each number and claim from the files, read only.
3. Every "not tested" item still open is said.

A two-dimensional UMAP or t-SNE picture is for looking, not for measuring: distances and cluster sizes in it are not
the data's (docs/UNSUPERVISED.md).
