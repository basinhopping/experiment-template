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

| Document | When | Template | Title, as a Claude Docs document |
|---|---|---|---|
| Results doc | after each round's reading | `templates/results_doc.md` | "<Round> Results on <data>" |
| Meeting summary | before each meeting | `templates/meeting_summary.md` | "Results for the Meeting of <date>" |
| How it works | once the first round has run, kept current | `templates/how_it_works.md` | "How <project> works" |
| Findings list | each finding, when it is made or changes | `templates/findings.csv` | none: `findings.csv` |
| Round README | at the reading | `templates/round_README.txt` (the preparation half) | none: `artifacts/analyses/<round>_<date>/README.txt` |

`reports/`, `findings.csv` and the rest stay out of git (`.gitignore`): results are not code.

- **Results doc**: a short opening paragraph with the main result, then the early findings (each with its number,
  interval and finding ID; the comparison with the previous best first), what I did, "How I found this" (the steps
  and files, enough to repeat it), the results with their charts and tables, the comparison with the previous best,
  what comes next, and the sources.
- **Meeting summary**: only what was asked, each item done fully under its own heading: the answer first, then how,
  with its tables and charts kept beside the text that explains them, and what it cannot show. It gathers the
  rounds' results docs since the last meeting; it ends with the sources.
- **Findings list**: one row per claim: an ID, the claim in one sentence, the runs and tables behind it, the test
  used, the result, a status (supported, not supported, pending), the round, and when it was last checked. A claim in
  a paper cites a finding ID.

## Where the documents live

When the session can make Claude artifacts (Cowork, or a Claude Project with artifacts), each results doc, meeting
summary and the "how it works" page is a **Claude Docs document** (the Docs artifact type), titled as in the table:
one per round, and one per meeting that gathers the rounds since the last. Its text, tables and charts stay editable,
it takes comments, and it exports to Word or PDF. Not a dashboard and not a hand-built HTML page: a page made for one
run cannot be edited or commented as a document can. The Dashboard type has one use here, once the project has many
rounds: an atlas of every sweep, configuration and tested comparison in one place, linked from the meeting summary,
rebuilt before each meeting, and adding no comparison of its own. Without artifacts, the documents are the Markdown
files: `reports/<round>_<date>.md`, `reports/meeting_<date>.md` and `reports/how_it_works.md`.

A document opens with its title, then the date and the author, then the main result in a short paragraph. Each
chart is the document's own chart, made from the rows of the file it shows (not a picture of a chart), with a
caption that names that file, the family and the seeds (`artifacts/analyses/<round>_<date>/report.json · family
F1, seeds 1 to 5`) and a paragraph under it that says what it shows. Long detail that only some readers need (each
architecture layer by layer, every configuration of a search) goes in a second tab. The last section is "Sources":
"Every number comes from these files of the project folder (under artifacts/ unless named otherwise), rounded half
up from the exact values", then the files, section by section.

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
