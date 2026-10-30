# How {{project}} works

A standing page for the people who read the results: what the system does to the data and how a number becomes a
result. Written by Claude once the first round has run, and kept current (each change dated). Plain language first,
the exact settings after. A Claude Docs document when the session can make Claude artifacts (docs/REPORTING.md),
otherwise reports/how_it_works.md.

## The question and the data

What one row is, what the columns are, what is known about each row (labels, groups, batches), and what is not.

## From data to a number

1. Preparation: each step, in order, and why (the settings of the chosen configuration, from its selection record).
   Steps are fitted on the training (or fit) rows only, and applied to the other rows unchanged.
2. The model or algorithm: what it does, in a paragraph, and its settings.
3. The number that chooses: the validation metric, or the label-free criterion, and its direction; how the
   configurations were compared (paired bootstrap, Holm's correction within each family of comparisons).
4. The number that judges: the test metrics, or the agreement with reference labels, read only after the choice was
   recorded, as the RULE fixed it beforehand.

## How to read a result

What a difference and its interval mean; what "no detectable difference" means; why five seeds; what the controls
show; what stability means for a clustering.

## Words used

Fold, seed, configuration, sweep, round, RULE, selection, validation, test, reveal, transductive.
