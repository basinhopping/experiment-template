Round {{round}}, prepared {{prepared_utc}} (UTC) for project {{project}}.

Question: {{question}}

Sweeps (each checked by expand; files under sweeps/, SHA-256 in sweep_files.sha256):
{{sweeps}}

On the node: the environment {{environment}} (~/venvs/{{project}}-{{environment}}, profiles {{profiles}}); the schedule
starts {{workers_per_gpu}} workers on each of GPUs {{gpus}} (only those no one else is using) and stops them
{{end_phrase}},
or earlier when no job is left (the round's session then ends) or less than {{min_free_gb}} GB is free in the home
folder.

Plan: {{plan}}
  the steps node/chain.sh runs on the node in one tmux session that ends by itself. It lives under sweeps/ so that
  it travels to the node with the code; the tree digest below covers it, and its SHA-256 is in sweep_files.sha256.

Cost: {{cost}}

Code: tree digest {{digest}} (node/tree_digest.sh on the node prints the same number for the same code).

Files of this folder:
  RULE_{{round}}.txt: what runs and how it is read, fixed with rule-fix before any result exists. Fixed at:
      {{rule_fixed}} (write rule-fix's time and SHA-256 here)
  sweep_files.sha256: the SHA-256 of each sweep file and of the plan at preparation.
  lines/: one file per step, numbered in the order they run: {{lines}}.

Changes after preparation (each with the time, why, and whether it changes a job of the RULE):
(none yet)
