# Start here

For a new user of the experiment template: what to prepare, how to get your own copy, how to set it up, and the first
message to give Claude. It takes about half an hour. After that, Claude reads `CLAUDE.md` and takes you through your
project one step at a time.

## What you need

- A GitHub account (free), and git: on Windows, Git for Windows; on a Mac, `xcode-select --install`; on Linux, your
  package manager.
- Python 3.12 or 3.13: on Windows from python.org (tick "Add python.exe to PATH" and keep the `py` launcher); on a Mac
  `brew install python@3.13` or python.org; on Linux your package manager.
- Claude. Either:
  - **Cowork** in the Claude desktop app (a paid plan): Claude works directly in folders you connect, reads
    `CLAUDE.md` itself, and can run the commands on your computer that you allow; or
  - **a Claude Project** in the Claude app (any plan): Claude works in the chat, you run each command and paste the
    output back (the end of this file says how to set it up).
- Your data and any code you already have, in a folder on your computer.
- If you will use a shared GPU node: your own login there (`user@host`), and its rules from whoever runs it. You will
  keep the login in one local file, `node.local.txt`, which git never sees. Never give it to Claude.

## Get your own copy

Each project is its own repository that keeps the template as `upstream`, so fixes to the template reach your
project later with one pull and your own code never conflicts with it.

1. On GitHub, create a new, empty repository for your project (no README, no licence, no .gitignore). Make it
   private if anything about the project is private. Note its address.
2. Clone the template into a folder named after your project and point it at your repository. In **Command Prompt**
   on Windows (change `C:\work`, `my_project` and the two addresses):

   ```
   cd /d C:\work && git clone -c core.autocrlf=false https://github.com/<owner>/experiment-template.git my_project && cd my_project && git remote rename origin upstream && git remote add origin https://github.com/<you>/my_project.git && git push -u origin main
   ```

   In **Terminal** on a Mac or Linux (the same changes):

   ```
   cd ~/work && git clone -c core.autocrlf=false https://github.com/<owner>/experiment-template.git my_project && cd my_project && git remote rename origin upstream && git remote add origin https://github.com/<you>/my_project.git && git push -u origin main
   ```

   `core.autocrlf=false` keeps the line endings the scripts need (`.gitattributes` keeps them too).
3. Later, to take the template's fixes: `git pull upstream main` in the project folder.

## Set up Python

From the project folder, build the main environment with the profiles your work needs. `docs/PROFILES.md` says which
profiles a problem needs; `core torch unsupervised` covers most projects, and Claude will tell you if you need more.

- Windows, in the same Command Prompt (the line starts PowerShell for the setup script; `-TorchIndex cpu`, or
  `cu130` with an NVIDIA GPU). It uses Python 3.12; with only 3.13 installed (`py -0` lists `-V:3.13` and no 3.12),
  add `-Python 3.13` at the end:

  ```
  powershell -ExecutionPolicy Bypass -File env\setup.ps1 -Environment main -Profiles "core,torch,unsupervised" -TorchIndex cpu
  ```

- Mac or Linux, Terminal (on a Mac with an Intel chip, torch and unsupervised do not install, since their
  packages publish no Intel Mac builds of these versions: build `main core` there and run the rest on a GPU node):

  ```
  bash env/setup.sh main core torch unsupervised
  ```

Near the end it prints `exact versions written to ...` and a `next:` line. Then check it (the tests take a few
minutes; every test should pass, and the ones for packages you did not install are skipped; TESTING.md has the whole
check, with the smoke test of every component and the steps on a shared node):

- Windows: `.venv-main\Scripts\python -m unittest discover -s tests -t .`
- Mac or Linux: `.venv-main/bin/python -m unittest discover -s tests -t .`

## Name the project

Edit `project.json`: set `"project"` to a short name (lowercase letters, digits and `_`, starting with a letter).
Folders and sessions on a shared node are named after it. Leave `"discipline": "enforced"` unless you have a reason
(docs/HOW_WE_WORK.md says what it enforces).

## Start with Claude

**With Cowork**: connect two folders, the project folder and the folder that holds your data and code. Then send:

> Read CLAUDE.md in the project folder and follow it. This is a new project. My data and my existing code are in
> <the other folder>. I want to <what you want to find out, in a sentence or two>. I will run <on my laptop / on a
> home GPU / on a shared GPU node>. Interview me first, then look at the data without changing it, and ask me before
> deciding anything I have not told you.

Claude will interview you (the question, the data, how results are chosen and judged, the methods, the compute, your
code), write `PROJECT.md`, `STARTING_PLAN.md` and `STATE.md`, look at your data, reproduce your code's output on a
sample, and then build the first sweep with you. Commands on a shared node are always yours to run: Claude writes
each line and checks what it printed.

**With a Claude Project instead**: create a project in the Claude app, add `CLAUDE.md`, `docs/HOW_WE_WORK.md`,
`docs/CONCEPTS.md` and `docs/COMMANDS.md` to its files, and paste `templates/claude_project_instructions.txt`
(with its blanks filled in) into the project's instructions. Start a chat with the same message as above. Claude
cannot see your computer here, so you run each command and paste the output back, and you attach files Claude
should read. At the end of each session, ask Claude for a state note and add it to the project's files as `STATE.md`
in place of the previous one.

## What stays private

`.gitignore` keeps out of git: `data/`, `artifacts/` (every result), `queue/`, `logs/`, `reports/`, `node.local.txt`,
`STATE.md`, `VERIFICATION_LOG.md`, `PROJECT.md`, `STARTING_PLAN.md` and `findings.csv`. Before every push, `git status`
should list only code and documentation you meant to share.
