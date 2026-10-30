"""project.json: the project's name and its machine-independent choices, edited once when a project adopts the
template.

    {"project": "cells", "discipline": "enforced"}

project     lowercase letters, digits and _, starting with a letter, at most 24 characters. It names the node
            folders (~/<project>), environments (~/venvs/<project>-<environment>) and tmux sessions
            (<project>_gpu0 ...), so two projects of one person never collide.
discipline  "enforced": test-report and reveal-report refuse without a RULE file fixed before the selection.
            "advisory": they run anyway and print a warning at the top of the report.
"""
import json
import re

from sweepkit import paths

KEYS = ("project", "discipline")


def load() -> dict:
    path = paths.repo_root() / "project.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; copy the template's project.json and set the project name")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    check(data)
    return data


def check(data) -> None:
    if type(data) is not dict or set(data) != set(KEYS):
        raise ValueError(f"project.json must hold exactly {list(KEYS)}, got {sorted(data) if type(data) is dict else data}")
    name = data["project"]
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", name):
        raise ValueError(f"project.json project {name!r}: lowercase letters, digits and _, starting with a letter, "
                         "at most 24 characters")
    if data["discipline"] not in ("enforced", "advisory"):
        raise ValueError("project.json discipline must be enforced or advisory")


def name() -> str:
    return load()["project"]


def discipline() -> str:
    return load()["discipline"]
