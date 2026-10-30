"""Which parts of a configuration are transductive: they use the inputs (never the labels) of rows outside the rows a
model is fitted on, such as normalization statistics per subject taken over every row of that subject, validation and
test rows included. A transductive result is not the result of a model that saw only its training rows, so every
report that ranks or judges such a configuration says so: analyze, select, test-report, reveal-report and export.

A component declares it in one of two ways:
    the registry meta transductive=True        always transductive (a preparation step)
    a hook is_transductive(section) -> bool    transductive for some settings only; section is the plain settings
                                               dict of its job section, such as a feature's
                                               {"name": "sequence", ..., "normalization": "group", ...}
"""
from sweepkit.registry import FEATURES, FITS, MODELS, PREP


def _declares(registry, section: dict) -> bool:
    entry = registry.get(section["name"])
    hook = getattr(entry.target, "is_transductive", None)
    return entry.meta.get("transductive") is True or (callable(hook) and bool(hook(section)))


def parts(settings: dict) -> list:
    """The transductive parts of a configuration's settings, as text ("prep <name>", "features <name>", ...); empty
    when the configuration fits on its fit rows alone."""
    found = [f"prep {step['name']}" for step in settings["prep"] if _declares(PREP, step)]
    for section, registry in (("features", FEATURES), ("model", MODELS), ("fit", FITS)):
        if _declares(registry, settings[section]):
            found.append(f"{section} {settings[section]['name']}")
    return found


def note(found: list) -> str:
    return ("transductive (" + ", ".join(found) + "): it used the inputs, never the labels, of rows outside its "
            "fit rows")
