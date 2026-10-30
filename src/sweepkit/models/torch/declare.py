"""Adjust a torch model's registry entry after sweepkit.models.torch.base.torch_model registered it: the environment
profile and import names it needs (torch_model declares the torch profile and torch alone), and whether it is
transductive. A transductive model reads the inputs of rows outside its training rows (never their labels), as a
graph network reads every node's features; its registry target then has is_transductive, which
sweepkit.analysis.transductive reads so that every report on such a configuration says so.
"""
from sweepkit.registry import MODELS


def _always(section: dict) -> bool:
    return True


def declare(name: str, profile: str, packages: tuple, transductive: bool) -> None:
    entry = MODELS.get(name)
    if entry.meta["family"] != "torch":
        raise ValueError(f"{name} is not a torch model")
    if type(profile) is not str or type(packages) is not tuple or not all(type(p) is str for p in packages):
        raise TypeError("declare: profile is text and packages a tuple of import names")
    entry.meta["profile"] = profile
    entry.meta["packages"] = packages
    if transductive:
        entry.target.is_transductive = staticmethod(_always)
