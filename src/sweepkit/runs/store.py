"""The life of one output folder.

A job writes into <parent>/<id>.tmp/ and renames it to <parent>/<id>/ only when complete. A failed job becomes
<parent>/<id>.failed-<utc>/ with the error in error.txt. A complete folder is never touched again: starting a job
whose folder is complete raises AlreadyComplete, so a re-queued finished job is skipped rather than overwritten.
Every file is opened with mode "x", so nothing inside a folder is ever overwritten either.
"""
import datetime
import json
import os
import traceback
from pathlib import Path


class AlreadyComplete(Exception):
    pass


def stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


class OutputFolder:
    def __init__(self, parent: Path, folder_id: str):
        if not folder_id or "/" in folder_id or "\\" in folder_id or folder_id.startswith("."):
            raise ValueError(f"bad folder id {folder_id!r}")
        self.parent = Path(parent)
        self.folder_id = folder_id
        self.final = self.parent / folder_id
        self.tmp = self.parent / f"{folder_id}.tmp"
        self.open = False

    def begin(self) -> "OutputFolder":
        if self.final.exists():
            raise AlreadyComplete(str(self.final))
        self.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp.exists():
            # Left by a job that died without cleaning up. Keep it, renamed, as evidence.
            os.rename(self.tmp, self.parent / f"{self.folder_id}.failed-{stamp()}-abandoned")
        self.tmp.mkdir()
        self.open = True
        return self

    def path(self, name: str) -> Path:
        if not self.open:
            raise RuntimeError("folder is not open")
        target = (self.tmp / name).resolve()
        if self.tmp.resolve() not in target.parents:
            raise ValueError(f"{name!r} is outside the folder")
        return target

    def write_json(self, name: str, obj) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=1, allow_nan=False)
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return target

    def write_bytes(self, name: str, data: bytes) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as handle:
            handle.write(data)
        return target

    def write_npy(self, name: str, array) -> Path:
        import numpy as np

        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as handle:
            np.save(handle, array, allow_pickle=False)
        return target

    def complete(self) -> Path:
        if not self.open:
            raise RuntimeError("folder is not open")
        os.rename(self.tmp, self.final)
        self.open = False
        return self.final

    def fail(self, error: BaseException) -> Path:
        if not self.open:
            raise RuntimeError("folder is not open")
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        with open(self.tmp / "error.txt", "w", encoding="utf-8") as handle:
            handle.write(text)
        failed = self.parent / f"{self.folder_id}.failed-{stamp()}"
        os.rename(self.tmp, failed)
        self.open = False
        return failed


def write_folder(parent: Path, folder_id: str, files: dict) -> Path:
    """Write a whole output folder at once: {name: content}. .json names get JSON, .npy names numpy arrays,
    bytes values raw bytes, everything else text. Fails the folder on any error."""
    folder = OutputFolder(parent, folder_id).begin()
    try:
        for name, content in files.items():
            if name.endswith(".json"):
                folder.write_json(name, content)
            elif name.endswith(".npy"):
                folder.write_npy(name, content)
            elif isinstance(content, bytes):
                folder.write_bytes(name, content)
            else:
                folder.write_text(name, content)
    except BaseException as err:
        folder.fail(err)
        raise
    return folder.complete()
