"""Pretrained models from the Hugging Face Hub or from a local folder, pinned and fingerprinted (the pretrained
profile: transformers, tokenizers, huggingface_hub, and peft for LoRA). Used by the fine-tuned text model
(sweepkit.models.torch.hf_finetune) and by frozen-encoder feature extraction (sweepkit.data.extract).

A pretrained model is named by three settings:
    source     "hub": model_id is a Hub repository ("<owner>/<name>") and revision the commit hash (40 hexadecimal
               characters, from the model page's "Files and versions" or huggingface_hub's model_info) the files are
               pinned to, so a later change to the repository changes nothing here;
               "local": model_id is a folder under data/ (a model saved with save_pretrained, or a Hub snapshot
               copied there for a machine without internet access) and revision is None
    model_id, revision
Downloads go to HF_HOME (node/common.sh sets ~/cache/huggingface on a node). load_arguments gives the arguments every
from_pretrained call takes, so the tokenizer, the processor and the weights always come from the same files. Models
that need code of their own from the Hub (trust_remote_code) are not loaded.

source_record is what a run (model_params.json, model_record) or a store (record.json) keeps about the model: source,
model_id and revision; commit_hash, the Hub commit the files were read from (the revision itself, which check_source
requires to be a full commit and every from_pretrained call loads at; transformers 5 no longer keeps the commit on
the config; None for a local folder); weights_sha256, every tensor of the loaded model's state (name, shape, dtype and
bytes, in name order), so the weights a run started from are known however they were stored; folder_sha256 for a
local folder (every file); and the transformers version.
"""
import hashlib
import re

_COMMIT = re.compile(r"[0-9a-f]{40}")


def check_source(source: str, model_id: str, revision, where: str) -> None:
    """The three settings' own checks (no download, no file read)."""
    from sweepkit import paths

    if source not in ("hub", "local"):
        raise ValueError(f"{where}: source is hub or local")
    if type(model_id) is not str or not model_id.strip():
        raise ValueError(f"{where}: model_id names the model")
    if source == "hub":
        if type(revision) is not str or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{where}: a hub model is pinned to a commit: revision is its 40 hexadecimal characters "
                             f"(the model page's Files and versions), not {revision!r}")
        if model_id.count("/") != 1:
            raise ValueError(f"{where}: a hub model_id is <owner>/<name>, not {model_id!r}")
    else:
        if revision is not None:
            raise ValueError(f"{where}: a local model takes revision None (its files are fingerprinted instead)")
        paths.data_path(model_id)


def check_extra(extra: dict, where: str) -> None:
    for key in ("revision", "local_files_only", "trust_remote_code", "cache_dir"):
        if key in extra:
            raise ValueError(f"{where}: extra may not set {key} (the source settings decide it; models that need "
                             "their own code from the Hub are not loaded)")


def load_arguments(source: str, model_id: str, revision) -> tuple:
    """(name or folder, keyword arguments) for every from_pretrained call of one model."""
    if source == "hub":
        return model_id, {"revision": revision}
    from sweepkit import paths

    folder = paths.data_path(model_id)
    if not folder.is_dir():
        raise FileNotFoundError(f"no folder {folder} (a local model is a folder under data/)")
    return str(folder), {"local_files_only": True}


def weights_sha256(module) -> str:
    """SHA-256 over a module's state: each tensor's name, shape and dtype, then its bytes, in name order."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous().reshape(-1)
        digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def folder_sha256(folder) -> str:
    """SHA-256 over every file of a folder: relative path and bytes, in path order."""
    from pathlib import Path

    from sweepkit.hashing import sha256_file

    root = Path(folder)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        _, file_hash = sha256_file(path)
        digest.update(f"{path.relative_to(root).as_posix()}|{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def source_record(source: str, model_id: str, revision, model) -> dict:
    """What a run or a store records about the pretrained model it used."""
    record = {"source": source, "model_id": model_id, "revision": revision,
              "commit_hash": revision if source == "hub" else None,
              "model_class": type(model).__name__, "weights_sha256": weights_sha256(model)}
    if source == "local":
        from sweepkit import paths

        record["folder_sha256"] = folder_sha256(paths.data_path(model_id))
    try:
        import transformers

        record["transformers"] = transformers.__version__
    except ImportError:
        pass
    return record
