"""extract-features: a frozen pretrained encoder (the pretrained profile) turns every row of a dataset of text, images
or audio into one vector, written once as a store (data/stores/<name>/, the format convert writes), so any sweep then
reads the features with the store dataset: tabular models, clusterers and embedders on a pretrained model's features.

The dataset is one dataset section in a JSON file (for example {"name": "items", "index": ..., "kind": "text", ...}),
loaded as a sweep would load it; its kind chooses how rows reach the encoder:
    text    the tokenizer of the same model_id and revision as the weights: padded to the longest text of each batch,
            cut at max_length tokens
    image   the model's image processor (resized and normalized as its configuration says); each item is an array,
            channels first, (channels, height, width) or (height, width); one channel is repeated to three. Values are
            given as they are stored: the processor rescales by its own rescale_factor (1/255 for most), so images
            stored in [0, 1] need a processor saved with do_rescale false (a local folder)
    audio   the model's feature extractor at the dataset's sample rate (the items dataset's sample_rate, or the
            synthetic audio generator's), one row per forward pass (batch_size 1), so no padding enters a mean
Pooling over the encoder's last_hidden_state:
    cls     the first position (BERT's [CLS], ViT's class token; models without one have no such summary)
    mean    the mean over the real positions: tokens (text, by the attention mask), patches and the class token
            (ViT-style images), frames (audio), or height and width (a convolutional network's [batch, channels,
            height, width])
    pooler  the model's pooler_output, flattened (refused when the model gives none)

The encoder runs in eval mode without gradients, in float32 or bfloat16 (dtype), on the device named; the values are
stored as float32. The store, written once (a name that exists is refused):
    values.npy     float32 [rows, width], one row per dataset row, in the dataset's order
    row_ids.txt    the dataset's row IDs
    columns.json   {"features": ["e0", "e1", ...], "categorical": {}}
    meta.csv       row_id, then the kept columns as text
    record.json    "to" store; "bytes" and "sha256" of values.npy (the store dataset's fingerprint reads sha256, so a
                   run's provenance names the features themselves); "source": the dataset section and its
                   fingerprint; "encoder": what sweepkit.pretrained.source_record gives (model_id, revision, the commit
                   the snapshot was read from, the weights' SHA-256), the processor's class and configuration, pooling,
                   max_length, dtype and device; rows, features, kept and seconds
"""
import csv
import json
import time

import numpy as np

from sweepkit import paths
from sweepkit.hashing import sha256_file
from sweepkit.pretrained import check_source, load_arguments, source_record
from sweepkit.runs.store import OutputFolder

KINDS = ("text", "image", "audio")


def check_arguments(kind: str, pooling: str, max_length, batch_size: int, dtype: str, device: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"extract-features reads text, image or audio rows; this dataset gives {kind}")
    if pooling not in ("cls", "mean", "pooler"):
        raise ValueError("--pooling: cls, mean or pooler")
    if kind == "text":
        if max_length is None or max_length < 2:
            raise ValueError("--max-length: text needs at least 2 tokens (the special tokens count)")
    elif max_length is not None:
        raise ValueError(f"--max-length: none for {kind} rows (the processor sets their size)")
    if batch_size < 1 or (kind == "audio" and batch_size != 1):
        raise ValueError("--batch-size: at least 1; audio runs one row per forward pass (1)")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("--dtype: float32 or bfloat16")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("--device: cpu, cuda or mps")


def load_section(data_file):
    """(the dataset section as written, the loaded data) from a JSON file holding one dataset section."""
    from sweepkit import catalog
    from sweepkit.registry import DATASETS
    from sweepkit.settings import Selected, to_plain

    catalog.load_all()
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    entry, settings = DATASETS.validate_section(raw, "data")
    return to_plain(Selected(entry.name, settings)), entry.target().load(settings), settings


def sample_rate(section: dict, settings) -> int:
    """The audio rows' sample rate, from the dataset's settings."""
    if section["name"] == "synthetic":
        from sweepkit.data.synthetic import audio_sample_rate

        return audio_sample_rate()
    rate = getattr(settings, "sample_rate", None)
    if type(rate) is not int or rate < 1:
        raise ValueError(f"extract-features: audio needs the dataset's sample rate; {section['name']} gives {rate!r} "
                         "(the items dataset with loader wav sets sample_rate)")
    return rate


def image_array(item) -> np.ndarray:
    """An image item as float32 [3, height, width]."""
    array = np.asarray(item, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise ValueError(f"an image item is (channels, height, width) with 1 or 3 channels, or (height, width); "
                         f"got shape {array.shape}")
    return np.repeat(array, 3, axis=0) if array.shape[0] == 1 else array


def pool(outputs, pooling: str, mask):
    """[batch, width] from the encoder's outputs. mask: [batch, positions] of real positions (text), or None."""
    if pooling == "pooler":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise ValueError("--pooling pooler: this model gives no pooler_output; use cls or mean")
        return pooled.reshape(pooled.shape[0], -1)
    hidden = outputs.last_hidden_state
    if hidden.ndim == 4:
        if pooling == "cls":
            raise ValueError("--pooling cls: this model's output is a feature map [batch, channels, height, width], "
                             "without a first token; use mean or pooler")
        return hidden.mean(dim=(2, 3))
    if pooling == "cls":
        return hidden[:, 0]
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _device(device: str):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda: torch sees no CUDA GPU here")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise ValueError("--device mps: torch's MPS backend is not available here")
    return torch.device(device)


class Encoder:
    """The frozen model and what prepares its input, for one kind of row."""

    def __init__(self, kind: str, source: str, model_id: str, revision, pooling: str, max_length, dtype: str,
                 device: str, rate):
        import torch
        import transformers

        name, kwargs = load_arguments(source, model_id, revision)
        self.kind, self.pooling, self.max_length, self.rate = kind, pooling, max_length, rate
        self.device = _device(device)
        if kind == "text":
            self.processor = transformers.AutoTokenizer.from_pretrained(name, **kwargs)
        elif kind == "image":
            self.processor = transformers.AutoImageProcessor.from_pretrained(name, **kwargs)
        else:
            self.processor = transformers.AutoFeatureExtractor.from_pretrained(name, **kwargs)
        weights = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        model = transformers.AutoModel.from_pretrained(name, dtype=weights, **kwargs)
        self.source = source_record(source, model_id, revision, model)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def processor_record(self) -> dict:
        record = {"class": type(self.processor).__name__}
        to_dict = getattr(self.processor, "to_dict", None)
        if self.kind != "text" and callable(to_dict):
            record["configuration"] = json.loads(json.dumps(to_dict(), default=str))
        return record

    def __call__(self, items: list) -> np.ndarray:
        """float32 [len(items), width]."""
        import torch

        mask = None
        if self.kind == "text":
            batch = self.processor(list(items), padding="longest", truncation=True, max_length=self.max_length,
                                   return_tensors="pt")
            mask = batch.get("attention_mask")
            if mask is None:
                raise ValueError("the tokenizer gives no attention_mask, so padding cannot be told from the text")
        elif self.kind == "image":
            batch = self.processor(images=[image_array(item) for item in items], input_data_format="channels_first",
                                   return_tensors="pt")
        else:
            samples = np.asarray(items[0], dtype=np.float32).reshape(-1)
            batch = self.processor(samples, sampling_rate=self.rate, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self.model.dtype != torch.float32:
            batch = {k: v.to(self.model.dtype) if torch.is_floating_point(v) else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            pooled = pool(outputs, self.pooling, mask.to(self.device) if mask is not None else None)
        return pooled.float().cpu().numpy()


def extract_features(data_file, source: str, model_id: str, revision, pooling: str, max_length, batch_size: int,
                     dtype: str, device: str, keep: tuple, name: str, log) -> dict:
    started = time.perf_counter()
    check_source(source, model_id, revision, "extract-features")
    section, data, settings = load_section(data_file)
    check_arguments(data.kind, pooling, max_length, batch_size, dtype, device)
    for column in keep:
        if not data.table.has(column):
            raise ValueError(f"--keep {column}: the dataset has no such column ({sorted(data.table.columns)})")
    rate = sample_rate(section, settings) if data.kind == "audio" else None
    encoder = Encoder(data.kind, source, model_id, revision, pooling, max_length, dtype, device, rate)
    n = len(data.table)
    log(f"extract-features: {n} {data.kind} rows through {model_id} ({encoder.source['model_class']}, pooling "
        f"{pooling}) on {device}")
    folder = OutputFolder(paths.data_dir() / "stores", name).begin()
    try:
        values = None
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            block = encoder([data.items.get(i) for i in range(start, stop)])
            if not np.isfinite(block).all():
                raise FloatingPointError(f"rows {start} to {stop}: the encoder gave values that are not finite")
            if values is None:
                values = np.lib.format.open_memmap(folder.path("values.npy"), mode="w+", dtype=np.float32,
                                                   shape=(n, int(block.shape[1])))
            values[start:stop] = block
            if stop == n or (start // batch_size) % 50 == 0:
                log(f"  {stop} of {n} rows")
        if values is None:
            raise ValueError("the dataset has no rows")
        width = int(values.shape[1])
        values.flush()
        del values
        folder.write_text("row_ids.txt", "".join(f"{rid}\n" for rid in data.table.ids.tolist()))
        folder.write_json("columns.json", {"features": [f"e{j}" for j in range(width)], "categorical": {}})
        with open(folder.path("meta.csv"), "x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_id"] + list(keep))
            kept = [data.table.column(c).astype(str) for c in keep]
            for i, rid in enumerate(data.table.ids.tolist()):
                writer.writerow([rid] + [k[i] for k in kept])
        n_bytes, digest = sha256_file(folder.path("values.npy"))
        record = {"to": "store", "bytes": n_bytes, "sha256": digest, "rows": n, "features": width,
                  "kept": list(keep), "row_id": None,
                  "source": {"data": section, "fingerprint": data.fingerprint},
                  "encoder": {**encoder.source, "kind": data.kind, "pooling": pooling, "max_length": max_length,
                              "dtype": dtype, "device": device, "processor": encoder.processor_record(),
                              "sample_rate": rate},
                  "seconds": round(time.perf_counter() - started, 1)}
        folder.write_json("record.json", json.loads(json.dumps(record, default=str)))
    except BaseException as err:
        folder.fail(err)
        raise
    done = folder.complete()
    log(f"written {done}")
    return record
