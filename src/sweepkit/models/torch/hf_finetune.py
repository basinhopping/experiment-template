"""A pretrained Hugging Face text encoder, fine-tuned as a torch family model on text rows (the pretrained profile:
transformers and tokenizers, and peft for LoRA).

hf_finetune  each row's text is tokenized inside the model (the tokenizer of the same model_id and revision as the
             weights: padded to the longest text of the batch, cut at max_length tokens), the encoder runs, and the
             summary (model.embed) is the first token's output (pooling cls, BERT's [CLS]) or the mean over the real
             tokens (pooling mean, padding left out by the attention mask). The classifier (dropout, an optional hidden
             layer, the output layer) reads it, as in every torch model.

What trains (setting train):
    full   every weight of the encoder and the classifier
    top    the last top_layers entries of the encoder's list of layers (the longest ModuleList named layer, layers, h
           or block: encoder.layer in BERT, RoBERTa, DeBERTa and ELECTRA, transformer.layer in DistilBERT) and the
           classifier; the embeddings, the lower layers, a pooler and any final normalization stay frozen
    lora   low-rank adapters (Hu et al. 2022; peft's LoraConfig with r = lora_r, lora_alpha, lora_dropout,
           target_modules = lora_targets, bias "none", the rest at peft's defaults, which model_record lists) on the
           named linear layers ("query" and "value" in BERT-style encoders), and the classifier; every pretrained
           weight stays frozen
Frozen weights receive no gradient; the optimizer (AdamW of the fit section) skips them, so weight decay never
touches them either. model_params.json records parameters_trainable.

The model is pinned (sweepkit.pretrained): source "hub" names model_id and the 40-character commit (revision), source
"local" a folder under data/ written by save_pretrained. The weights load in float32 (transformers 5 otherwise takes
the dtype in config.json; the fit section's precision bf16 runs the network under autocast); extra passes further
from_pretrained arguments (for example {"attn_implementation": "eager"}). model_params.json gains model_record: the
source, the commit the snapshot was read from, the SHA-256 of the loaded weights (before any adapter), the tokenizer's
class, max_length, pooling, what trained, and the LoRA configuration with its defaults.

The weights are built after the run's "weights" stream seeds torch, so the classifier and the adapters start the same
for a seed; dropout inside the encoder (its config's hidden_dropout_prob and attention_probs_dropout_prob) draws from
the "dropout" stream like every torch model. Encoder models only: the model's AutoModel forward must take the
tokenizer's output alone and return last_hidden_state (BERT, RoBERTa, DeBERTa, DistilBERT, ELECTRA, MPNet, MiniLM,
XLM-R, ModernBERT); models that need code of their own from the Hub (trust_remote_code) are not loaded.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.declare import declare
from sweepkit.models.torch.heads import check_head
from sweepkit.pretrained import check_extra, check_source
from sweepkit.settings import Scalar

LAYER_LISTS = ("layer", "layers", "h", "block")


@dataclasses.dataclass(frozen=True)
class HFFinetuneSettings:
    source: Literal["hub", "local"]
    model_id: str                   # hub: "<owner>/<name>"; local: a folder under data/
    revision: str | None            # hub: the commit's 40 hexadecimal characters; local: None
    max_length: int                 # tokens per text, special tokens included; longer texts are cut
    pooling: Literal["cls", "mean"]
    train: Literal["full", "top", "lora"]
    top_layers: int | None          # train top: how many of the last layers train; None otherwise
    lora_r: int | None              # train lora: the adapters' rank; None otherwise
    lora_alpha: int | None          # train lora: the update is scaled by lora_alpha / lora_r; None otherwise
    lora_dropout: float | None      # train lora: dropout on the adapters' input; None otherwise
    lora_targets: tuple[str, ...]   # train lora: names of the linear layers given adapters; empty otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # the classifier's hidden layer
    dropout: float                  # in the classifier
    classifier_hidden: int | None   # a hidden layer in the classifier, or None
    extra: dict[str, Scalar]        # other from_pretrained arguments of the model


def check_finetune(s: HFFinetuneSettings, input_shape) -> None:
    where = "model hf_finetune"
    check_source(s.source, s.model_id, s.revision, where)
    check_extra(s.extra, where)
    if "dtype" in s.extra or "torch_dtype" in s.extra:
        raise ValueError(f"{where}: the weights load in float32 (fit.precision bf16 gives autocast); extra may not "
                         "set dtype")
    if s.max_length < 2:
        raise ValueError(f"{where}: max_length must be at least 2 (the special tokens count)")
    if (s.train == "top") != (s.top_layers is not None):
        raise ValueError(f"{where}: top_layers goes with train top (at least 1), and is None otherwise")
    if s.top_layers is not None and s.top_layers < 1:
        raise ValueError(f"{where}: top_layers must be at least 1")
    lora = (s.lora_r, s.lora_alpha, s.lora_dropout)
    if s.train == "lora":
        if None in lora or not s.lora_targets:
            raise ValueError(f"{where}: train lora needs lora_r, lora_alpha, lora_dropout and lora_targets")
        if s.lora_r < 1 or s.lora_alpha < 1 or not 0.0 <= s.lora_dropout < 1.0:
            raise ValueError(f"{where}: lora_r and lora_alpha at least 1, lora_dropout in [0, 1)")
    elif any(v is not None for v in lora) or s.lora_targets:
        raise ValueError(f"{where}: the lora settings are None (and lora_targets empty) unless train is lora")
    check_head(s)
    if input_shape is not None and tuple(input_shape) != ():
        raise ValueError(f"{where} reads text (each row a string), not input {input_shape}")


def layer_list(encoder) -> tuple:
    """(name, ModuleList) of the encoder's layers: the longest ModuleList whose own name is layer, layers, h or
    block."""
    from torch import nn

    found = None
    for name, module in encoder.named_modules():
        if isinstance(module, nn.ModuleList) and name.split(".")[-1] in LAYER_LISTS:
            if found is None or len(module) > len(found[1]):
                found = (name, module)
    if found is None:
        raise ValueError(f"train top: {type(encoder).__name__} has no list of layers named {LAYER_LISTS}")
    return found


def set_trainable(encoder, s: HFFinetuneSettings) -> tuple:
    """Freeze what the train setting keeps fixed. Returns (the encoder, wrapped by peft for lora; what trains, for
    model_record)."""
    if s.train == "full":
        return encoder, {"train": "full"}
    if s.train == "top":
        name, layers = layer_list(encoder)
        if s.top_layers > len(layers):
            raise ValueError(f"train top: top_layers {s.top_layers}, but {name} has {len(layers)} layers")
        for p in encoder.parameters():
            p.requires_grad_(False)
        first = len(layers) - s.top_layers
        for k in range(first, len(layers)):
            for p in layers[k].parameters():
                p.requires_grad_(True)
        return encoder, {"train": "top", "layer_list": name, "layers": len(layers),
                         "layers_trained": [f"{name}.{k}" for k in range(first, len(layers))]}
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
                        target_modules=list(s.lora_targets), bias="none")
    wrapped = get_peft_model(encoder, config)
    adapted = sorted({n.rsplit(".lora_", 1)[0] for n, _ in wrapped.named_parameters() if ".lora_" in n})
    if not adapted:
        raise ValueError(f"train lora: no layer of {type(encoder).__name__} is named in lora_targets "
                         f"{list(s.lora_targets)}")
    record = {"train": "lora", "lora_config": _plain(config.to_dict()), "adapted_layers": len(adapted)}
    return wrapped, record


def _plain(value):
    """A library configuration as JSON values (sets as sorted lists, other objects as their text)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _module_class():
    import torch
    from torch import nn

    from sweepkit.models.base import merged_kwargs
    from sweepkit.models.torch.heads import classifier_head
    from sweepkit.pretrained import load_arguments, source_record

    class TextEncoderClassifier(nn.Module):
        def __init__(self, s: HFFinetuneSettings, n_outputs: int):
            super().__init__()
            from transformers import AutoModel, AutoTokenizer

            name, kwargs = load_arguments(s.source, s.model_id, s.revision)
            self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
            encoder = AutoModel.from_pretrained(name, **merged_kwargs(dict(kwargs, dtype=torch.float32),
                                                                       dict(s.extra), "model hf_finetune"))
            width = getattr(encoder.config, "hidden_size", None)
            if type(width) is not int:
                raise ValueError(f"{s.model_id}: its config has no hidden_size; hf_finetune reads encoder models")
            self.source = source_record(s.source, s.model_id, s.revision, encoder)
            self.encoder, self.trained = set_trainable(encoder, s)
            self.max_length, self.pooling = s.max_length, s.pooling
            self.embedding_dim = width
            self.head_dropout = s.dropout
            self.head = classifier_head(s, width, n_outputs)

        def embed(self, x, lengths):
            """x: the batch rows' strings; lengths: None. The summaries [batch, hidden_size]."""
            device = self.head[-1].weight.device
            batch = self.tokenizer(list(x), padding="longest", truncation=True, max_length=self.max_length,
                                   return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = self.encoder(**batch).last_hidden_state
            if self.pooling == "cls":
                return hidden[:, 0]
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

        def model_record(self) -> dict:
            return {**self.source, "tokenizer_class": type(self.tokenizer).__name__, "max_length": self.max_length,
                    "pooling": self.pooling, **self.trained,
                    "encoder_parameters": int(sum(p.numel() for p in self.encoder.parameters())),
                    "encoder_trainable": int(sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))}

    return TextEncoderClassifier


@torch_model("hf_finetune", settings=HFFinetuneSettings, kinds=("text",),
             description="a pretrained Hugging Face text encoder fine-tuned (full, top layers or LoRA), pinned to a "
                         "commit", example="text_finetune", check=check_finetune)
def build_hf_finetune(settings: HFFinetuneSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, n_outputs)


declare("hf_finetune", "pretrained", ("torch", "transformers", "peft"), transductive=False)
