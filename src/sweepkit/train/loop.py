"""The training loop of the torch family: one training step, the epochs with validation and early stopping, and
prediction on whole rows or on averaged windows. sweepkit.families.torch builds what a job needs and calls these; the
smoke test and estimate run the same code through the real runner.

One training step (Step.loss), in this order:
    the batch       the rows picked for the step (sweepkit.train.batches), sequences and audio cropped at random starts
                    (stream "batches"), audio played at a random speed first when the job asks for it ("augment speed")
    the input       the feature (sweepkit.features: log-mel of audio, or the sequence or image as given) and the
                    normalization, on the job's device
    augmentations   masks ("augment masks"), noise ("augment noise"), then mixup ("augment mixup"), which mixes every
                    target the loss reads with the same weights (sweepkit.train.augment)
    the network     under bfloat16 autocast when precision is bf16: model(x, lengths), or model.embed and model.classify
                    when an extra head reads the summary
    the loss        the task's loss (sweepkit.train.losses): soft cross entropy for classification, binary cross entropy
                    for multi-label, mse, l1 or ccc on standardized targets for regression; plus the model's activity
                    penalty (spiking models), the auxiliary head's loss times its weight, and the supervised contrastive
                    loss times its weight

Each random stream is named (sweepkit.streams) and used by one thing only, so an option a configuration does not use
never changes the draws of the others. The network's weights and dropout come from torch's own generator, seeded by
sweepkit.families.torch before the model, before each extra head, and once more before training.

An epoch: a fresh order of the training rows (a permutation, or draws with replacement under the balanced sampler),
batch_size rows per step (the last incomplete batch is dropped; with fewer rows than batch_size, one batch of all of
them), at most max_batches_per_epoch steps (smoke tests and estimates), AdamW with the learning rate warmed up
linearly and then held or decayed on a cosine to zero at the last epoch of the schedule. After each epoch the
validation rows are predicted (eval mode, no augmentation) and scored with every metric of the task; the epoch with
the best value of the task's select metric is kept (strictly better only, so ties keep the earlier epoch), and
training stops early after early_stopping_patience epochs without a better one. With eval_train, the training rows are
predicted and scored the same way after each epoch (true labels, no augmentation); this draws no random number and
changes no weight.
"""
import dataclasses
import math
import time

import numpy as np

from sweepkit.streams import generator, int_seed
from sweepkit.train import augment as augmentations
from sweepkit.train.batches import eval_batches, train_order, training_batch, window_batches
from sweepkit.train.losses import multilabel_loss, regression_loss, soft_cross_entropy, supervised_contrastive_loss


def lr_factor(step: int, warmup_steps: int, total_steps: int, schedule: str) -> float:
    """The learning rate of a step as a share of lr: linear warmup from lr / warmup_steps to lr, then constant, or a
    cosine from lr to 0 at total_steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def to_device(x, device):
    """A numpy batch as a tensor on the device. Anything else (a list of strings, for a model that tokenizes inside its
    embed) is passed on unchanged."""
    import torch

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device, non_blocking=True)
    return x


class Streams:
    """The random streams of one training run, each used by one thing only."""

    def __init__(self, seed: int):
        import torch

        self.batches = generator(seed, "batches")                 # epoch order and crops
        self.speed = generator(seed, "augment speed")
        self.mixup = generator(seed, "augment mixup")
        self.masks = torch.Generator().manual_seed(int_seed(seed, "augment masks"))
        self.noise = torch.Generator().manual_seed(int_seed(seed, "augment noise"))


@dataclasses.dataclass
class Targets:
    """What the loss reads for the rows of one source, in the source's order, as training sees them (shuffled under
    the control, smoothed, standardized)."""

    main: np.ndarray                # float32 [rows, outputs]
    labels: np.ndarray | None       # int64 [rows] class indices (classification): the balanced sampler and the
                                    # contrastive loss read them; None for the other tasks
    aux: np.ndarray | None          # float32 [rows, auxiliary outputs] (one-hot, or standardized values), or None


@dataclasses.dataclass
class Losses:
    """The loss settings of one training run."""

    task: str                       # classification, multilabel or regression
    class_weights: object           # classification: a [classes] tensor (weighted_loss), or None
    pos_weight: object              # multi-label: a [labels] tensor (weighted_loss), or None
    regression: str                 # regression: mse, l1 or ccc; "none" for the other tasks
    aux_task: str | None            # classification or regression, or None without an auxiliary head
    aux_loss: str | None            # cross_entropy, mse or ccc
    aux_weight: float
    contrastive_weight: float
    contrastive_temperature: float | None


@dataclasses.dataclass
class Predictions:
    """A role's predictions, one row per source row, in the source's order. With windows, each row's values are the
    mean over its windows."""

    main: np.ndarray                # float64 [rows, outputs]: class probabilities (classification), label probabilities
                                    # (multi-label) or standardized values (regression)
    aux: np.ndarray | None          # float64 [rows, auxiliary outputs]: class probabilities or standardized values
    seconds: float


def output_values(task: str, outputs):
    """What is averaged over windows and scored: softmax probabilities, sigmoid probabilities, or the values."""
    import torch

    if task == "classification":
        return torch.softmax(outputs.float(), dim=-1)
    if task == "multilabel":
        return torch.sigmoid(outputs.float())
    return outputs.float()


class Network:
    """A job's model with what feeds it and reads it: the feature, the normalizer, the extra heads (auxiliary and
    projection, each an nn.Module or None), the device, the precision and the task."""

    def __init__(self, model, aux_head, contrastive_head, feature, normalizer, device, precision: str, task: str,
                 aux_task):
        self.model, self.aux_head, self.contrastive_head = model, aux_head, contrastive_head
        self.feature, self.normalizer = feature, normalizer
        self.device, self.precision, self.task, self.aux_task = device, precision, task, aux_task

    def heads(self) -> list:
        return [h for h in (self.aux_head, self.contrastive_head) if h is not None]

    def parameters(self) -> list:
        """The model's parameters, then the auxiliary head's, then the projection head's, so a job without extra
        heads gives the optimizer the list it always did."""
        out = list(self.model.parameters())
        for head in self.heads():
            out += list(head.parameters())
        return out

    def train(self) -> None:
        self.model.train()
        for head in self.heads():
            head.train()

    def eval(self) -> None:
        self.model.eval()
        for head in self.heads():
            head.eval()

    def state(self) -> dict:
        """A copy of every weight and statistic (batch norm running means included)."""
        def copy(module):
            return None if module is None else {k: v.detach().clone() for k, v in module.state_dict().items()}

        return {"model": copy(self.model), "aux": copy(self.aux_head), "contrastive": copy(self.contrastive_head)}

    def load(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        if self.aux_head is not None:
            self.aux_head.load_state_dict(state["aux"])
        if self.contrastive_head is not None:
            self.contrastive_head.load_state_dict(state["contrastive"])

    def autocast(self):
        import torch

        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16")

    def inputs(self, x_np, lengths_np, positions) -> tuple:
        """A batch from sweepkit.train.batches as the model reads it: on the device, through the feature, with the
        real steps after the feature, normalized. positions: the batch rows' dataset positions (group normalization)."""
        import torch

        x = self.feature(to_device(x_np, self.device))
        lengths = None
        if lengths_np is not None:
            raw = torch.from_numpy(np.asarray(lengths_np, dtype=np.int64)).to(self.device)
            lengths = self.feature.lengths(raw).clamp_max(x.shape[-1])
        return self.normalizer(x, lengths, positions), lengths

    def predict(self, source, batch_size: int, window=None, with_aux: bool = False) -> Predictions:
        """Every row of the source, in eval mode with no augmentation: whole rows (window None), or the mean over its
        windows (window: settings with window and hop, in steps or samples)."""
        import torch

        self.eval()
        n = len(source)
        main_sums = aux_sums = None
        counts = np.zeros(n, dtype=np.float64)
        batches = (eval_batches(source, batch_size) if window is None
                   else window_batches(source, batch_size, window.window, window.hop))
        read_aux = with_aux and self.aux_head is not None
        start = time.perf_counter()
        with torch.no_grad():
            for ks, x_np, lengths_np in batches:
                x, lengths = self.inputs(x_np, lengths_np, source.positions[ks])
                with self.autocast():
                    if read_aux:
                        embedding = self.model.embed(x, lengths)
                        outputs = self.model.classify(embedding)
                        aux_outputs = self.aux_head(embedding)
                    else:
                        outputs = self.model(x, lengths)
                values = output_values(self.task, outputs).cpu().numpy().astype(np.float64)
                if main_sums is None:
                    main_sums = np.zeros((n, values.shape[1]), dtype=np.float64)
                if window is None:
                    main_sums[ks] = values
                else:
                    np.add.at(main_sums, ks, values)
                if read_aux:
                    aux_values = output_values(self.aux_task, aux_outputs).cpu().numpy().astype(np.float64)
                    if aux_sums is None:
                        aux_sums = np.zeros((n, aux_values.shape[1]), dtype=np.float64)
                    if window is None:
                        aux_sums[ks] = aux_values
                    else:
                        np.add.at(aux_sums, ks, aux_values)
                np.add.at(counts, ks, 1.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        if n == 0 or main_sums is None:
            raise ValueError("no rows to predict")
        if window is None:
            if not (counts == 1).all():
                raise RuntimeError("whole-row evaluation must see every row exactly once")
        else:
            if not (counts > 0).all():
                raise RuntimeError("window evaluation left a row without a window")
            main_sums /= counts[:, None]
            if aux_sums is not None:
                aux_sums /= counts[:, None]
        return Predictions(main_sums, aux_sums, seconds)


class Step:
    """The loss of one training batch (the order is in the module docstring). Returns (the loss to minimize, its
    parts: "main", and "penalty", "aux" and "contrastive" when present, each a 0-dimensional tensor)."""

    def __init__(self, network: Network, augments: dict, losses: Losses):
        self.network, self.augments, self.losses = network, augments, losses
        if (network.contrastive_head is not None) != (losses.contrastive_weight > 0):
            raise ValueError("a contrastive weight above 0 and a projection head go together")
        if network.contrastive_head is not None and "mixup" in augments:
            raise ValueError("the contrastive loss pairs rows by label, which mixup blends; use one or the other")
        if (network.aux_head is not None) != (losses.aux_task is not None):
            raise ValueError("an auxiliary head and its loss settings go together")

    def main_loss(self, outputs, target):
        task = self.losses.task
        if task == "classification":
            return soft_cross_entropy(outputs, target, self.losses.class_weights)
        if task == "multilabel":
            return multilabel_loss(outputs, target, self.losses.pos_weight)
        return regression_loss(outputs, target, self.losses.regression)

    def aux_loss(self, outputs, target):
        if self.losses.aux_task == "classification":
            return soft_cross_entropy(outputs, target, None)
        return regression_loss(outputs, target, self.losses.aux_loss)

    def loss(self, source, picked: np.ndarray, targets: Targets, crop, streams: Streams) -> tuple:
        import torch

        net = self.network
        device = net.device
        x_np, lengths_np = training_batch(source, picked, crop, streams.batches, self.augments.get("speed"),
                                          streams.speed)
        x, lengths = net.inputs(x_np, lengths_np, source.positions[picked])
        if "mask" in self.augments:
            x = augmentations.axis_masks(x, lengths, self.augments["mask"], streams.masks)
        if "noise" in self.augments:
            x = augmentations.add_noise(x, lengths, self.augments["noise"], streams.noise)
        t = torch.from_numpy(np.ascontiguousarray(targets.main[picked])).to(device)
        a = (torch.from_numpy(np.ascontiguousarray(targets.aux[picked])).to(device)
             if net.aux_head is not None else None)
        y = torch.from_numpy(targets.labels[picked]).to(device) if net.contrastive_head is not None else None
        if "mixup" in self.augments:
            weights, partner = augmentations.mixup_draw(self.augments["mixup"].alpha, int(picked.shape[0]),
                                                        streams.mixup)
            x, lengths, (t, a) = augmentations.mixup_mix(x, lengths, [t, a], weights, partner)
        with net.autocast():
            if not net.heads():
                outputs = net.model(x, lengths)
            else:
                embedding = net.model.embed(x, lengths)
                outputs = net.model.classify(embedding)
                aux_outputs = net.aux_head(embedding) if net.aux_head is not None else None
                projections = net.contrastive_head(embedding) if net.contrastive_head is not None else None
        main = self.main_loss(outputs, t)
        total, parts = main, {"main": main}
        # a spiking model's activity penalty (sweepkit.models.torch.lif); other models have none
        penalty = net.model.activity_penalty() if hasattr(net.model, "activity_penalty") else None
        if penalty is not None:
            total = total + penalty
            parts["penalty"] = penalty
        if net.aux_head is not None:
            parts["aux"] = self.aux_loss(aux_outputs, a)
            total = total + self.losses.aux_weight * parts["aux"]
        if net.contrastive_head is not None:
            parts["contrastive"] = supervised_contrastive_loss(projections, y, self.losses.contrastive_temperature)
            total = total + self.losses.contrastive_weight * parts["contrastive"]
        return total, parts


@dataclasses.dataclass
class Watch:
    """How the epoch loop scores a role after each epoch: its source, and score(predictions) -> {metric: value or
    None}, every metric of the task (None where a metric is undefined on these rows)."""

    source: object
    score: object


@dataclasses.dataclass
class Fit:
    curves: list                    # one entry per epoch run
    best_epoch: int                 # the epoch kept (the last epoch when there was no validation)
    best_value: float | None        # its value of the select metric (None without validation, or when undefined)
    best_state: dict | None         # Network.state() of that epoch; None without validation (the network holds the
                                    # last epoch's weights)
    best_train: Predictions | None  # the training rows' predictions at that epoch (eval_train)
    stopped_early: bool
    steps_per_epoch: int
    optimizer: dict                 # the optimizer's name and every setting it used, its own defaults included
    seconds: float


def better(value, best, direction: str, first: bool) -> bool:
    """Whether an epoch's value beats the best so far: the first epoch always does; an undefined value (None or NaN)
    never beats a defined one, and a defined one always beats an undefined best."""
    if first:
        return True
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if best is None or (isinstance(best, float) and math.isnan(best)):
        return True
    return value > best if direction == "max" else value < best


def steps_per_epoch(n_rows: int, batch_size: int, max_batches) -> int:
    steps = max(1, n_rows // batch_size)
    return steps if max_batches is None else min(steps, max_batches)


def optimizer_record(optimizer) -> dict:
    """The optimizer's class and every setting it uses, its library defaults included (betas, eps, ...), so a default
    never hides: model_params.json records it."""
    out = {"name": type(optimizer).__name__}
    for key, value in sorted(optimizer.defaults.items()):
        if isinstance(value, tuple):
            value = list(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(v, (bool, int, float)) for v in value):
            out[key] = value
    return out


def run_epochs(network: Network, step: Step, source, targets: Targets, settings, epochs: int, schedule_epochs: int,
               streams: Streams, log, val=None, select=None, train_watch=None, show_val=True) -> Fit:
    """Train for `epochs` epochs on a schedule of `schedule_epochs` (the refit stops early on the job's own schedule).
    val: a Watch of the validation rows and select = (metric, direction), which choose the epoch kept and stop early;
    without them every epoch runs and the network keeps the last epoch's weights. train_watch: a Watch of the
    training rows (eval_train). show_val False keeps validation numbers out of the log (the identity check fold,
    whose validation rows are test rows: a log line never carries a test number)."""
    import torch

    s = settings
    n = len(source)
    steps = steps_per_epoch(n, s.batch_size, s.max_batches_per_epoch)
    if n < s.batch_size:
        log(f"  NOTE {n} training rows, fewer than batch_size {s.batch_size}: one batch of all of them per epoch")
    total_steps, warmup_steps = schedule_epochs * steps, s.warmup_epochs * steps
    optimizer = torch.optim.AdamW(network.parameters(), lr=s.lr, weight_decay=s.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda k: lr_factor(k, warmup_steps, total_steps, s.schedule))
    curves, best_value, best_epoch, best_state, best_train, stopped = [], None, 0, None, None, False
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        network.train()
        epoch_start = time.perf_counter()
        order = train_order(targets.labels, s.class_balance, streams.batches, n)
        sums = {}
        epoch_lr = optimizer.param_groups[0]["lr"]          # the rate of the epoch's first step
        for b in range(steps):
            picked = order[b * s.batch_size:(b + 1) * s.batch_size]
            loss, parts = step.loss(source, picked, targets, s.crop, streams)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sums["train_loss"] = sums.get("train_loss", 0.0) + loss.detach()
            if len(parts) > 1:
                for name, value in parts.items():
                    sums[f"train_loss_{name}"] = sums.get(f"train_loss_{name}", 0.0) + value.detach()
        curve = {"epoch": epoch}
        for name, total in sums.items():
            curve[name] = float(total.item()) / steps
        if not math.isfinite(curve["train_loss"]):
            raise FloatingPointError(f"training loss is {curve['train_loss']} in epoch {epoch}")
        curve["lr"] = epoch_lr
        curve["train_seconds"] = round(time.perf_counter() - epoch_start, 3)
        train_pred, note = None, ""
        if train_watch is not None:
            train_pred = network.predict(train_watch.source, s.eval_batch_size, s.eval_window)
            values = train_watch.score(train_pred)
            curve.update({f"train_{k}": v for k, v in values.items()})
            curve["train_eval_seconds"] = round(train_pred.seconds, 3)
            if select is not None and values.get(select[0]) is not None:
                note = f" train {select[0]} {values[select[0]]:.4f}"
        if val is None:
            log(f"  epoch {epoch:3d} loss {curve['train_loss']:.4f} ({curve['train_seconds']:.1f}s)")
            curves.append(curve)
            best_epoch, best_train = epoch, train_pred
            continue
        pred = network.predict(val.source, s.eval_batch_size, s.eval_window)
        values = val.score(pred)
        curve.update({f"val_{k}": v for k, v in values.items()})
        curve["eval_seconds"] = round(pred.seconds, 3)
        value = values.get(select[0])
        improved = better(value, best_value, select[1], best_state is None)
        shown = ("undefined" if value is None else f"{value:.4f}") if show_val else "(not shown on this fold)"
        log(f"  epoch {epoch:3d} loss {curve['train_loss']:.4f}{note} val {select[0]} {shown} "
            f"({curve['train_seconds']:.1f}s + {pred.seconds:.1f}s){' best' if improved else ''}")
        curves.append(curve)
        if improved:
            best_epoch, best_value, best_state, best_train = epoch, value, network.state(), train_pred
        if s.early_stopping_patience is not None and epoch - best_epoch >= s.early_stopping_patience:
            stopped = True
            break
    return Fit(curves, best_epoch, best_value, best_state, best_train, stopped, steps, optimizer_record(optimizer),
               round(time.perf_counter() - start, 3))
