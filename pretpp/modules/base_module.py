import warnings
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch

from hotpp.data import PaddedBatch


class BaseModule(pl.LightningModule):
    """Base module class.

    The model is composed of the following modules:
    1. input encoder, responsible for input-to-vector conversion,
    2. sequential encoder, which captures time dependencies,
    3. fc head for embeddings projection (optional),
    4. loss, which estimates likelihood and predictions.

    Input encoder and sequential encoder are combined within SeqEncoder from Pytorch Lifestream.

    Parameters
        seq_encoder: Backbone model, which includes input encoder and sequential encoder.
        loss: Training loss.
        timestamps_field: The name of the timestamps field.
        head_partial: Head model class which accepts encoder output and makes prediction.
        loss_head_partial: Loss preprocessing head.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        init_state_dict: Checkpoint to initialize all parameters except loss.
        val_metric: Validation set metric.
        test_metric: Test set metric.
    """
    def __init__(self, seq_encoder, loss,
                 timestamps_field="timestamps",
                 head_partial=None,
                 loss_head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 init_state_dict=None,
                 val_metric=None,
                 test_metric=None):

        super().__init__()
        self._timestamps_field = timestamps_field

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False

        self._val_metric = val_metric
        self._test_metric = test_metric
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        hidden_size = seq_encoder.hidden_size
        if head_partial is not None:
            self._head = head_partial(hidden_size)
            hidden_size = self._head.output_size
        else:
            self._head = torch.nn.Identity()
        self._loss_head = loss_head_partial(hidden_size, loss.input_size) if loss_head_partial is not None else torch.nn.Identity()

        if init_state_dict is not None:
            state_dict = {k: v for k, v in init_state_dict.items()
                          if not k.startswith("_loss.") and not k.startswith("_loss_head.")}
            for k, v in self._loss.named_parameters():
                state_dict["_loss." + k] = v
            self.load_state_dict(init_state_dict)

    def encode(self, x):
        """Compatibility with HoTPP."""
        return self(x), None

    def forward(self, x):
        """Extract embeddings."""
        hiddens, _ = self._seq_encoder(x)  # (B, L, D).
        hiddens = self._head(hiddens)
        return hiddens

    def training_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        outputs = self._loss_head(self.forward(inputs))  # (B, L, D).
        losses, metrics = self._loss(outputs, targets)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v)
        for k, v in metrics.items():
            self.log(f"train/{k}", v)
        self.log("train/loss", loss, prog_bar=True)
        self.log("sequence_length", x.seq_lens.float().mean(), prog_bar=True)
        if batch_idx == 0:
            with torch.no_grad():
                for k, v in self._compute_single_batch_metrics(x, inputs, outputs, targets).items():
                    self.log(f"train/{k}", v, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        outputs = self._loss_head(self.forward(inputs))  # (B, L, D).
        losses, metrics = self._loss(outputs, targets)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"val/{k}", v, batch_size=len(x))
        self.log("val/loss", loss, batch_size=len(x), prog_bar=True)
        if self._val_metric is not None:
            self._update_metric(self._val_metric, x, inputs, outputs, targets)
        if batch_idx == 0:
            for k, v in self._compute_single_batch_metrics(x, inputs, outputs, targets).items():
                self.log(f"val/{k}", v, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        outputs = self._loss_head(self.forward(inputs))  # (B, L, D).
        losses, metrics = self._loss(outputs, targets)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"test/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"test/{k}", v, batch_size=len(x))
        self.log("test/loss", loss, batch_size=len(x), prog_bar=True)
        if self._test_metric is not None:
            self._update_metric(self._test_metric, x, inputs, outputs, targets)
        if batch_idx == 0:
            for k, v in self._compute_single_batch_metrics(x, inputs, outputs, targets).items():
                self.log(f"test/{k}", v, batch_size=len(x))

    def on_validation_epoch_end(self):
        if self._val_metric is not None:
            metrics = self._val_metric.compute()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, prog_bar=True)
            self._val_metric.reset()

    def on_test_epoch_end(self):
        if self._test_metric is not None:
            metrics = self._test_metric.compute()
            for k, v in metrics.items():
                self.log(f"test/{k}", v, prog_bar=True)
            self._test_metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        if self._lr_scheduler_partial is None:
            return optimizer
        else:
            scheduler = self._lr_scheduler_partial(optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler = {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                }
            return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer=None, optimizer_idx=None):
        self.log("grad_norm", self._get_grad_norm(), prog_bar=True)

    @torch.autocast("cuda", enabled=False)
    def _update_metric(self, metric, x, inputs, outputs, targets):
        pass

    @torch.autocast("cuda", enabled=False)
    def _compute_single_batch_metrics(self, x, inputs, outputs, targets):
        """Slow debug metrics."""
        return self._loss.compute_metrics(inputs, outputs, targets)

    @torch.no_grad()
    def _get_grad_norm(self):
        names, parameters = zip(*[pair for pair in self.named_parameters() if pair[1].requires_grad])
        norms = torch.zeros(len(parameters), device=parameters[0].device)
        for i, (name, p) in enumerate(zip(names, parameters)):
            if p.grad is None:
                warnings.warn(f"No grad for {name}")
                continue
            norms[i] = p.grad.data.norm(2)
        return norms.square().sum().item() ** 0.5
