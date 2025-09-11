import warnings
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch
import yaml
import os
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from hotpp.data import PaddedBatch
from pretpp.nn import IdentityHead


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


class BaseModule(pl.LightningModule):
    """Base module class.

    The model is composed of the following modules:
    1. input embedder, responsible for input-to-vector conversion,
    2. sequential encoder, which captures time dependencies,
    3. encoder head for embeddings projection (optional),
    4. loss projection head (optional), for transforming embeddnigs into loss input,
    5. loss, which estimates likelihood and predictions.

    - Input encoder and sequential encoder are combined within SeqEncoder from Pytorch Lifestream.
    - Embeddings are generated from the output of the encoder head.

    Parameters
        seq_encoder: Backbone model, which includes input encoder and sequential encoder.
        loss: Training loss.
        timestamps_field: The name of the timestamps field.
        head_partial: Head model class which accepts encoder output and makes prediction.
            Input size is provided as positional argument.
        loss_projection_partial: Loss preprocessing head. Input and output sizes are provided as positional arguments.
        aggregator: Embeddings aggregator. By default try to use encoder aggregator.
        optimizer_partial: Optimizer init partial. Network parameters are missed.
        lr_scheduler_partial: Scheduler init partial. Optimizer are missed.
        init_state_dict: Checkpoint to initialize all parameters except loss.
        init_prefixes: A list of prefixes to initialize from checkpoint. By default, initialize all parameters
            except loss and loss projection.
        freeze_prefixes: A list of prefixes to exclude from training.
        peft_adapter: A function for applying PEFT to the encoder model.
        val_metric: Validation set metric.
        test_metric: Test set metric.
        downstream_validation_config: If provided, evaluate downstream metrics on each validation step.
    """
    def __init__(self, seq_encoder, loss,
                 timestamps_field="timestamps",
                 head_partial=None,
                 loss_projection_partial=None,
                 aggregator=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 init_state_dict=None,
                 init_prefixes=None,
                 freeze_prefixes=None,
                 peft_adapter=None,
                 val_metric=None,
                 test_metric=None,
                 downstream_validation_config=None):
        super().__init__()
        self._timestamps_field = timestamps_field

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False  # PyTorch Lifestream compatibility.

        self._val_metric = val_metric
        self._test_metric = test_metric
        self._downstream_config = downstream_validation_config
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial
        self._freeze_prefixes = freeze_prefixes
        self._peft_adapter = peft_adapter
        self._peft_applied = False

        if head_partial is None:
            head_partial = IdentityHead
        self._head = head_partial(seq_encoder.hidden_size)
        if loss_projection_partial is None:
            loss_projection_partial = IdentityHead
        self._loss_projection = loss_projection_partial(self._head.output_size, loss.input_size)
        self._aggregator = aggregator

        if init_state_dict is not None:
            if isinstance(init_state_dict, str):
                init_state_dict = torch.load(init_state_dict)
            if "state_dict" in init_state_dict:
                init_state_dict = init_state_dict["state_dict"]
            my_state_dict = self.state_dict()
            init_names = set(my_state_dict)
            if init_prefixes is None:
                init_names = {name for name in init_names
                              if not name.startswith("_loss.") and not name.startswith("_loss_projection.")}
            else:
                names = set()
                for prefix in init_prefixes:
                    names |= {name for name in init_names if name.startswith(prefix)}
                init_names = names
            print("Initialize", init_names)
            state_dict = {k: (init_state_dict[k] if k in init_names else my_state_dict[k])
                          for k in my_state_dict}
            self.load_state_dict(state_dict)

    def setup(self, stage):
        assert not self._peft_applied
        freeze_prefixes = self._freeze_prefixes or set()
        if (stage == "fit") and (self._peft_adapter is not None):
            self._seq_encoder = self._peft_adapter(self._seq_encoder)
            import peft
            if not isinstance(self._seq_encoder.peft_config["default"], peft.LoraConfig):
                raise NotImplementedError("Only LoRA adapters are supported")
            freeze_prefixes = {prefix.replace("_seq_encoder", "_seq_encoder.base_model.model")
                               for prefix in freeze_prefixes}
            self._peft_applied = True

        if freeze_prefixes:
            all_names = [name for name, p in self.named_parameters()]
            freeze_parameters = set()
            for prefix in freeze_prefixes:
                freeze_parameters |= {name for name in all_names if name.startswith(prefix)}
            print("Freeze", freeze_parameters)
            for name, p in self.named_parameters():
                if name in freeze_parameters:
                    p.requires_grad = False
                    freeze_parameters.discard(name)
            if freeze_parameters:
                raise RuntimeError(f"The following parameters were not found: {freeze_parameters}")

    def teardown(self, stage):
        if self._peft_applied:
            self._seq_encoder = self._seq_encoder.merge_and_unload()
            self._peft_applied = False

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            if self._peft_adapter is None:
                raise
        print("Try to restore from PEFT checkpoint")
        self._seq_encoder = self._peft_adapter(self._seq_encoder)
        try:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        finally:
            self._seq_encoder = self._seq_encoder.merge_and_unload()
        return result

    def forward(self, x, return_states=False):
        """Extract embeddings."""
        seq_encoder = self._seq_encoder if not self._peft_applied else self._seq_encoder.base_model
        hiddens, states = seq_encoder(x, return_states=return_states)  # (B, L, D).
        outputs = self._head(hiddens)
        return outputs, states

    def _embed_impl(self, inputs):
        if self._aggregator is None:
            embeddings = self._seq_encoder.embed(inputs)  # (B, D).
            embeddings = self._head(PaddedBatch(embeddings.unsqueeze(1), torch.ones_like(inputs.seq_lens))).payload.squeeze(1)  # (B, D).
        else:
            return_states = "full" if self._aggregator.need_states else False
            hiddens, states = self.forward(inputs, return_states=return_states)
            embeddings = self._aggregator(hiddens, states)
        return embeddings  # (B, D).

    def embed(self, x):
        """Compatibility with HoTPP."""
        inputs = self._loss.prepare_inference_batch(x)
        embeddings = self._embed_impl(inputs)
        return embeddings

    def _compute_loss(self, inputs, targets):
        if self._loss.aggregate:
            outputs = PaddedBatch(self._embed_impl(inputs).unsqueeze(1),
                                  torch.ones_like(inputs.seq_lens))  # (B, 1, D).
        else:
            outputs, _ = self.forward(inputs)
        outputs = self._loss_projection(outputs)  # (B, L, D).
        losses, metrics = self._loss(outputs, targets)
        return outputs, losses, metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        outputs, losses, metrics = self._compute_loss(inputs, targets)
        loss = sum(losses.values())

        # Log statistics.
        if batch_idx == 0:
            with torch.no_grad():
                single_batch_metrics = self._compute_single_batch_metrics(x, inputs, outputs, targets)
        else:
            single_batch_metrics = None
        mean_seq_len = x.seq_lens.float().mean()
        self._log_metrics("train", len(x), loss, losses, metrics, single_batch_metrics=None, mean_seq_len=mean_seq_len)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        outputs, losses, metrics = self._compute_loss(inputs, targets)
        loss = sum(losses.values())

        if self._val_metric is not None:
            self._update_metric(self._val_metric, x, inputs, outputs, targets)

        # Log statistics.
        if batch_idx == 0:
            single_batch_metrics = self._compute_single_batch_metrics(x, inputs, outputs, targets)
        else:
            single_batch_metrics = None
        self._log_metrics("val", len(x), loss, losses, metrics, single_batch_metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        outputs, losses, metrics = self._compute_loss(inputs, targets)
        loss = sum(losses.values())

        if self._test_metric is not None:
            self._update_metric(self._test_metric, x, inputs, outputs, targets)

        # Log statistics.
        if batch_idx == 0:
            single_batch_metrics = self._compute_single_batch_metrics(x, inputs, outputs, targets)
        else:
            single_batch_metrics = None
        self._log_metrics("test", len(x), loss, losses, metrics, single_batch_metrics)

    def on_validation_epoch_end(self):
        if self._val_metric is not None:
            metrics = self._val_metric.compute()
            metrics = {f"val/{k}": v for k, v in metrics.items()}
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
            self._val_metric.reset()

    def on_test_epoch_end(self):
        if self._test_metric is not None:
            metrics = self._test_metric.compute()
            metrics = {f"test/{k}": v for k, v in metrics.items()}
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
            self._test_metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial([v for k, v in self.named_parameters() if v.requires_grad])
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

    def configure_callbacks(self):
        callbacks = []
        if self._downstream_config is not None:
            from pretpp.downstream import DownstreamCallback, DownstreamCheckpointCallback
            with open(self._downstream_config, "r") as fp:
                downstream_config = OmegaConf.create(yaml.safe_load(fp))
            root = self.logger.root_dir or self.logger.save_dir or "lightning_logs"
            safe_mkdir(root)
            version = self.logger.version
            root = os.path.join(root, version if isinstance(version, str) else f"version_{version}")
            safe_mkdir(root)
            downstream_root = os.path.join(root, "downstream")

            monitor = None
            maximize = None
            for callback in self.trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    if (callback.monitor is None) or ("downstream" not in callback.monitor):
                        continue
                    monitor = callback.monitor
                    assert callback.mode in {"min", "max"}, f"Unexpected checkpoint selection mode: {callback.mode}"
                    maximize = callback.mode == "max"
            if (monitor is not None) and self.trainer.training:
                self.trainer.callbacks = [callback for callback in self.trainer.callbacks
                                          if not isinstance(callback, ModelCheckpoint)]
                print(f"Monitor downstream quality on {monitor}")
                callback = DownstreamCheckpointCallback(
                    root=downstream_root,
                    downstream_config=downstream_config,
                    monitor=monitor,
                    maximize=maximize
                )
            else:
                callback = DownstreamCallback(
                    root=downstream_root,
                    downstream_config=downstream_config
                )
            callbacks.append(callback)
        return callbacks

    def on_before_optimizer_step(self, optimizer=None, optimizer_idx=None):
        self.log("grad_norm", self._get_grad_norm(), prog_bar=True)

    @torch.autocast("cuda", enabled=False)
    def _update_metric(self, metric, x, inputs, outputs, targets):
        predictions = self._loss.predict(outputs)
        targets = self._loss.get_prediction_targets(targets)
        metric.update(predictions, targets)

    @torch.autocast("cuda", enabled=False)
    def _compute_single_batch_metrics(self, x, inputs, outputs, targets):
        """Slow debug metrics."""
        return self._loss.compute_metrics(inputs, outputs, targets)

    @torch.no_grad()
    def _get_grad_norm(self, warn_empty_grads=True):
        names, parameters = zip(*[pair for pair in self.named_parameters()
                                  if pair[1].requires_grad])
        norms = torch.zeros(len(parameters), device=parameters[0].device)
        for i, (name, p) in enumerate(zip(names, parameters)):
            if p.grad is None:
                if warn_empty_grads:
                    warnings.warn(f"No grad for {name}")
                continue
            norms[i] = p.grad.data.norm(2)
        return norms.square().sum().item() ** 0.5

    def _log_metrics(self, split, batch_size, loss, losses, metrics, single_batch_metrics=None, mean_seq_len=None):
        log_values = {}
        # Sorting fixes distributed aggregation errors.
        for k, v in sorted(losses.items()):
            log_values[f"{split}/loss_{k}"] = v
        for k, v in sorted(metrics.items()):
            log_values[f"{split}/{k}"] = v
        if single_batch_metrics is not None:
            for k, v in sorted(single_batch_metrics.items()):
                log_values[f"{split}/{k}"] = v

        log_values_bar = {
            f"{split}/loss": loss
        }
        if mean_seq_len is not None:
            log_values_bar["sequence_length"] = mean_seq_len

        self.log_dict(log_values, batch_size=batch_size, sync_dist=True)
        self.log_dict(log_values_bar, batch_size=batch_size, sync_dist=True, prog_bar=True)
