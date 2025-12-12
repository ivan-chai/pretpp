import warnings
from abc import ABC, abstractmethod
import math
import pytorch_lightning as pl
import torch
import yaml
import os
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from hotpp.data import PaddedBatch
from pretpp.nn import IdentityHead
from aligned_hpo import AlignedHPOptimizer, HPO_STAGE_DOWNSTREAM
from .base_module import BaseModule


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def to_logit(x, eps=1e-3):
    return math.log(y + esp) - math.log(1 - y + eps)


def to_sequence_length(x, length, padding=0):
    if x.shape[1] >= length:
        return x[:, :length]
    padding_shape = list(x.shape)
    padding_shape[1] = length - x.shape[1]
    padding = torch.full(padding_shape, padding, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], 1)


class HPOModule(BaseModule):
    """Tune loss weights using SGD.

    Args:
        hpo_losses: A list of losses to tune hyperparameters for.
        downstream_loss: The name of the downstream loss.
        hpo_params: Parameters of the HP optimizer.
        hp_group_params: Specific parameters for weights optimization (lr etc.).
        tune_on_val: Use validation data for hyperparameter tuning.
    """
    def __init__(self, seq_encoder, loss, hpo_losses, downstream_loss,
                 hpo_params=None, hp_group_params=None, tune_on_val=False, **kwargs):
        super().__init__(seq_encoder, loss, **kwargs)
        self.automatic_optimization = False
        # Register loss parameters.
        self.hpo_losses = list(hpo_losses)
        self.downstream_loss = downstream_loss
        self.hpo_params = hpo_params
        self.hp_group_params = hp_group_params
        self.tune_on_val = tune_on_val
        self.loss_weights = torch.nn.Parameter(torch.ones([len(hpo_losses)]))
        self.register_buffer("n_weights_updates", torch.zeros([], dtype=torch.long))
        self.register_buffer("avg_weights", torch.zeros(len(self.hpo_losses)))
        self.gradient_clip_val = None
        self.max_sequence_length = None

    @BaseModule.trainer.setter
    def trainer(self, trainer):
        self.gradient_clip_val = trainer.gradient_clip_val
        trainer.gradient_clip_val = None
        BaseModule.trainer.fset(self, trainer)

        if self.max_sequence_length is None:
            datamodule = trainer.datamodule
            for split in trainer.datamodule.splits:
                self.max_sequence_length = max(self.max_sequence_length or 0, getattr(trainer.datamodule, f"{split}_data").max_length)

    def training_step(self, batch, batch_idx):
        dataloader_idx = batch[0].payload.get("_dataloader_idx", None)
        if self.tune_on_val and ((dataloader_idx is None) or (dataloader_idx > 1)):
            raise ValueError(f"When tune_on_val is used, there must be exact 2 dataloaders. Got {dataloader_idx}")

        opt = self.optimizers()
        if opt.encoder_decoder and not self.max_sequence_length:
            raise RuntimeError("Failed to fetch the maximum sequence length.")

        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)
        if opt.encoder_decoder:
            if self._loss.aggregate:
                embeddings = PaddedBatch(self._embed_impl(inputs).unsqueeze(1),
                                         torch.ones_like(inputs.seq_lens))  # (B, 1, D).
            else:
                embeddings, _ = self.forward(inputs)
            z = PaddedBatch(embeddings.payload.detach().clone(), embeddings.seq_lens)
            z.payload.requires_grad = True
            outputs = self._loss_projection(z)  # (B, L, D).
            losses, metrics = self._loss(outputs, targets)
        else:
            outputs, losses, metrics = self._compute_loss(inputs, targets)

        def closure(down, weights, retain_graph=False, stage=None):
            opt.zero_grad()
            assert len(weights) == len(self.hpo_losses)
            loss = sum([w * losses[k] for k, w in zip(self.hpo_losses, weights)], down * losses[self.downstream_loss])
            self.manual_backward(loss, retain_graph=retain_graph)
            if stage == HPO_STAGE_DOWNSTREAM:
                metrics["hpo_grad_norm_downstream"] = self._get_grad_norm(warn_empty_grads=False)
            elif isinstance(stage, int):
                metrics[f"hpo_grad_norm_weight_{stage}"] = self._get_grad_norm(warn_empty_grads=False)
            if opt.encoder_decoder:
                return to_sequence_length(z.payload.grad, self.max_sequence_length).flatten()

        if opt.encoder_decoder:
            def closure_encoder(z_grad):
                opt.zero_grad()
                target_tensor = to_sequence_length(embeddings.payload, self.max_sequence_length).flatten()
                self.manual_backward(target_tensor, z_grad)
        else:
            closure_encoder = None

        if self.tune_on_val and dataloader_idx == 1:
            opt.cache_downstream(closure)
        else:
            grad_clip_fn = lambda: self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm) if self.gradient_clip_val is not None else None
            final_weights = opt.hpo_step(closure, closure_encoder, use_cached_downstream=self.tune_on_val,
                                         after_backward_hook=grad_clip_fn)
            metrics.update({f"hpo_{name}": w.item() for name, w in zip(self.hpo_losses, final_weights)})

            self.n_weights_updates += 1
            self.avg_weights *= (self.n_weights_updates - 1) / self.n_weights_updates
            self.avg_weights += final_weights / self.n_weights_updates
            metrics.update({f"hpo_avg_{name}": self.avg_weights[i] for i, name in enumerate(self.hpo_losses)})

            hpo_grads = self.loss_weights.grad
            if hpo_grads:
                hpo_grads = torch.stack(hpo_grads)
                hpo_grad_norm = torch.linalg.norm(hpo_grads)
                metrics["hpo_grad_norm"] = hpo_grad_norm
            metrics.update(opt.metrics)
            self._log_metrics("train", len(x), None, losses, metrics, single_batch_metrics=None)

        # Make scheduler step if necessary.
        for config in self.trainer.lr_scheduler_configs:
            if config.interval != "step":
                continue
            if config.frequency != 1:
                raise NotImplementedError("Frequency in LR scheduler.")
            if config.reduce_on_plateau:
                raise NotImplementedError("ReduceOnPlateau LR scheduler.")
            config.scheduler.step()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # Make scheduler step if necessary.
        for config in self.trainer.lr_scheduler_configs:
            if config.interval != "epoch":
                continue
            if config.frequency != 1:
                raise NotImplementedError("Frequency in LR scheduler.")
            if config.reduce_on_plateau:
                raise NotImplementedError("ReduceOnPlateau LR scheduler.")
            config.scheduler.step()

    def configure_optimizers(self):
        model_params = [v for k, v in self.named_parameters() if v.requires_grad and k != "loss_weights" and not k.startswith("_loss")]
        loss_params = [v for k, v in self.named_parameters() if v.requires_grad and k != "loss_weights" and k.startswith("_loss")]
        params = [
            {"params": [self.loss_weights]},
            {"params": loss_params},
            {"params": model_params}
        ]
        if self.hp_group_params is not None:
            params[0].update(self.hp_group_params)
        optimizer = AlignedHPOptimizer(params, self._optimizer_partial,
                                       weights_names=self.hpo_losses,
                                       **(self.hpo_params or {}))
        if self._lr_scheduler_partial is None:
            return optimizer
        else:
            scheduler = self._lr_scheduler_partial(optimizer)
            scheduler = {
                "scheduler": scheduler,
                "interval": getattr(scheduler, "default_interval", "epoch")
            }
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler["monitor"] = "val/loss"  # TODO: get metric from trainer.
            return [optimizer], [scheduler]
