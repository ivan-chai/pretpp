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
from aligned_hpo import MultiTaskOptimizer
from .base_module import BaseModule


def to_sequence_length(x, length, padding=0):
    if x.shape[-2] >= length:
        return x[..., :length, :]
    padding_shape = list(x.shape)
    padding_shape[-2] = length - x.shape[-2]
    padding = torch.full(padding_shape, padding, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], -2)


class MTLModule(BaseModule):
    """Tune loss weights using SGD.

    Args:
        mtl_losses: A list of losses to tune hyperparameters for.
        mtl_params: Parameters of the HP optimizer.
        shared_prefix: The prefix for shared parameters weights.
        hp_group_params: Specific parameters for weights optimization (lr etc.).
        loss_group_params: Specific parameters for loss optimization (lr etc.).
        shared_group_params: Specific parameters for shared weights optimization (lr etc.).
    """
    def __init__(self, seq_encoder, loss, mtl_losses,
                 mtl_params=None, shared_prefix=None,
                 hp_group_params=None, loss_group_params=None, shared_group_params=None,
                 **kwargs):
        super().__init__(seq_encoder, loss, **kwargs)
        self.automatic_optimization = False
        # Register loss parameters.
        self.mtl_losses = list(mtl_losses)
        self.mtl_params = mtl_params
        self.shared_prefix = shared_prefix
        self.loss_group_params = loss_group_params
        self.shared_group_params = shared_group_params
        self.register_buffer("n_weights_updates", torch.zeros([], dtype=torch.long))
        self.register_buffer("avg_weights", torch.zeros(len(self.mtl_losses)))
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

        def closure(weights, retain_graph=False, stage=None):
            opt.zero_grad()
            if opt.encoder_decoder:
                z.payload.grad = None
            assert len(weights) == len(self.mtl_losses)
            loss = sum([w * losses[k] for k, w in zip(self.mtl_losses, weights)])
            self.manual_backward(loss, retain_graph=retain_graph)
            metrics[f"mtl_grad_norm_weight_{self.mtl_losses[stage]}"] = self._get_grad_norm(warn_empty_grads=False)
            if opt.encoder_decoder:
                b, _, d = z.payload.shape
                grads = z.payload.grad  # (B, L, D).
                grads = to_sequence_length(grads, self.max_sequence_length)  # (B, L, D).
                return grads.flatten()

        if opt.encoder_decoder:
            def closure_encoder(z_grad):
                opt.zero_grad()
                b, _, d = z.payload.shape
                z_grad = z_grad.reshape(b, -1, d)
                if z_grad.shape[1] >= embeddings.shape[1]:
                    self.manual_backward(embeddings.payload, z_grad[:, :embeddings.shape[1]])
                else:
                    self.manual_backward(embeddings.payload[:, :z_grad.shape[1]], z_grad)
        else:
            closure_encoder = None

        grad_clip_fn = lambda: self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm) if self.gradient_clip_val is not None else None

        final_weights = opt.mtl_step(closure, closure_encoder, after_backward_hook=grad_clip_fn)
        metrics.update({f"mtl_{name}": w.item() for name, w in zip(self.mtl_losses, final_weights)})

        self.n_weights_updates += 1
        self.avg_weights *= (self.n_weights_updates - 1) / self.n_weights_updates
        self.avg_weights += final_weights / self.n_weights_updates
        metrics.update({f"mtl_avg_{name}": self.avg_weights[i] for i, name in enumerate(self.mtl_losses)})
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
        shared_prefix = self.shared_prefix if self.shared_prefix is not None else "<none>"
        shared_params = [v for k, v in self.named_parameters() if v.requires_grad and k.startswith(shared_prefix)]
        if (self.shared_prefix is not None) and (not shared_params):
            raise ValueError(f"No weights found for prefix: {self.shared_prefix} ({list(self.state_dict())})")
        loss_params = [v for k, v in self.named_parameters() if v.requires_grad and not k.startswith(shared_prefix) and k.startswith("_loss")]
        model_params = [v for k, v in self.named_parameters() if v.requires_grad and not k.startswith(shared_prefix) and not k.startswith("_loss")]
        params = [
            {"params": loss_params},
            {"params": shared_params},
            {"params": model_params}
        ]
        if self.loss_group_params is not None:
            params[0].update(self.loss_group_params)
        if self.shared_group_params is not None:
            params[1].update(self.shared_group_params)
        optimizer = MultiTaskOptimizer(self.mtl_losses, params, self._optimizer_partial,
                                       **(self.mtl_params or {}))
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
