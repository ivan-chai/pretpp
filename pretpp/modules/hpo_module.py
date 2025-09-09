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
from ..losses import HybridNextClsLoss
from ..optim import CorrHPOptimizer
from .base_module import BaseModule


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def to_logit(x, eps=1e-3):
    return math.log(y + esp) - math.log(1 - y + eps)


class HPOModule(BaseModule):
    """Tune loss weights using SGD."""
    def __init__(self, seq_encoder, loss, hpo_losses, down_loss, down_weight=1, **kwargs):
        super().__init__(seq_encoder, loss, **kwargs)
        self.automatic_optimization = False
        # Register loss parameters.
        self.hpo_losses = hpo_losses
        self.down_loss = down_loss
        self.down_weight = down_weight
        self.loss_weights = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.zeros([]))
            for k in self.hpo_losses
        })
        self.gradient_clip_val = None

    @BaseModule.trainer.setter
    def trainer(self, trainer):
        self.gradient_clip_val = trainer.gradient_clip_val
        trainer.gradient_clip_val = None
        BaseModule.trainer.fset(self, trainer)

    def training_step(self, batch, batch_idx):
        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)

        opt = self.optimizers()
        def closure(down, *weights):
            is_final = isinstance(weights[0], torch.Tensor)
            opt.zero_grad()
            outputs, losses, metrics = self._compute_loss(inputs, targets)
            if is_final:
                # Final call with actual parameters.
                report_loss = sum([l.detach().item() for l in losses.values()])
                metrics = metrics | {f"hpo_{k}": w.item() for k, w in zip(self.hpo_losses, weights)}
                self._log_metrics("train", len(x), report_loss, losses, metrics, single_batch_metrics=None)
            assert len(weights) == len(self.hpo_losses)
            loss = sum([w * losses[k] for k, w in zip(self.hpo_losses, weights)], down * losses[self.down_loss])
            self.manual_backward(loss)
            if is_final and (self.gradient_clip_val is not None):
                self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm)
        opt.hpo_step(closure=closure)

    def configure_optimizers(self):
        model_params = [v for k, v in self.named_parameters() if v.requires_grad and not k.startswith("loss_weights.")]
        params = [
            {"params": [self.loss_weights[k] for k in self.hpo_losses]},
            {"params": model_params}
        ]
        optimizer = CorrHPOptimizer(params, self._optimizer_partial, downstream_weight=self.down_weight)
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
