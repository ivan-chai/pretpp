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
from ..optim import CorrHPOptimizer, HPO_STAGE_DOWNSTREAM, HPO_STAGE_FINAL
from .base_module import BaseModule


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def to_logit(x, eps=1e-3):
    return math.log(y + esp) - math.log(1 - y + eps)


class HPOModule(BaseModule):
    """Tune loss weights using SGD.

    Args:
        hpo_losses: A list of losses to tune hyperparameters for.
        downstream_loss: The name of the downstream loss.
        hpo_lr: A custom LR for hyperparameters.
        hpo_params: Parameters of the HP optimizer.
    """
    def __init__(self, seq_encoder, loss, hpo_losses, downstream_loss,
                 hpo_lr=None, hpo_params=None, **kwargs):
        super().__init__(seq_encoder, loss, **kwargs)
        self.automatic_optimization = False
        # Register loss parameters.
        self.hpo_losses = hpo_losses
        self.downstream_loss = downstream_loss
        self.hpo_lr = hpo_lr
        self.hpo_params = hpo_params
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
        outputs, losses, metrics = self._compute_loss(inputs, targets)
        final_loss = next(iter(losses.values())).clone()

        opt = self.optimizers()
        def closure(down, *weights, stage=None):
            opt.zero_grad()
            assert len(weights) == len(self.hpo_losses)
            loss = sum([w * losses[k] for k, w in zip(self.hpo_losses, weights)], down * losses[self.downstream_loss])
            self.manual_backward(loss, retain_graph=stage != HPO_STAGE_FINAL)
            if stage == HPO_STAGE_DOWNSTREAM:
                metrics["hpo_grad_norm_downstream"] = self._get_grad_norm(warn_empty_grads=False)
            elif stage == HPO_STAGE_FINAL:
                final_loss.copy_(loss)
                metrics.update({f"hpo_{name}": w.item() for name, w in zip(self.hpo_losses, weights)})
                if self.gradient_clip_val is not None:
                    self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm)
            else:
                assert isinstance(stage, int)
                metrics[f"hpo_grad_norm_weight_{stage}"] = self._get_grad_norm(warn_empty_grads=False)
        opt.hpo_step(closure=closure)
        hpo_grads = torch.stack([w.grad for w in self.loss_weights.values()])
        hpo_grad_norm = torch.linalg.norm(hpo_grads)
        metrics["hpo_grad_norm"] = hpo_grad_norm
        self._log_metrics("train", len(x), final_loss, losses, metrics, single_batch_metrics=None)

    def configure_optimizers(self):
        model_params = [v for k, v in self.named_parameters() if v.requires_grad and not k.startswith("loss_weights.")]
        params = [
            {"params": [self.loss_weights[k] for k in self.hpo_losses]},
            {"params": model_params}
        ]
        if self.hpo_lr is not None:
            params[0]["lr"] = self.hpo_lr
        optimizer = CorrHPOptimizer(params, self._optimizer_partial,
                                    **(self.hpo_params or {}))
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
