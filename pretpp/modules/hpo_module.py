import warnings
import math
import pytorch_lightning as pl
import torch
import yaml
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
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


def recursive_map(data, func):
    """Recursively applies a function to all leaf nodes in a structure."""
    if isinstance(data, dict):
        return {k: recursive_map(v, func) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return type(data)(recursive_map(item, func) for item in data)
    return func(data)


class Structure(tuple):
    """Operations with nested structures."""
    def __new__(cls, structure):
        return super(Structure, cls).__new__(cls, [(Structure(v) if isinstance(v, (tuple, list)) else v) for v in structure])

    @property
    def size(self):
        return sum([(v.size if isinstance(v, Structure) else 1) for v in self])

    def flatten(self):
        return sum([(v.flatten() if isinstance(v, Structure) else [v]) for v in self], [])

    def replace_string(self, key, value):
        result = []
        for v in self:
            if isinstance(v, str) and (v == key):
                result.append(value)
            elif isinstance(v, Structure):
                result.append(v.replace_string(key, value))
            else:
                result.append(v)
        return Structure(result)

    def get_by_key(self, key, value_structure):
        if len(self) != len(value_structure):
            raise ValueError("Structures mismatch")
        results = []
        for k, v in zip(self, value_structure):
            if isinstance(k, Structure):
                assert isinstance(v, Structure)
                try:
                    results.append(k.get_by_key(key, v))
                except KeyError:
                    pass
            else:
                assert isinstance(k, str) and (not isinstance(v, Structure))
                if k == key:
                    results.append(v)
        if not results:
            raise KeyError(f"Key not found: {key}")
        if len(results) > 1:
            raise RuntimeError(f"Multiple matches for {key}")
        return results[0]

    def replace_keys_with_values(self, key_to_value):
        return Structure([(k.replace_keys_with_values(key_to_value) if isinstance(k, Structure)
                           else key_to_value.get(k))
                          for k in self])


class HPOModule(BaseModule):
    """Tune loss weights using SGD.

    Args:
        hpo_losses: A list of losses to tune hyperparameters for.
        downstream_loss: The name of the downstream loss or a mapping from loss name to weight.
        hpo_params: Parameters of the HP optimizer.
        shared_prefix: The prefix for shared parameters weights.
        hp_group_params: Specific parameters for weights optimization (lr etc.).
        loss_group_params: Specific parameters for loss optimization (lr etc.).
        shared_group_params: Specific parameters for shared weights optimization (lr etc.).
        encoder_group_params: Specific parameters for encoder weights optimization (lr etc.).
    """
    def __init__(self, seq_encoder, loss, hpo_losses, downstream_loss,
                 hpo_params=None, shared_prefix=None,
                 hp_group_params=None, loss_group_params=None,
                 shared_group_params=None, encoder_group_params=None,
                 cache_embedding_gradients=False,
                 **kwargs):
        super().__init__(seq_encoder, loss, **kwargs)
        self.automatic_optimization = False
        # Register loss parameters.
        self.hpo_losses = list(hpo_losses)
        self.downstream_loss = downstream_loss if isinstance(downstream_loss, Mapping) else {downstream_loss: 1}
        self.hpo_params = hpo_params
        self.shared_prefix = shared_prefix
        self.hp_group_params = hp_group_params
        self.loss_group_params = loss_group_params
        self.shared_group_params = shared_group_params
        self.encoder_group_params = encoder_group_params
        self.cache_embedding_gradients = cache_embedding_gradients
        self.loss_weights = torch.nn.Parameter(torch.ones([len(hpo_losses)]))
        self.gradient_clip_val = None

    @BaseModule.trainer.setter
    def trainer(self, trainer):
        if hasattr(trainer, "_gradient_clip_val_bck"):
            self.gradient_clip_val = trainer._gradient_clip_val_bck
        else:
            self.gradient_clip_val = trainer.gradient_clip_val
        trainer.gradient_clip_val = None
        trainer._gradient_clip_val_bck = self.gradient_clip_val
        BaseModule.trainer.fset(self, trainer)

    def training_step(self, batch, batch_idx):
        dataloader_idx = batch[0].payload.get("_dataloader_idx", None)
        opt = self.optimizers()
        if opt.tune_on_val and ((dataloader_idx is None) or (dataloader_idx > 1)):
            raise ValueError(f"When tune_on_val is used, there must be exact 2 dataloaders. Got {dataloader_idx}")

        cache_val_step = opt.tune_on_val and dataloader_idx == 1

        x, y = batch
        inputs, targets = self._loss.prepare_batch(x, y)

        # Similar to self._compute_loss, but with encoder_decoder logic.
        if self._loss.aggregate:
            embeddings = PaddedBatch(self._embed_impl(inputs).unsqueeze(1),
                                     torch.ones_like(inputs.seq_lens))  # (B, 1, D).
        else:
            embeddings, _ = self.forward_impl(inputs)

        if opt.encoder_decoder:
            # Detach embeddings.
            encoder_embeddings = embeddings
            embeddings = PaddedBatch(encoder_embeddings.payload.detach(), encoder_embeddings.seq_lens)
            embeddings.payload.requires_grad = True

        use_cached_grads = (not cache_val_step) and opt.encoder_decoder and self.cache_embedding_gradients

        if use_cached_grads:
            # Cache gradients for each head.
            opt.zero_grad()
            embeddings.payload.grad = None
            if opt.param_groups[2]["params"]:
                raise RuntimeError("Can't cache embedding gradients when shared paramaters are used.")
            loss_structure = Structure(self._loss.structure)
            with self._loss_projection.cache_input_grads() as projection:
                # Compute loss and backward.
                outputs = self._apply_to_outputs(embeddings, projection)  # (B, L, D).
                losses, metrics = self._loss(outputs, targets)
                loss = sum([losses[name] for name in set(loss_structure.flatten())])
                # Sync heads. Embedding grads stay local by design.
                self.manual_backward(loss)
                # Gather gradients.
                z_grads_cache = Structure(projection.input_grads)
                heads_grads_cache = Structure(projection.grads)
                assert z_grads_cache.size == loss_structure.size
                assert heads_grads_cache.size == loss_structure.size
        else:
            # Compute losses without backward.
            outputs = self._apply_to_outputs(embeddings, self._loss_projection)  # (B, L, D).
            losses, metrics = self._loss(outputs, targets)

        def closure(down, weights, retain_graph=False, stage=None):
            opt.zero_grad()
            if opt.encoder_decoder:
                embeddings.payload.grad = None
            assert len(weights) == len(self.hpo_losses)
            if use_cached_grads:
                # Set gradients from the cache.
                named_weights = {name: w * down for name, w in self.downstream_loss.items()} if down != 0 else {}
                named_weights.update({self.hpo_losses[i]: w for i, w in enumerate(weights) if w != 0})
                assert not (set(named_weights) - set(loss_structure.flatten()))
                embeddings.payload.grad = sum(w * loss_structure.get_by_key(name, z_grads_cache)
                                              for name, w in named_weights.items())
                self._loss_projection.grads = loss_structure.replace_keys_with_values(
                    {name: recursive_map(loss_structure.get_by_key(name, heads_grads_cache), lambda x: w * x)
                     for name, w in named_weights.items()})
            else:
                # Do backward pass.
                downstream_loss = sum([w * losses[name] for name, w in self.downstream_loss.items()])
                loss = sum([w * losses[k] for k, w in zip(self.hpo_losses, weights)], down * downstream_loss)
                self.manual_backward(loss, retain_graph=retain_graph)
                if stage == HPO_STAGE_DOWNSTREAM:
                    metrics["hpo_grad_norm_downstream"] = self._get_grad_norm(warn_empty_grads=False)
                elif isinstance(stage, int):
                    metrics[f"hpo_grad_norm_weight_{self.hpo_losses[stage]}"] = self._get_grad_norm(warn_empty_grads=False)
            if opt.encoder_decoder:
                return embeddings.payload.grad.flatten()

        if opt.encoder_decoder:
            def closure_encoder(z_grad):
                opt.zero_grad()
                # DDP synchronization will be made in after_backward_hook.
                encoder_embeddings.payload.backward(z_grad.reshape(*encoder_embeddings.payload.shape))
        else:
            closure_encoder = None

        def after_backward_hook():
            if self.gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm)
            if opt.encoder_decoder:
                # Synchronize gradients in DDP.
                with torch.enable_grad():
                    zero_loss = 0 * sum(p.flatten()[0] for group in opt.param_groups for p in group["params"] if p.requires_grad)
                    self.manual_backward(zero_loss)

        if cache_val_step:
            opt.val_step(closure, after_backward_hook=after_backward_hook)
        else:
            opt.hpo_step(closure, closure_encoder, after_backward_hook=after_backward_hook)
            hpo_grads = self.loss_weights.grad
            if hpo_grads is not None:
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
        shared_prefix = self.shared_prefix if self.shared_prefix is not None else "<none>"
        shared_params = [v for k, v in self.named_parameters() if v.requires_grad and k != "loss_weights" and k.startswith(shared_prefix)]
        if (self.shared_prefix is not None) and (not shared_params):
            raise ValueError(f"No weights found for prefix: {self.shared_prefix} ({list(self.state_dict())})")
        loss_params = [v for k, v in self.named_parameters() if v.requires_grad and k != "loss_weights" and not k.startswith(shared_prefix) and k.startswith("_loss")]
        model_params = [v for k, v in self.named_parameters() if v.requires_grad and k != "loss_weights" and not k.startswith(shared_prefix) and not k.startswith("_loss")]
        params = [
            {"params": [self.loss_weights]},
            {"params": loss_params},
            {"params": shared_params},
            {"params": model_params}
        ]
        if self.hp_group_params is not None:
            params[0].update(self.hp_group_params)
        if self.loss_group_params is not None:
            params[1].update(self.loss_group_params)
        if self.shared_group_params is not None:
            params[2].update(self.shared_group_params)
        if self.encoder_group_params is not None:
            params[3].update(self.encoder_group_params)
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
