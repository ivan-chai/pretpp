import os

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from hotpp.data import PaddedBatch
from aligned_hpo import AlignedHPOptimizer, AuxiLearnOptimizer, BatchNormStatsSwitch
from .hpo_module import HPOModule


class MethodWrapper(torch.nn.Module):
    def __init__(self, module, method_name):
        super().__init__()
        self.module = module
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.module, self.method_name)(*args, **kwargs)


def func_call(module, method_name, params, buffers, *args, **kwargs):
    wrapper = MethodWrapper(module, method_name)
    params = {"module." + k: v for k, v in params.items()}
    buffers = {"module." + k: v for k, v in buffers.items()}
    with sdpa_kernel(SDPBackend.MATH):
        return torch.func.functional_call(wrapper, (params, buffers), args=args, kwargs=kwargs)


class HyperModule(HPOModule):
    """Tune loss weights using SGD.

    Args:
        hpo_losses: A list of losses to tune hyperparameters for.
        downstream_loss: The name of the downstream loss or a mapping from loss name to weight.
        initial_weights: Initial values (default to ones).
        hpo_params: Parameters of the HP optimizer.
        shared_prefix: The prefix for validation parameters weights.
        hp_group_params: Specific parameters for weights optimization (lr etc.).
        loss_group_params: Specific parameters for loss optimization (lr etc.).
        shared_group_params: Specific parameters for shared weights optimization (lr etc.).
        encoder_group_params: Specific parameters for encoder weights optimization (lr etc.).
        cache_embedding_gradients: Compute and cache embedding gradients for all heads via a single backward pass in the encoder-decoder mode.
        warmup_steps: Reset all except HPO statistics after the specified number of steps.
    """

    OPTIMIZERS = {
        "aligned-hpo": AlignedHPOptimizer,
        "auxilearn": AuxiLearnOptimizer
    }

    def __init__(self, *args, optimizer="aligned-hpo", **kwargs):
        super().__init__(*args, optimizer=optimizer, **kwargs)
        self.bn_stats_switch = BatchNormStatsSwitch(self)
        loss_weights = self.loss_weights
        del self.loss_weights
        self.loss_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(w))
            for w in loss_weights
        ])

    def training_step(self, batch, batch_idx):
        dataloader_idx = batch[0].payload.get("_dataloader_idx", None)
        opt = self.optimizers()
        if (dataloader_idx is None) or (dataloader_idx > 1):
            raise ValueError(f"When validation set is used for tuning, there must be exact 2 dataloaders. Got {dataloader_idx}")
        do_val_step = dataloader_idx == 1
        if do_val_step:
            self.eval()

        x, y = batch
        inputs, targets, _ = self._loss.prepare_batch(x, y)
        outputs, losses, metrics = self._compute_loss(inputs, targets)

        def after_backward_hook():
            if hasattr(self.trainer.strategy, "reduce_gradients"):
                self.trainer.strategy.reduce_gradients()
            if self.gradient_clip_val is not None:
                self.clip_gradients(opt.base_optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm)

        if do_val_step:
            loss = torch.stack([losses[name] * w for name, w in self.downstream_loss.items()]).sum()
            opt.zero_grad()
            self.manual_backward(loss)
            after_backward_hook()
            opt.val_step()
            metrics["val-loss"] = loss.item()
        else:
            def loss_fn(weights, params):
                val_state = dict(zip(self._shared_names, params))
                model_state = dict(zip(self._model_names, params[len(val_state):]))
                assert len(val_state) + len(model_state) == len(params)

                state_dict = val_state | model_state
                buffers = dict(self.named_buffers())
                with self.bn_stats_switch:
                    _, fn_losses, _ = func_call(self, "_compute_loss", state_dict, buffers, inputs, targets)
                return torch.stack([w * fn_losses[name] for name, w in zip(self.hpo_losses, weights)]).sum()

            with torch.no_grad():
                weights = [w.detach() for w in opt.weights]
            assert len(weights) == len(self.hpo_losses)
            loss = torch.stack([losses[name] * w for name, w in zip(self.hpo_losses, weights)]).sum()
            opt.zero_grad()
            self.manual_backward(loss)
            after_backward_hook()
            opt.train_step(loss_fn)
            metrics["loss"] = loss.item()

            if self.should_log:
                metrics["grad_norm"] = self._get_grad_norm()
                metrics.update(opt.metrics)

            # Make scheduler step if necessary.
            for config in self.trainer.lr_scheduler_configs:
                if config.interval != "step":
                    continue
                if config.frequency != 1:
                    raise NotImplementedError("Frequency in LR scheduler.")
                if config.reduce_on_plateau:
                    raise NotImplementedError("ReduceOnPlateau LR scheduler.")
                config.scheduler.step()
        self._log_metrics("train", len(x), None, losses, metrics, single_batch_metrics=None)
        if do_val_step:
            self.train()

    def configure_optimizers(self):
        shared_prefix = self.shared_prefix if self.shared_prefix is not None else "<none>"
        shared_params = [(k, v) for k, v in self.named_parameters() if v.requires_grad and not k.startswith("loss_weights.") and k.startswith(shared_prefix)]
        if (self.shared_prefix is not None) and (not shared_params):
            raise ValueError(f"No weights found for prefix: {self.shared_prefix} ({list(self.state_dict())})")
        model_params = [(k, v) for k, v in self.named_parameters() if v.requires_grad and not k.startswith("loss_weights.") and not k.startswith(shared_prefix)]

        self._shared_names, shared_params = zip(*sorted(shared_params))
        self._model_names, model_params = zip(*sorted(model_params))
        self._shared_shapes = [p.shape for p in shared_params]
        self._model_shapes = [p.shape for p in model_params]
        if len(shared_params) == 0:
            raise RuntimeError("Empty downstream head")
        params = [
            {"params": self.loss_weights},
            {"params": shared_params},
            {"params": model_params}
        ]
        if self.hp_group_params is not None:
            params[0].update(self.hp_group_params)
        if self.shared_group_params is not None:
            params[1].update(self.shared_group_params)
        if self.encoder_group_params is not None:
            params[2].update(self.encoder_group_params)
        optimizer = self.OPTIMIZERS[self.optimizer_type](params, self._optimizer_partial,
                                                         weights_names=self.hpo_losses,
                                                         weights_parametrization=lambda hiddens: [torch.nn.functional.softplus(h) for h in hiddens],
                                                         val_training_groups=(1,),
                                                         **(self.hpo_params or {}))
        configs = self._build_scheduler_configs(optimizer)
        if not configs:
            return optimizer
        return [optimizer], configs
