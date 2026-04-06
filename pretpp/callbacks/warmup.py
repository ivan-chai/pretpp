import os
import torch
from pytorch_lightning.callbacks import Callback

from pretpp.common import safe_mkdir, get_workdir
try:
    from aligned_hpo import AlignedHPOptimizer
    USE_ALIGNED_HPO = True
except ImportError:
    USE_ALIGNED_HPO = False


class AlignedHPOWarmupCallback(Callback):
    def __init__(self, warmup_steps):
        if not USE_ALIGNED_HPO:
            raise ImportError("Need aligned_hpo module for HPO warmup")
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_counter = 0
        self.warmup_done = False
        self.warmup_checkpoint_path = None

    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0]
        if not isinstance(opt, AlignedHPOptimizer):
            raise TypeError("HPO Warmup can be applied only to HPO optimizer.")
        if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
            root = get_workdir(trainer.logger, make=True)
            warmup_root = os.path.join(root, "warmup")
            safe_mkdir(warmup_root)
            self.warmup_checkpoint_path = os.path.join(warmup_root, "checkpoint.pth")
            checkpoint = {
                "model": pl_module.state_dict(),
                "optimizer": opt.state_dict(),
            }
            torch.save(checkpoint, self.warmup_checkpoint_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        new_counter = self.warmup_counter + 1
        load = (not self.warmup_done) and (new_counter >= self.warmup_steps)
        new_done = self.warmup_done or load
        try:
            if load:
                if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
                    if self.warmup_checkpoint_path is None:
                        raise RuntimeError("Initial checkpoint path is empty")
                    if not os.path.exists(self.warmup_checkpoint_path):
                        raise FileNotFoundError(f"Initial checkpoint is missing: {self.warmup_checkpoint_path}")
                    checkpoint = torch.load(self.warmup_checkpoint_path, weights_only=False)
                else:
                    checkpoint = None
                checkpoint = trainer.strategy.broadcast(checkpoint, src=0)

                opt = trainer.optimizers[0]

                # Save HPO results accumulated during warmup.
                current_grads_cache = {
                    k: (v.clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in opt._grads_cache.items()
                }
                current_buffers = {
                    k: (v.clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in opt._buffers.items()
                }
                current_group0_params = [p.data.clone() for p in opt.param_groups[0]["params"]]

                # Restore model and optimizer to pre-warmup state.
                pl_module.load_state_dict(checkpoint["model"])
                opt.load_state_dict(checkpoint["optimizer"])

                # Put back the HPO results we want to keep.
                opt._grads_cache.update(current_grads_cache)
                opt._buffers.update(current_buffers)
                for p, saved_data in zip(opt.param_groups[0]["params"], current_group0_params):
                    p.data.copy_(saved_data)
        finally:
            self.warmup_counter = new_counter
            self.warmup_done = new_done

    def state_dict(self):
        return {
            "warmup_counter": self.warmup_counter,
            "warmup_done": self.warmup_done,
            "warmup_checkpoint_path": self.warmup_checkpoint_path
        }

    def load_state_dict(self, state_dict):
        self.warmup_counter = state_dict["warmup_counter"]
        self.warmup_done = state_dict["warmup_done"]
        self.warmup_checkpoint_path = state_dict["warmup_checkpoint_path"]
