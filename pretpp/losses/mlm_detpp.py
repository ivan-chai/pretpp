import torch

from hotpp.data import PaddedBatch
from hotpp.losses import DetectionLoss
from .base import BaseLoss


EVAL_FIELD = "_eval_mask"


class MLMDeTPPLoss(DetectionLoss, BaseLoss):
    """Wrapper around HoTPP Detection Loss."""
    def __init__(self, mask_token,
                 eval_fraction=0.1, mask_prob=0.8,
                 **detpp_kwargs):
        super().__init__(**detpp_kwargs)
        self._mask_token = mask_token
        self._eval_fraction = eval_fraction
        self._mask_prob = mask_prob

    @property
    def aggregate(self):
        return False

    def prepare_batch(self, inputs, targets=None):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets (unused): Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        b, l = inputs.shape
        times = inputs.payload[self._timestamps_field]  # (B, L).

        # Select loss evaluation indices, add them to inputs as boolean flags.
        assert l > 0
        k = max(1, int(l * self._eval_fraction))
        eval_rand = torch.rand(b, l - k, device=inputs.device) < self._eval_fraction  # (B, L).
        _, eval_indices = eval_rand.long().topk(k, dim=1)  # (B, K).
        eval_indices, _ = eval_indices.sort(dim=1)  # (B, K).
        eval_timestamps = inputs.payload["timestamps"].take_along_dim(eval_indices, 1)  # (B, K).
        enable = eval_indices < inputs.seq_lens[:, None]
        enable[:, 1:] = (eval_timestamps[:, 1:] - eval_timestamps[:, :-1]) > self._horizon
        eval_indices = torch.where(enable, eval_indices, torch.arange(l - k, l, device=inputs.device)[None].expand(b, k))
        eval_indices, _ = eval_indices.sort(dim=1)  # (B, K).
        eval_mask = torch.zeros(b, l, device=inputs.device, dtype=torch.bool).scatter_(1, eval_indices, torch.ones_like(eval_indices, dtype=torch.bool))

        # Mask horizons for selected evaluation indices.
        final_times = times.take_along_dim((inputs.seq_lens - 1).clip(min=0).unsqueeze(1), 1).squeeze(1)  # (B).
        start_indices = (eval_indices + 1).clip(max=l - 1)  # (B, K).
        start_times = times.take_along_dim(start_indices, dim=1)  # (B, K).
        start_times = torch.where(start_indices == l - 1, final_times.unsqueeze(1).expand(b, k), start_times)
        end_times = start_times + self._horizon  # (B, K).
        out_horizon = end_times[:, :, None] <= times[:, None, :]  # (B, K, L).
        end_values, end_indices = out_horizon.max(2)  # (B, K).
        end_indices = torch.where(end_values, end_indices, inputs.seq_lens[:, None].expand(b, k))  # (B, K).
        rng = torch.arange(l, device=inputs.device)[None, None]  # (1, 1, L).
        mask = torch.logical_and(rng >= start_indices.unsqueeze(2), rng < end_indices.unsqueeze(2))  # (B, K, L).
        mask = mask.logical_and_(torch.rand(b, k, 1, device=inputs.device) < self._mask_prob)
        mask = mask.any(1)  # (B, L).
        new_inputs = inputs.payload.copy()
        for field, value in self._mask_token.items():
            new_inputs[field] = inputs.payload[field].masked_fill(mask, value)

        # Return masked inputs and GT inputs, served as targets.
        new_inputs = PaddedBatch(new_inputs, inputs.seq_lens,
                                 seq_names=set(inputs.seq_names) | {EVAL_FIELD})
        targets = PaddedBatch(inputs.payload | {EVAL_FIELD: eval_mask}, inputs.seq_lens,
                              seq_names=set(inputs.seq_names) | {EVAL_FIELD})
        return new_inputs, targets

    def get_loss_indices(self, inputs):
        b, l = inputs.shape
        index_mask = inputs.payload[EVAL_FIELD]  # (B, L).
        n_indices = index_mask.sum(1)  # (B).
        assert (n_indices == n_indices[0]).all()
        k = n_indices[0].item()
        indices = torch.nonzero(index_mask)[:, 1].reshape(b, k)  # TODO: debug.

        lengths = (indices < inputs.seq_lens[:, None]).sum(1)
        full_mask = indices + self._prefetch_k < inputs.seq_lens[:, None]
        return PaddedBatch({"index": indices,
                            "full_mask": full_mask},
                           lengths)

    def forward(self, outputs, targets):
        """Compute loss and metrics.

        Args:
            outputs: Model outputs with shape (B, L, *, D) or (B, 1, *, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L, *).

        Returns:
            Losses dict and metrics dict.
        """
        return super().forward(targets, outputs, states=None)
