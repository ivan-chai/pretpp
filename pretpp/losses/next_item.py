import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class NextItemLoss(BaseLoss):
    """Hybrid loss for next item prediction.

    Args:
        losses: Mapping from the feature name to the loss function.
        train_repeat: Repeat input sequence twice during training (for an auto-encoder training).
    """
    def __init__(self, losses, train_repeat=False):
        super().__init__()
        self._losses = torch.nn.ModuleDict(losses)
        self._order = list(sorted(losses))
        self._train_repeat = train_repeat

    @property
    def aggregate(self):
        return False

    @property
    def input_size(self):
        return sum([loss.input_size for loss in self._losses.values()])

    def prepare_batch(self, inputs, targets=None):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets (unused): Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        if self._train_repeat:
            # Repeat each sequence twice.
            l = inputs.shape[1]
            indices = torch.arange(2 * l, device=inputs.device)[None] % inputs.seq_lens.unsqueeze(1)  # (B, 2L).
            payload = {k: v for k, v in inputs.payload.items() if k not in inputs.seq_names}
            for k in inputs.seq_names:
                v = inputs.payload[k]  # (B, L, *).
                k_indices = indices.reshape(indices.shape + (1,) * (v.ndim  - 2))  # (B, 2L, *).
                payload[k] = v.take_along_dim(k_indices, 1)  # (B, 2L, *).
            inputs = PaddedBatch(payload, inputs.seq_lens * 2, seq_names=inputs.seq_names)
        # For the next-item loss inputs and targets are the same.
        # Offset is applied in base loss classes.
        return inputs, inputs

    def forward_impl(self, outputs, targets, reduction="mean"):
        inputs = targets
        # Compute losses. It is assumed that predictions lengths are equal to targets lengths.
        if not isinstance(outputs, dict):
            outputs = self._split_outputs(outputs.payload)
        mask = inputs.seq_len_mask.bool() if (inputs.seq_lens != inputs.shape[1]).any() else None
        losses = {}
        metrics = {}
        for name in sorted(set(inputs.payload) & set(outputs)):
            losses[name], loss_metrics = self._losses[name](inputs.payload[name], outputs[name], mask, reduction=reduction)
            for k, v in loss_metrics.items():
                metrics[f"{name}-{k}"] = v
        return losses, metrics

    def forward(self, outputs, targets, reduction="mean"):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model outputs with shape (B, L, *, D) or (B, 1, *, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L, *).
            reduction: `mean` or `none`.

        Returns:
            Losses dict and metrics dict.
        """
        if self._train_repeat:
            b, l = outputs.shape
            assert (l % 2 == 0) and (outputs.seq_lens % 2 == 0).all()
            half_l = l // 2
            half_lengths = outputs.seq_lens // 2
            first_outputs = PaddedBatch(outputs.payload[:, :half_l], half_lengths)
            index = torch.arange(half_l, device=outputs.device)[None] + half_lengths[:, None]  # (B, L).
            second_outputs = PaddedBatch(outputs.payload.take_along_dim(index.unsqueeze(2), 1), half_lengths)  # (B, L, D).

            first_targets = PaddedBatch({k: targets.payload[k][:, :half_l] for k in targets.seq_names}, half_lengths)
            second_targets = PaddedBatch({k: targets.payload[k].take_along_dim(index, 1) for k in targets.seq_names}, half_lengths)

            first_losses, first_metrics = self.forward_impl(first_outputs, first_targets)
            second_losses, second_metrics = self.forward_impl(second_outputs, second_targets)
            assert set(first_losses) == set(second_losses)
            assert set(first_metrics) == set(second_metrics)
            losses = {k: 0.5 * (first_losses[k] + second_losses[k]) for k in first_losses}
            metrics = {k: 0.5 * (first_metrics[k] + second_metrics[k]) for k in first_metrics}
            return losses, metrics
        else:
            return self.forward_impl(outputs, targets, reduction=reduction)

    def _split_outputs(self, outputs):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        offset = 0
        result = {}
        for name in self._order:
            loss = self._losses[name]
            result[name] = outputs[..., offset:offset + loss.input_size]
            offset += loss.input_size
        if offset != outputs.shape[-1]:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result
