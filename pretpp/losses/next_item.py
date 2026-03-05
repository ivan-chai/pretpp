import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class NextItemLoss(BaseLoss):
    """Hybrid loss for next item prediction.

    Args:
        losses: Mapping from the feature name to the loss function.
        apply_to_tokens: Controls a subset of outputs to apply loss to. Either `regular` for non-special tokens or `all`.
    """
    def __init__(self, losses, apply_to_tokens="regular"):
        super().__init__()
        if apply_to_tokens not in {"all", "regular"}:
            raise ValueError(f"Unknown application strategy: {apply_to_tokens}")
        self._losses = torch.nn.ModuleDict(losses)
        self._order = list(sorted(losses))
        self._apply_to_tokens = apply_to_tokens

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
        # For the next-item loss inputs and targets are the same.
        # Offset is applied in base loss classes.
        return inputs, inputs

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
        inputs = targets
        outputs = self._select_token_subset(outputs)
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

    def _select_token_subset(self, outputs):
        if self._apply_to_tokens == "all":
            return outputs
        elif isinstance(outputs, dict):
            raise NotImplementedError("Can't combine application strategy with a dictionary loss input.")
        assert self._apply_to_tokens == "regular"
        special_mask = self.get_special_token_mask(outputs)
        outputs = self.unwrap_model_outputs(outputs)
        if special_mask is None:
            return outputs
        regular_mask = ~special_mask.bool()
        return self.select_embeddings_by_mask(outputs, regular_mask)

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
