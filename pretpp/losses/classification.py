import torch

from hotpp.data import PaddedBatch
from hotpp.losses.common import ScaleGradient
from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """Global target prediction loss.

    Args:
        targets: A mapping from a target name to dictionary with "num_classes" and optional "weight" and "cast" fields.
    """
    def __init__(self, targets):
        super().__init__()
        for name, spec in targets.items():
            if "num_classes" not in spec:
                raise ValueError("Need 'num_classes' for each target.")
            unknown_fields = set(spec) - {"num_classes", "weight", "cast"}
            if unknown_fields:
                raise ValueError(f"Unknown fields in loss specification: {unknown_fields}")
        self._targets = targets
        self._order = list(sorted(targets))

    @property
    def input_size(self):
        return sum([spec["num_classes"] for spec in self._targets.values()])

    @property
    def aggregate(self):
        return True

    def prepare_batch(self, inputs, targets):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets: Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        targets = PaddedBatch({name: targets.payload[name] for name in self._targets}, targets.seq_lens,
                              seq_names={name for name in targets.seq_names if name in self._targets})
        return inputs, targets

    def forward(self, outputs, targets):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model outputs with shape (B, L, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L) or (B).

        Returns:
            Losses dict and metrics dict.
        """
        # Input is an aggregated embedding.
        if not isinstance(outputs, dict):
            outputs = self._split_outputs(outputs.payload)  # (B, 1, D).
        losses = {}
        metrics = {}
        for name in set(targets.payload) & set(outputs):
            spec = self._targets[name]
            if (outputs[name].ndim != 3) or (outputs[name].shape[1] != 1):
                raise NotImplementedError("Expected aggregated embedding with shape (B, 1, C).")
            if targets.payload[name].ndim != 1:
                raise NotImplementedError("Only global targets are supported.")
            logits = outputs[name].squeeze(1)
            target = targets.payload[name]
            if spec.get("cast", False):
                target = target.long()
            losses[name] = torch.nn.functional.cross_entropy(logits, target)
            if spec.get("weight", 1) != 1:
                losses[name] = ScaleGradient.apply(losses[name], spec["weight"])
            metrics[f"batch-accuracy-{name}"] = (logits.detach().argmax(1) == target).float().mean()
        return losses, metrics

    def predict(self, outputs):
        lengths = outputs.seq_lens
        if not isinstance(outputs, dict):
            outputs = self._split_outputs(outputs.payload)  # (B, 1, D).
        result = {}
        for name in set(self._targets) & set(outputs):
            if (outputs[name].ndim != 3) or (outputs[name].shape[1] != 1):
                raise NotImplementedError("Expected aggregated embedding with shape (B, 1, C).")
            result[name] = outputs[name].squeeze(1).argmax(-1)  # (B).
        return PaddedBatch(result, lengths, seq_names={})

    def _split_outputs(self, outputs):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        offset = 0
        result = {}
        for name in self._order:
            nc = self._targets[name]["num_classes"]
            result[name] = outputs[..., offset:offset + nc]
            offset += nc
        if offset != outputs.shape[-1]:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result
