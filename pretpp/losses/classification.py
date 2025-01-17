import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """Global target prediction loss.

    Args:
        targets: A mapping from a target name to dictionary with "num_classes" and optional "weight" fields.
        aggregator: Embeddings aggregator for global classification.
    """
    def __init__(self, hidden_size, head_partial, aggregator, targets):
        super().__init__()
        for name, spec in targets.items():
            if "num_classes" not in spec:
                raise ValueError("Need 'num_classes' for each target.")
        self._order = list(sorted(targets))
        self._head = head_partial(hidden_size, sum([loss.input_size for loss in losses.values()]))
        self._targets = targets
        self._aggregator = aggregator

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
        outputs = self._split_outputs(self._head(outputs).payload)  # (B, L, D).
        losses = {}
        metrics = {}
        for name, spec in self._targets.items():
            if outputs[name].ndim != 3:
                raise NotImplementedError("Expected output with shape (B, L, C).")
            if targets.payload[name].ndim != 1:
                raise NotImplementedError("Only global targets are supported.")
            global_embeddings = self._aggregator(outputs[name])  # (B, D).
            losses[name] = torch.nn.functional.cross_entropy(global_embeddings, targets.payload[name])
        return losses, {}

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
