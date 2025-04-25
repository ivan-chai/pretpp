import torch

from hotpp.data import PaddedBatch
from hotpp.losses.common import ScaleGradient
from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """Global target prediction loss.

    Args:
        targets: A mapping from a target name to dictionary with "num_classes" and optional "weight" and "cast" fields.
        cls_token: A dictionary with field values for a CLS token (optional, typically for transformer models).
    """
    def __init__(self, targets, cls_token=None):
        super().__init__()
        for name, spec in targets.items():
            if "num_classes" not in spec:
                raise ValueError("Need 'num_classes' for each target.")
            unknown_fields = set(spec) - {"num_classes", "weight", "cast"}
            if unknown_fields:
                raise ValueError(f"Unknown fields in loss specification: {unknown_fields}")
        self._targets = targets
        self._order = list(sorted(targets))
        self._cls_token = cls_token

    @property
    def input_size(self):
        return sum([spec["num_classes"] for spec in self._targets.values()])

    @property
    def aggregate(self):
        # Use aggregation if there is no special token.
        return self._cls_token is None

    def prepare_inference_batch(self, inputs):
        """Extract model inputs for inference.

        Args:
            inputs: Input events with shape (B, L, *).

        Returns:
            Model inputs with shape (B, L', *).
        """
        if self._cls_token is not None:
            # Add CLS token to the end of inputs and add fake targets.
            new_inputs = {k: v for k, v in inputs.payload.items() if k not in inputs.seq_names}
            last_indices = inputs.seq_lens.unsqueeze(1)  # (B).
            b = len(inputs)
            for k, t in self._cls_token.items():
                v = inputs.payload[k]
                new_inputs[k] = torch.cat([v, v[:, -1:]], 1)  # (B, L, *).
                token = torch.full_like(v[:, 0], t)
                new_inputs[k].scatter_(1, last_indices.reshape(*([b] + [1] * (v.ndim - 1))), token)
            inputs = PaddedBatch(new_inputs, inputs.seq_lens + 1,
                                 seq_names={k for k in inputs.seq_names if k in self._cls_token})
        return inputs

    def prepare_batch(self, inputs, targets):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets: Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        inputs = self.prepare_inference_batch(inputs)
        targets = PaddedBatch({name: targets.payload[name] for name in self._targets}, targets.seq_lens,
                              seq_names={name for name in targets.seq_names if name in self._targets})
        if self._cls_token is not None:
            new_targets = {}
            for k in self._targets:
                v = targets.payload[k]
                if k not in targets.seq_names:
                    new_targets[k] = v
                else:
                    new_targets[k] = torch.cat([v, v[:, -1:]], 1)  # (B, L, *).
            targets = PaddedBatch(new_targets, targets.seq_lens + 1, seq_names=set(targets.seq_names) & set(self._targets))
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
            outputs = self._split_outputs(outputs)  # (B, 1, D).
        losses = {}
        metrics = {}
        for name in set(targets.payload) & set(outputs):
            spec = self._targets[name]
            if targets.payload[name].ndim != 1:
                raise NotImplementedError("Only global targets are supported.")
            logits = outputs[name]
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
            outputs = self._split_outputs(outputs)  # (B, 1, D).
        result = {}
        for name in set(self._targets) & set(outputs):
            result[name] = outputs[name].squeeze(1).argmax(-1)  # (B).
        return PaddedBatch(result, lengths, seq_names={})

    def _split_outputs(self, outputs):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        outputs, lengths = outputs.payload, outputs.seq_lens
        if self._cls_token is not None:
            last_indices = lengths - 1
        offset = 0
        result = {}
        for name in self._order:
            nc = self._targets[name]["num_classes"]
            output = outputs[..., offset:offset + nc]
            if output.ndim != 3:
                raise NotImplementedError("Expected outputs with shape (B, L, C).")
            if self._cls_token is not None:
                # Extract CLS token embedding.
                b = len(last_indices)
                output = output.take_along_dim(last_indices.reshape(*([b] + [1] * (output.ndim - 1))), 1).squeeze(1)
            else:
                if output.shape[1] != 1:
                    raise NotImplementedError("Expected aggregated embedding with shape (B, 1, C).")
                output = output.squeeze(1)
            result[name] = output
            offset += nc
        if offset != outputs.shape[-1]:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result
