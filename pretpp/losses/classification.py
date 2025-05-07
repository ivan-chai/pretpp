import torch

from hotpp.data import PaddedBatch
from hotpp.losses.common import ScaleGradient
from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """Global target prediction loss.

    Args:
        targets: A mapping from a target name to dictionary with "num_classes" and optional "weight" and "cast" fields.
        cls_token: A dictionary with field values for a CLS token (optional, typically for transformer models).
        apply_to_all_outputs: Whether to compute loss for global classification targets with aggregated embedding or for each embedding.
    """
    def __init__(self, targets, cls_token=None, apply_to_all_outputs=False):
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
        self._apply_to_all_outputs = apply_to_all_outputs

    @property
    def input_size(self):
        return sum([spec["num_classes"] for spec in self._targets.values()])

    @property
    def aggregate(self):
        # Use aggregation if there is no special token.
        return (self._cls_token is None) and (not self._apply_to_all_outputs)

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
            last_indices = inputs.seq_lens  # (B).
            b = len(inputs)
            for k, t in self._cls_token.items():
                v = inputs.payload[k]
                new_inputs[k] = torch.cat([v, v[:, -1:]], 1)  # (B, L, *).
                token = torch.full_like(v[:, :1], t)
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
            targets: Target features with shape (B, L) or (B).

        Returns:
            Losses dict and metrics dict.
        """
        outputs, lengths = self._split_outputs(outputs)  # (B, L, D).
        last = (lengths - 1).clip(min=0)[:, None, None]  # (B, 1, 1).
        losses = {}
        metrics = {}
        for name in set(targets.payload) & set(outputs):
            spec = self._targets[name]
            target = targets.payload[name]  # (B).
            if target.ndim != 1:
                raise NotImplementedError("Only global targets are supported.")
            logits = outputs[name]  # (B, L, D).
            if spec.get("cast", False):
                target = target.long()
            losses[name] = torch.nn.functional.cross_entropy(logits.flatten(0, -2), target[:, None].repeat(1, logits.shape[1]).flatten())
            if spec.get("weight", 1) != 1:
                losses[name] = ScaleGradient.apply(losses[name], spec["weight"])
            with torch.no_grad():
                last_logits = logits.take_along_dim(last, 1).squeeze(1)  # (B, D).
                metrics[f"batch-accuracy-{name}"] = (last_logits.argmax(-1) == target).float().mean()
        return losses, metrics

    def predict(self, outputs):
        orig_lengths = outputs.seq_lens
        outputs, lengths = self._split_outputs(outputs)  # (B, L, D).
        last = (lengths - 1).clip(min=0)[:, None, None]  # (B, 1, 1).
        result = {}
        for name in set(self._targets) & set(outputs):
            result[name] = outputs[name].take_along_dim(last, 1).squeeze(1).argmax(-1)  # (B).
        return PaddedBatch(result, orig_lengths, seq_names={})

    def _split_outputs(self, outputs):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        outputs, lengths = outputs.payload, outputs.seq_lens
        if outputs.ndim != 3:
            raise NotImplementedError("Expected outputs with shape (B, L, C).")
        if self._apply_to_all_outputs:
            # Use all output vectors.
            pass
        elif self._cls_token is not None:
            # Extract CLS token embedding.
            last_indices = lengths - 1
            b = len(last_indices)
            outputs = outputs.take_along_dim(last_indices.reshape(*([b] + [1] * (outputs.ndim - 1))), 1)  # (B, 1, C).
            lengths = (lengths > 0).long()
        elif outputs.shape[1] != 1:
            raise NotImplementedError("Expected aggregated embedding with shape (B, 1, C).")
        offset = 0
        result = {}
        for name in self._order:
            nc = self._targets[name]["num_classes"]
            result[name] = outputs[..., offset:offset + nc]
            offset += nc
        if offset != outputs.shape[-1]:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result, lengths
