import torch

from hotpp.data import PaddedBatch
from hotpp.losses.common import ScaleGradient
from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """Global target prediction loss.

    Args:
        targets: A mapping from a target name to dictionary with "num_classes" and optional "weight" and "cast" fields.
        cls_token: A dictionary with field values for a CLS token (optional, typically for transformer models).
        overwrite_timestamp: Assign the latest timestamp to the CLS token.
        apply_to_all_outputs: Whether to compute loss for global classification targets with aggregated embedding or for each embedding.
        drop_nans: Exclude elements with nan targets.
    """
    def __init__(self, targets=None, cls_token=None, overwrite_timestamp=False, apply_to_all_outputs=False,
                 drop_nans=False, local_targets=None, local_targets_indices_field=None):
        if (not targets) and not (local_targets):
            raise ValueError("Need at least one of targets or local_targets")
        if (cls_token is not None) and local_targets:
            raise NotImplementedError("CLS token with local targets")
        if (cls_token is not None) and apply_to_all_outputs:
            raise ValueError("Can't mix CLS token with apply-to-all-outputs.")
        if local_target and (local_targets_indices_field is None):
            raise ValueError("Need indices field for local targets.")
        super().__init__()
        for name, spec in targets.items():
            if "num_classes" not in spec:
                raise ValueError("Need 'num_classes' for each target.")
            unknown_fields = set(spec) - {"num_classes", "weight", "cast"}
            if unknown_fields:
                raise ValueError(f"Unknown fields in loss specification: {unknown_fields}")
        self._targets = targets
        self._local_targets = local_targets
        self._local_targets_indices_field = local_targets_indices_field
        self._order = list(sorted(targets or [])) + list(sorted(local_targets.keys() or []))
        self._cls_token = cls_token
        self._overwrite_timestamp = overwrite_timestamp
        self._apply_to_all_outputs = apply_to_all_outputs
        self._drop_nans = drop_nans

    @property
    def input_size(self):
        return sum([spec["num_classes"] for spec in self._targets.values()]) + sum([spec["num_classes"] for spec in self._local_targets.values()])

    @property
    def aggregate(self):
        # Use aggregation if there is no special token.
        return (self._cls_token is None) and (not self._apply_to_all_outputs) and (not self._local_targets)

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
                v = inputs.payload[k]  # (B, L, *).
                new_inputs[k] = torch.cat([v, v[:, -1:]], 1)  # (B, L + 1, *).
                if self._overwrite_timestamp and (k == "timestamps"):
                    assert v.ndim == 2  # (B, L).
                    token = v.take_along_dim((last_indices - 1).clip(min=0)[:, None], 1)  # (B, 1).
                else:
                    token = torch.full_like(v[:, :1], t)  # (B, 1, *).
                last_indices_expanded = last_indices.reshape(*([b] + [1] * (v.ndim - 1)))  # (B, 1, ..., 1).
                last_indices_expanded = last_indices_expanded.expand(*([b, 1] + list(v.shape[2:])))  # (B, 1, *).
                new_inputs[k].scatter_(1, last_indices_expanded, token)
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
        losses = {}
        metrics = {}

        # Global targets.
        last = (lengths - 1).clip(min=0)[:, None, None]  # (B, 1, 1).
        for name in sorted(set(targets.payload) & set(outputs) & set(self._targets)):
            spec = self._targets[name]
            target = targets.payload[name]  # (B).
            if target.ndim != 1:
                raise NotImplementedError("Only global targets are supported.")
            logits = outputs[name]  # (B, L, D).
            if self._drop_nans:
                mask = target.isfinite()
                nlabels = mask.sum()
                metrics[f"nans-{name}"] = mask.numel() - nlabels
                if nlabels == 0:
                    losses[name] = logits.sum() * 0
                    continue
                logits = logits[mask]
                target = target[mask]
            else:
                mask = slice(None, None, None)
            if spec.get("cast", False):
                target = target.long()
            losses[name] = torch.nn.functional.cross_entropy(logits.flatten(0, -2), target[:, None].repeat(1, logits.shape[1]).flatten())
            if spec.get("weight", 1) != 1:
                losses[name] = ScaleGradient.apply(losses[name], spec["weight"])
            with torch.no_grad():
                last_logits = logits.take_along_dim(last[mask], 1).squeeze(1)  # (B, D).
                metrics[f"batch-accuracy-{name}"] = (last_logits.argmax(-1) == target).float().mean()

        # Local targets.
        for name in sorted(set(targets.payload) & set(outputs) & set(self._local_targets)):
            spec = self._local_targets[name]
            target = targets.payload[name]  # (B, M).
            logits = outputs[name]  # (B, L, D).
            indices = targets.payload[self._local_targets_indices_field].long()  # (B, M).
            valid_mask = targets.seq_len_mask.bool()  # (B, M).
            if not valid_mask.any():
                losses[name] = logits.sum() * 0
                continue
            logits = logits.take_along_dim(indices.unsqueeze(-1), dim=1)  # (B, M, C).
            logits = logits[valid_mask]  # (N, C).
            target = target[valid_mask]  # (N).
            if spec.get("cast", False):
                target = target.long()
            losses[name] = torch.nn.functional.cross_entropy(logits, target)
            if spec.get("weight", 1) != 1:
                losses[name] = ScaleGradient.apply(losses[name], spec["weight"])
            with torch.no_grad():
                predictions = selected_logits_flat.argmax(-1)  # (N).
                metrics[f"batch-accuracy-{name}"] = (predictions == target).float().mean()

        return losses, metrics

    def predict(self, outputs):
        orig_lengths = outputs.seq_lens
        outputs, lengths = self._split_outputs(outputs)  # (B, L, D).
        last = (lengths - 1).clip(min=0)[:, None, None]  # (B, 1, 1).
        result = {}
        for name in sorted(set(self._targets) & set(outputs)):
            logits = outputs[name].take_along_dim(last, 1).squeeze(1)
            result[f"logits-{name}"] = logits  # (B, C).
            result[name] = logits.argmax(-1)  # (B).

        seq_names = set()
        if self._local_targets:
            for name in sorted(set(self._local_targets) & set(outputs)):
                logits = outputs[name]  # (B, L, D).
                predictions = logits.argmax(-1)  # (B, L).
                result[name] = predictions
                seq_names.add(name)

        return PaddedBatch(result, orig_lengths, seq_names=seq_names)

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
            if name in self._targets:
                nc = self._targets[name]["num_classes"]
            else:
                nc = self._local_targets[name]["num_classes"]
            result[name] = outputs[..., offset:offset + nc]
            offset += nc
        if offset != outputs.shape[-1]:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result, lengths
