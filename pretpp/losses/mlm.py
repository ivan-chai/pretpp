import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


EVAL_MASK_FIELD = "_evaluation_mask"


class MLMLoss(BaseLoss):
    """Masked token prediction.

    Args:
        losses: Mapping from the feature name to the loss function.
        mask_token: A dictionary of values for a masked token.
        eval_fraction: The fraction of elements selected for loss evaluation.
        mask_prob: The probability of token masking.
        random_prob: The probability of token replacement with random token from the same batch.
        timedeltas_field: The field to put time deltas in after masking.
        field_mask_probs: Mapping from a field name to the masking probability. By default all fields are masked.
    """
    def __init__(self, losses, mask_token,
                 timestamps_field="timestamps", timedeltas_field=None,
                 eval_fraction=0.15, mask_prob=0.8, random_prob=0.1,
                 field_mask_probs=None):
        super().__init__()
        self._losses = torch.nn.ModuleDict(losses)
        self._order = list(sorted(losses))

        self._mask_token = mask_token
        self._timestamps_field = timestamps_field
        self._timedeltas_field = timedeltas_field
        self._eval_fraction = eval_fraction
        self._field_mask_probs = field_mask_probs

        if mask_prob + random_prob > 1:
            raise ValueError("Probabilities sum can't exceed 1")
        unchanged_prob = 1 - mask_prob - random_prob
        self.register_buffer("_augment_type_probs", torch.tensor([unchanged_prob, mask_prob, random_prob], dtype=torch.float))

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
        # Add deltas if necessary.
        # TODO: max_delta and smoothing.
        if self._timedeltas_field:
            timestamps = inputs.payload[self._timestamps_field]  # (B, L).
            deltas = timestamps.clone()
            deltas[:, 0].fill_(0)
            deltas[:, 1:] -= timestamps[:, :-1]
            inputs = PaddedBatch(inputs.payload | {self._timedeltas_field: deltas}, inputs.seq_lens,
                                 seq_names=set(inputs.seq_names) | {self._timedeltas_field})

        # Add augmentations.
        b, l = inputs.shape
        mask = inputs.seq_len_mask
        eval_mask = torch.rand(b, l, device=inputs.device) < self._eval_fraction  # (B, L).
        eval_mask = torch.logical_and(eval_mask, mask)
        truncated_lengths = (inputs.seq_lens - 2).clip(min=0)
        # Need at least one element for loss computation after applying offsets before and during loss computation.
        if (eval_mask[:, 1:-1].sum() == 0) and (truncated_lengths.sum() > 0):
            i = torch.distributions.Categorical(truncated_lengths / truncated_lengths.sum()).sample([1]).argmax()
            j = torch.randint(1, inputs.seq_lens[i] - 1, [])
            eval_mask[i, j] = True
        augment_type_distribution = torch.distributions.Multinomial(probs=self._augment_type_probs)
        augment_type = augment_type_distribution.sample([b, l]).argmax(-1)  # (B, L) tensor of integer values 0, 1, and 2.
        augment_type.masked_fill_(~eval_mask, 0)
        masking_mask = augment_type == 1  # (B, L).
        random_mask = augment_type == 2  # (B, L).
        do_masking = bool(masking_mask.any())
        do_sampling = bool(random_mask.any())

        all_indices = torch.arange(b * l, device=inputs.device).reshape(b, l)[mask]  # (V).
        random_indices = torch.randint(len(all_indices), (random_mask.sum().item(),), device=inputs.device)
        alt_indices = all_indices.take_along_dim(random_indices, 0)  # (R).

        model_inputs = {}
        targets = {}
        for k, v in inputs.payload.items():
            if k in inputs.seq_names:
                # Augment.
                a = v.clone()
                if do_sampling:
                    if a.ndim != 2:
                        raise NotImplementedError("Expected scalar features")
                    alt_values = v.flatten(0, 1).take_along_dim(alt_indices, 0)  # (R).
                    a.masked_scatter_(random_mask, alt_values)
                if do_masking:
                    a.masked_fill_(masking_mask, self._mask_token[k])
                # We need to make reverse offset for compatibility with base losses, that apply forward offset.
                model_inputs[k] = a[:, 1:]
                targets[k] = v[:, :-1]
            else:
                model_inputs[k] = v
        if self._field_mask_probs is not None:
            for field in inputs.seq_names:
                if field not in self._field_mask_probs:
                    raise ValueError(f"No masking probability for the field {field}")
                prob = self._field_mask_probs[field]
                mask = torch.rand(b, max(l, 1) - 1, device=inputs.device) < prob  # (B, L).
                model_inputs[field] = torch.where(mask, model_inputs[field], inputs.payload[field][:, 1:])
        targets[EVAL_MASK_FIELD] = eval_mask[:, :-1]
        lengths = (inputs.seq_lens - 1).clip(min=0)
        model_inputs = PaddedBatch(model_inputs, lengths, seq_names=inputs.seq_names)
        targets = PaddedBatch(targets, lengths, seq_names=inputs.seq_names)
        return model_inputs, targets

    def forward(self, outputs, targets):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model outputs with shape (B, L, *, D) or (B, 1, *, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L, *).

        Returns:
            Losses dict and metrics dict.
        """
        # Compute losses. It is assumed that predictions lengths are equal to targets lengths.
        if not isinstance(outputs, dict):
            outputs = self._split_outputs(outputs.payload)
        mask = targets.seq_len_mask.bool() if (targets.seq_lens != targets.shape[1]).any() else None
        losses = {}
        metrics = {}
        for name in set(targets.payload) & set(outputs):
            loss, loss_metrics = self._losses[name](targets.payload[name], outputs[name], mask, reduction="none")
            losses[name] = loss[targets.payload[EVAL_MASK_FIELD][:, 1:]].mean()
            for k, v in loss_metrics.items():
                metrics[f"{name}-{k}"] = v
        return losses, metrics

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
