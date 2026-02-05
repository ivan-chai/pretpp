import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class ColesLoss(BaseLoss):
    """Contrastive pretrainer.

    This class is a wrapper for a PytorchLifestream CoLES loss.
    We need to reimplement subsequence extractor for the fast batch computation.

    Args:
        coles_loss: A contrastive loss from pytorch-lifestream.
        min_length: The minimum number of subsequence events or minimum fraction if the value is less than 1.
        max_length: The maximum number of subsequence events or maximum fraction if the value is less than 1.
        cls_token: A dictionary with field values for a CLS token (optional, typically for transformer models).
    """
    def __init__(self, embedding_dim, coles_loss, id_field="id",
                 n_splits=5, min_length=0.1, max_length=0.9,
                 cls_token=None):
        if min_length > max_length:
            raise ValueError("Max length must be greater than min")
        super().__init__()
        self.embedding_dim = embedding_dim
        self.coles_loss = coles_loss
        self.id_field = id_field
        self.n_splits = n_splits
        self.min_length = min_length
        self.max_length = max_length
        self.cls_token = cls_token

    @property
    def input_size(self):
        return self.embedding_dim

    @property
    def aggregate(self):
        # Use aggregation if there is no special token.
        return self.cls_token is None

    def prepare_inference_batch(self, inputs):
        """Extract model inputs for inference.

        Args:
            inputs: Input events with shape (B, L, *).

        Returns:
            Model inputs with shape (B, L', *).
        """
        if self.cls_token is not None:
            # Add CLS token to the end of inputs and add fake targets.
            new_inputs = {k: v for k, v in inputs.payload.items() if k not in inputs.seq_names}
            last_indices = inputs.seq_lens  # (B).
            b = len(inputs)
            for k, t in self.cls_token.items():
                v = inputs.payload[k]  # (B, L, *).
                new_inputs[k] = torch.cat([v, v[:, -1:]], 1)  # (B, L + 1, *).
                token = torch.full_like(v[:, :1], t)  # (B, 1, *).
                last_indices_expanded = last_indices.reshape(*([b] + [1] * (v.ndim - 1)))  # (B, 1, ..., 1).
                last_indices_expanded = last_indices_expanded.expand(*([b, 1] + list(v.shape[2:])))  # (B, 1, *).
                new_inputs[k].scatter_(1, last_indices_expanded, token)
            inputs = PaddedBatch(new_inputs, inputs.seq_lens + 1,
                                 seq_names={k for k in inputs.seq_names if k in self.cls_token})
        return inputs

    def prepare_batch(self, inputs, targets=None):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets (unused): Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        device = inputs.device
        b, l = inputs.shape
        n = self.n_splits
        # Sample subsequence lengths.
        if self.min_length < 1:
            min_lengths = (inputs.seq_lens * self.min_length).round().long()  # (B).
        else:
            min_lengths = torch.minimum(torch.full_like(inputs.seq_lens, self.min_length),
                                        inputs.seq_lens)  # (B).
        if self.max_length < 1:
            max_lengths = (inputs.seq_lens * self.max_length).round().long()  # (B).
        else:
            max_lengths = torch.minimum(torch.full_like(inputs.seq_lens, self.max_length),
                                        inputs.seq_lens)  # (B).
        sample_sizes = min_lengths[:, None] + (max_lengths - min_lengths)[:, None] * torch.rand(b, n, device=device)  # (B, N).
        sample_sizes = sample_sizes.round().long()  # (B, N).
        assert (sample_sizes <= l).all()
        offsets = ((inputs.seq_lens[:, None] - sample_sizes) * torch.rand(b, n, device=device)).long()  # (B, N).

        new_l = sample_sizes.max().item()
        indices = (offsets[:, :, None] + torch.arange(new_l, device=device)[None, None]).clip(max=l - 1)  # (B, N, L').
        
        def uni_repeat_interleave(arr, repeats):
            if isinstance(arr, list):
                return [x for x in arr for _ in range(repeats)]
            else:
                return arr.repeat_interleave(repeats)
        new_inputs = {k: uni_repeat_interleave(v, n) for k, v in inputs.payload.items() if k not in inputs.seq_names}  # (BN).
        # Need: (B, N, L').
        for k in inputs.seq_names:
            v = inputs.payload[k]  # (B, L).
            slices = v[:, None, :].take_along_dim(indices, 2)  # (B, N, L').
            new_inputs[k] = slices.reshape(b * n, new_l)  # (BN, L').
        new_inputs = PaddedBatch(new_inputs, sample_sizes.flatten(), seq_names=inputs.seq_names)
        targets = new_inputs.payload[self.id_field]  # (BN).

        # Postprocess sequences.
        new_inputs = self.prepare_inference_batch(new_inputs)
        return new_inputs, targets

    def forward(self, outputs, targets):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model output embeddings with shape (B, 1, D).
            targets: Target indices with shape (B).

        Returns:
            Losses dict and metrics dict.
        """
        outputs, lengths = outputs.payload, outputs.seq_lens
        if self.cls_token is not None:
            # Extract CLS token embedding.
            last_indices = lengths - 1
            b = len(last_indices)
            outputs = outputs.take_along_dim(last_indices.reshape(*([b] + [1] * (outputs.ndim - 1))), 1).squeeze(1)
        else:
            # Input is an aggregated embedding.
            if outputs.shape[1] != 1:
                raise NotImplementedError("Expected aggregated embedding with shape (B, 1, C).")
            outputs = outputs.squeeze(1)
            
        def name_to_number(names, device=None):
            name_dict = {}
            n_unique = 0
            result = []
            for name in names:
                if name_dict.get(name, 0) == 0:
                    name_dict[name] = n_unique
                    n_unique += 1
                result.append(name_dict[name])
            return torch.tensor(result, device=device)
        
        if isinstance(targets[0], str):
            targets = name_to_number(targets, device=outputs.device)

        loss = self.coles_loss(outputs, targets)
        losses = {"coles": loss}
        metrics = {}
        return losses, metrics
