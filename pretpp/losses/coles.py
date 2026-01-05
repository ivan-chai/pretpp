import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


CLS_POS_NONE = "none"
CLS_POS_BEGIN = "begin"
CLS_POS_END = "end"


class ColesLoss(BaseLoss):
    """Contrastive pretrainer.

    This class is a wrapper for a PytorchLifestream CoLES loss.
    We need to reimplement subsequence extractor for the fast batch computation.

    Args:
        coles_loss: A contrastive loss from pytorch-lifestream.
        min_length: The minimum number of subsequence events or minimum fraction if the value is less than 1.
        max_length: The maximum number of subsequence events or maximum fraction if the value is less than 1.
        cls_token: A dictionary with field values for a CLS token (optional, typically for transformer models).
        cls_token_begin: Whether to put CLS at the beginning or at the end.
    """
    def __init__(self, embedding_dim, coles_loss, id_field="id",
                 n_splits=5, min_length=0.1, max_length=0.9,
                 cls_token=None, cls_token_begin=False, normalize=True):
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
        self.cls_token_begin = cls_token_begin
        self.normalize = normalize

    @property
    def input_size(self):
        return self.embedding_dim

    @property
    def aggregate(self):
        # Use aggregation if there is no special token.
        return self.cls_token is None

    @property
    def cls_token_pos(self):
        if self.cls_token is None:
            return CLS_POS_NONE
        elif self.cls_token_begin:
            return CLS_POS_BEGIN
        else:
            return CLS_POS_END

    def prepare_inference_batch(self, inputs):
        """Extract model inputs for inference.

        Args:
            inputs: Input events with shape (B, L, *).

        Returns:
            Model inputs with shape (B, L', *).
        """
        if self.cls_token is not None:
            new_inputs = {k: v for k, v in inputs.payload.items() if k not in inputs.seq_names}
            if self.cls_token_begin:
                # Add CLS token at the beginning.
                for k, t in self.cls_token.items():
                    v = inputs.payload[k]  # (B, L, *).
                    new_inputs[k] = torch.cat([v[:, :1], v], 1)  # (B, 1 + L, *).
                    new_inputs[k][:, 0] = t
            else:
                # Add CLS token to the end of inputs and add fake targets.
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

        new_inputs = {k: v.repeat_interleave(n) for k, v in inputs.payload.items() if k not in inputs.seq_names}  # (BN).
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
            if self.cls_token_begin:
                # Extract CLS token embedding from the beginning.
                outputs = outputs[:, 0]  # (B, D).
            else:
                # Extract CLS token embedding from the end.
                last_indices = lengths - 1
                b = len(last_indices)
                outputs = outputs.take_along_dim(last_indices.reshape(*([b] + [1] * (outputs.ndim - 1))), 1).squeeze(1)
        else:
            # Input is an aggregated embedding.
            if outputs.shape[1] != 1:
                raise NotImplementedError("Expected aggregated embedding with shape (B, 1, C).")
            outputs = outputs.squeeze(1)
        if self.normalize:
            outputs = torch.nn.functional.normalize(outputs, dim=-1)
        loss = self.coles_loss(outputs, targets)
        losses = {"coles": loss}
        metrics = {}
        return losses, metrics
