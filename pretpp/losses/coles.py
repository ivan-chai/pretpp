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
    """
    def __init__(self, embedding_dim, coles_loss, id_field="id",
                 n_splits=5, min_length=0.1, max_length=0.9):
        if min_length > max_length:
            raise ValueError("Max length must be greater than min")
        super().__init__()
        self.embedding_dim = embedding_dim
        self.coles_loss = coles_loss
        self.id_field = id_field
        self.n_splits = n_splits
        self.min_length = min_length
        self.max_length = max_length

    @property
    def input_size(self):
        return self.embedding_dim

    @property
    def aggregate(self):
        return True

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
        return new_inputs, targets

    def forward(self, outputs, targets):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model output embeddings with shape (B, 1, D).
            targets: Target indices with shape (B).

        Returns:
            Losses dict and metrics dict.
        """
        # Input is an aggregated embedding.
        loss = self.coles_loss(outputs.payload.squeeze(1), targets)
        losses = {"coles": loss}
        metrics = {}
        return losses, metrics
