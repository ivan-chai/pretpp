import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class ColesLoss(BaseLoss):
    """Contrastive pretrainer.

    Args:
        coles_loss: A contrastive loss from pytorch-lifestream.
    """
    def __init__(self, embedding_dim, coles_loss, id_field="id",
                 n_splits=5, min_fraction=0.1, max_fraction=0.9):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.coles_loss = coles_loss
        self.id_field = id_field
        self.n_splits = n_splits
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    @property
    def input_size(self):
        return self.embedding_dim

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
        device = inputs.device
        b, l = inputs.shape
        n = self.n_splits
        sample_fractions = self.min_fraction + (self.max_fraction - self.min_fraction) * torch.rand(b, n, device=device)  # (B, N).
        sample_sizes = (sample_fractions * inputs.seq_lens[:, None].expand(b, n)).round().long()  # (B, N).
        sample_sizes = torch.minimum(sample_sizes.clip(min=1), inputs.seq_lens[:, None])  # (B, N).
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
