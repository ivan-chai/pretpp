import torch
from hotpp.nn import Encoder

from functools import partial


class StatsEncoder(Encoder):
    def __init__(self, embedder, model_partial, stats_prob=0.5, *args, **kwargs):
        model_partial = partial(model_partial, stats_dim=self.get_stats_dim(embedder), n_stats=self.get_n_stats())
        super().__init__(embedder, model_partial, *args, **kwargs)
        self.stats_prob = stats_prob

    def embed(self, x):
        """Extract embeddings with shape (B, D)."""
        if not hasattr(self.model, "embed"):
            raise NotImplementedError("The model doesn't support embeddings extraction.")
        times = (self.compute_time_deltas(x) if self.model.delta_time else x)[self._timestamps_field]  # (B, L).
        stats = self.extract_stats(x)
        x = self.apply_embedder(x)
        return self.model.embed(x, times, stats=stats)  # (B, D).

    def forward(self, x, return_states=False):
        """Apply the encoder network.

        Args:
            x: PaddedBatch with input features.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Outputs is with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D).
        """
        times = (self.compute_time_deltas(x) if self.model.delta_time else x)[self._timestamps_field]  # (B, L).
        stats = self.extract_stats(x) if (not self.training) or (torch.rand([]) < self.stats_prob) else None
        embeddings = self.apply_embedder(x)
        outputs, states = self.model(embeddings, times, return_states=return_states, stats=stats)   # (B, L, D), (N, B, L, D).
        return outputs, states

    def extract_stats(self, x):
        n_labels = self.embedder.embeddings["labels"].num_embeddings
        mask = x.seq_len_mask.bool()  # (B, L).
        labels = x.payload["labels"].long()  # (B, L).
        amounts = x.payload["log_amount"].float().exp() - 1  # (B, L).
        masked_labels = labels.masked_fill(~mask, n_labels)
        encoded = torch.nn.functional.one_hot(masked_labels, n_labels + 1)  # (B, L, C + 1).
        counts = encoded.sum(1)[:, :n_labels].float()  # (B, C).
        counts_norm = counts / x.seq_lens.clip(min=1).unsqueeze(1)
        sums = (encoded * amounts.unsqueeze(2)).sum(1)[:, :n_labels]  # (B, C).
        stats = (torch.stack([counts, counts_norm, sums], 1) + 1).log()  # (B, 3, C).
        return stats

    def get_stats_dim(self, embedder):
        n_labels = embedder.embeddings["labels"].num_embeddings
        return n_labels

    def get_n_stats(self):
        return 3
