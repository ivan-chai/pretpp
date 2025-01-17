import torch


class MeanAggregator(torch.nn.Module):
    def forward(self, embeddings):
        embeddings, mask, lengths = embeddings.payload, embeddings.seq_len_mask.bool(), embeddings.seq_lens
        embeddings = embeddings.masked_fill(~mask, 0)  # (B, L, D).
        sums = embeddings.sum(1)  # (B, D).
        means = sums / lengths.unsqueeze(1).clip(min=1)
        return means  # (B, D).


class LastAggregator(torch.nn.Module):
    def forward(self, embeddings):
        embeddings, lengths = embeddings.payload, embeddings.seq_lens
        empty_mask = lengths == 0
        indices = (lengths - 1).clip(min=0)  # (B).
        last = embeddings.take_along_dim(indices[:, None, None], 1).squeeze(1)  # (B, D).
        last.masked_fill_(empty_mask.unsqueeze(1), 0)
        return last  # (B, D).
