import torch
from hotpp.data import PaddedBatch
from hotpp.nn import SimpleTransformer


def insert_tokens(embeddings, timestamps, positions, token):
    """Insert token at specified positions.

    Args:
        embeddings: (B, L, D).
        timestamps: (B, L).
        positions: (B, R) in the range [0, L + R).
        token: (D).
    Returns:
        - A modified input tensor with shape (B, L + R, D).
        - Modified timestamps with shape (B, L + R).
        - Source data indices with shape (B, L).
    """
    b, l, d = embeddings.shape
    r = positions.shape[1]
    device = embeddings.device

    # Compose indices.
    if not (positions[:, 1:] > positions[:, :-1]).all():
        raise ValueError("Positions must be unique and sorted.")
    insert_mask = torch.ones(b, l + r, device=device, dtype=torch.long)  # (B, L + R).
    insert_mask.scatter_(1, positions, 0)  # (B, L + R).
    last_input = (insert_mask.cumsum(1) - 1).clip(min=0)  # (B, L + R).
    index = last_input.scatter(1, positions, l)

    # Insert.
    extended = torch.cat([embeddings, token[None, None].expand(b, 1, d)], 1)  # (B, L + 1, D).
    new_embeddings = extended.take_along_dim(index.unsqueeze(2), 1)  # (B, L + R, D).
    extended = torch.cat([timestamps, timestamps[:, :1]], 1)  # (B, L + 1)
    new_timestamps = extended.take_along_dim(index, 1)  # (B, L + R).
    prev_input = last_input.take_along_dim(positions, 1)  # (B, R).
    prev_ts = timestamps.take_along_dim(prev_input, 1)  # (B, R).
    new_timestamps.scatter_(1, positions, prev_ts)

    # Get source indices and return.
    source_indices = insert_mask.nonzero()[:, 1].reshape(b, l)  # (B, L).
    return new_embeddings, new_timestamps, source_indices


def sample_mask(positions, l):
    """Generate attention mask for history tokens at specified locations.

    Args:
        positions: History tokens positions with shape (R) in the range [0, L + R).
        l: The original sequence length.

    Returns:
        Attention mask with shape (L + R, L + R). Ones indicate excluded tokens.
    """
    r = len(positions)
    device = positions.device

    # Skip events before the last history token.
    insert_mask = torch.zeros(l + r, device=device, dtype=torch.bool)  # (L + R).
    insert_mask.scatter_(0, positions, 1)  # (L + R).
    mask = insert_mask[None].expand(l + r, l + r)  # (L + R, L + R).
    mask = torch.tril(mask, diagonal=-1)
    mask = mask.fliplr().cumsum(1).fliplr() > insert_mask  # (L + R, L + R).

    # Allow history tokens to access full history.
    mask[insert_mask] = 0
    return mask


class HistoryTokenTransformer(SimpleTransformer):
    """An extension of the transformer model with extra <history-tokens> for context aggregation.

    Args:
        history_token_fraction: The fraction of batches to apply history token to.
        n_history_tokens: The number of tokens inserted in each sequence (including padding).
        mode: Either `pretrain`, `supervised-append` or `supervised-replace`.
    """
    def __init__(self, input_size, history_token_fraction=1, n_history_tokens=1, mode="pretrain", **kwargs):
        if mode not in {"pretrain", "supervised-append", "supervised-replace"}:
            raise ValueError(f"Unknown mode: {mode}")
        super().__init__(input_size, **kwargs)
        if not self.causal:
            raise NotImplementedError("A history-token transformer must be causal.")
        self.history_token = torch.nn.Parameter(torch.rand(self.n_embd))  # (D).
        self.history_token_fraction = history_token_fraction
        self.n_history_tokens = n_history_tokens
        self.mode = mode

    def _add_token_to_the_end(self, payload, timestamps, seq_lens, append=True):
        b, l, d = payload.shape
        if append:
            payload = torch.cat([payload, payload[:, :1]], 1)  # (B, L + 1, D).
            timestamps = torch.cat([timestamps, timestamps[:, :1]], 1)  # (B, L + 1).
            last = seq_lens  # (B).
        else:
            last = (seq_lens - 1).clip(min=0)  # (B).
        payload.scatter_(1, last[:, None, None].expand(b, 1, d), self.history_token.to(payload.dtype)[None, None].expand(b, 1, d))

        last_ts = timestamps.take_along_dim((seq_lens[:, None] - 1).clip(min=0), 1)  # (B, 1).
        timestamps.scatter_(1, last[:, None], last_ts)
        return payload, timestamps, last

    def embed(self, x, timestamps):
        payload = self.input_projection(x.payload)  # (B, L, D).

        # Append history token to the end.
        payload, timestamps, last = self._add_token_to_the_end(payload, timestamps.payload, x.seq_lens)

        # Extract history token embedding.
        payload = self.positional(payload, timestamps)  # (B, L + 1, D).
        outputs, _ = self.transform(PaddedBatch(payload, x.seq_lens + 1))
        return outputs.payload.take_along_dim(last[:, None, None], 1).squeeze(1)  # (B, D).

    def forward(self, x, timestamps, states=None, return_states=False):
        if self.mode in {"supervised-append", "supervised-replace"}:
            if return_states:
                raise NotImplementedError("HistoryTokenTransformer doesn't support states return.")
            append = self.mode == "supervised-append"
            payload = self.input_projection(x.payload)  # (B, L, D).
            payload, timestamps, last = self._add_token_to_the_end(payload, timestamps.payload, x.seq_lens, append=append)
            payload = self.positional(payload, timestamps)  # (B, L, D).
            outputs, _ = self.transform(PaddedBatch(payload, x.seq_lens + int(append)))
            token_branch = 0 * self.history_token.mean()
            outputs = PaddedBatch(outputs.payload + token_branch, outputs.seq_lens)
            states = None
            return outputs, states
        if not self.training:
            # Don't insert history tokens.
            return super().forward(x, timestamps, states=states, return_states=return_states)
        if torch.rand([]) > self.history_token_fraction:
            # Don't insert history tokens.
            outputs, states = super().forward(x, timestamps, states=states, return_states=return_states)
            token_branch = 0 * self.history_token.mean()
            outputs = PaddedBatch(outputs.payload + token_branch, outputs.seq_lens)
            states = states + token_branch if states is not None else None
            return outputs, states
        if return_states:
            raise NotImplementedError("HistoryTokenTransformer doesn't support states return.")
        b, l = x.shape
        device = x.device
        payload = self.input_projection(x.payload)  # (B, L, D).
        timestamps = timestamps.payload  # (B, L).

        # Insert history tokens.
        max_length = x.seq_lens.max()
        positions = torch.randperm(max_length, device=device)[:self.n_history_tokens].sort()[0]  # (R), sorted.
        r = len(positions)
        payload, timestamps, indices = insert_tokens(payload, timestamps, positions[None].expand(b, r), self.history_token)  # (B, L + R, D), (B, L + R), (B, L).

        # Update attention mask.
        history_mask = sample_mask(positions, l)  # (L + R, L + R).
        mask = history_mask if self.sa_mask is None else torch.logical_or(self.sa_mask[:l + r, :l + r], history_mask)

        # Apply transformer.
        payload = self.positional(payload, timestamps)  # (B, L + R, D).
        assert self.causal
        # src_key_padding_mask is optional for causal transformers.
        outputs = self.encoder(payload,
                               mask=mask,
                               is_causal=self.causal)  # (B, L + R, D).

        # Remove history tokens and return.
        outputs = PaddedBatch(outputs.take_along_dim(indices.unsqueeze(2), 1), x.seq_lens)  # (B, L, D).
        states = None
        return outputs, states
