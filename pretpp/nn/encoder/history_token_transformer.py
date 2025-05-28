import torch
from hotpp.data import PaddedBatch
from hotpp.nn import SimpleTransformer


def insert_tokens(embeddings, timestamps, token):
    """Insert the specified token before each input token.

    Args:
        embeddings: (B, L, D).
        timestamps: (B, L).
        token: (D).
    Returns:
        - A modified input tensor with shape (B, 2 * L, D).
        - Modified timestamps with shape (B, 2 * L).
        - Source data indices with shape (B, L).
    """
    b, l, d = embeddings.shape
    device = embeddings.device
    new_embeddings = torch.stack([embeddings, token[None, None].expand(b, l, d)], 2).flatten(1, 2)  # (B, 2 * L, D).
    new_timestamps = timestamps.repeat_interleave(2, 1)  # (B, 2 * L).
    return new_embeddings, new_timestamps


def remove_tokens(embeddings):
    b, l, d = embeddings.shape
    if l % 2 != 0:
        raise ValueError("Unexpected input shape")
    return embeddings[:, ::2]


def make_mask(n_summarize):
    """Generate attention mask for history tokens with the required number
    of summarized tokens for each element of the batch.

    Args:
        n_summarize: The number of tokens to summarize in the range [0, L). with shape (L).

    Returns:
        Attention mask with shape (2 * L, 2 * L) with True values at masked positions.
    """
    device = n_summarize.device
    l = len(n_summarize)
    mask = torch.zeros(l, 4 * l, device=device, dtype=torch.bool)  # (L, 4 * L).

    n_summarize2 = n_summarize * 2  # (L).
    mask = torch.arange(2 * l, device=device)[None] < n_summarize2[:, None]  # (L, 2 * L).
    mask[:, 1::2] = True
    mask.scatter_(1, (n_summarize2 - 1).clip(min=0).unsqueeze(1), False)
    mask = torch.cat([mask, torch.zeros_like(mask)], 1).reshape(2 * l, 2 * l)
    return mask


def sample_mask(l, locality=0, device=None):
    """Generate attention mask for history tokens at specified locations.

    Args:
        l: The original sequence length.
        device: Target device.
        locality: The value between 0 and 1 with 0 meaning uniform history token selection
            and 1 for using the last available token.

    Returns:
        Attention mask with shape (2 * L, 2 * L). Ones indicate excluded tokens.
    """
    if device is None:
        device = torch.get_default_device()
    n_summarize = (torch.rand(l, device=device) * torch.arange(l, device=device)).round().long()  # (L).
    if locality > 0:
        prefix_size = torch.arange(l, device=device)
        n_summarize = torch.where(torch.rand(l, device=device) < locality, prefix_size, n_summarize)  # (L).
    return make_mask(n_summarize)


class HistoryTokenTransformer(SimpleTransformer):
    """An extension of the transformer model with extra <history-tokens> (HT) for context aggregation.

    Args:
        history_token_locality: The value between 0 and 1 with 0 meaning uniform history token selection
            and 1 for using the last available token.
        embed_layer: The layer to extract HT embeddings from.
    """
    def __init__(self, input_size, history_token_locality=0, embed_layer=None, **kwargs):
        super().__init__(input_size, **kwargs)
        if not self.causal:
            raise NotImplementedError("A history-token transformer must be causal.")
        self.history_token = torch.nn.Parameter(torch.rand(self.n_embd))  # (D).
        self.history_token_locality = history_token_locality
        self.embed_layer = embed_layer

    def _add_token_to_the_end(self, payload, timestamps, seq_lens):
        b, l, d = payload.shape
        last = seq_lens  # (B).

        # Add HT to to the end.
        payload = torch.cat([payload, payload[:, :1]], 1)  # (B, L + 1, D).
        payload.scatter_(1, last[:, None, None].expand(b, 1, d), self.history_token.to(payload.dtype)[None, None].expand(b, 1, d))

        # Duplicate the last timestamp.
        last_ts = timestamps.take_along_dim((seq_lens[:, None] - 1).clip(min=0), 1)  # (B, 1).
        timestamps = torch.cat([timestamps, timestamps[:, :1]], 1)  # (B, L + 1).
        timestamps.scatter_(1, last[:, None], last_ts)
        return payload, timestamps, last

    def embed(self, x, timestamps):
        payload = self.input_projection(x.payload)  # (B, L, D).

        # Append history token to the end.
        payload, timestamps, last = self._add_token_to_the_end(payload, timestamps.payload, x.seq_lens)
        new_lengths = x.seq_lens + 1

        # Extract history token embedding.
        payload = self.positional(payload, timestamps)  # (B, L + 1, D).
        if self.embed_layer is not None:
            _, states = self.transform(PaddedBatch(payload, new_lengths), return_states="full")  # N * (B, L, D).
            outputs = states[self.embed_layer]
            layer = self.encoder.layers[0]
            is_last_layer = self.embed_layer == len(states) - 1
            if layer.norm_first and (not is_last_layer):
                outputs = layer.norm1(outputs)
            outputs = PaddedBatch(outputs, new_lengths)
        else:
            outputs, _ = self.transform(PaddedBatch(payload, new_lengths))
        embeddings = outputs.payload.take_along_dim(last[:, None, None], 1).squeeze(1)  # (B, D).
        return embeddings

    def forward(self, x, timestamps, states=None, return_states=False):
        if not self.training:
            # Don't insert history tokens.
            return super().forward(x, timestamps, states=states, return_states=return_states)
        if return_states:
            raise NotImplementedError("HistoryTokenTransformer doesn't support states return.")
        b, l = x.shape
        device = x.device
        payload = self.input_projection(x.payload)  # (B, L, D).
        timestamps = timestamps.payload  # (B, L).

        # Insert history token before each real token.
        payload, timestamps = insert_tokens(payload, timestamps, self.history_token)  # (B, 2 * L, D), (B, 2 * L).

        # Update attention mask.
        history_mask = sample_mask(l, locality=self.history_token_locality, device=device)  # (2 * L, 2 * L).
        mask = history_mask if self.sa_mask is None else torch.logical_or(self.sa_mask[:2 * l, :2 * l], history_mask)

        # Apply transformer.
        payload = self.positional(payload, timestamps)  # (B, 2 * L, D).
        assert self.causal
        # src_key_padding_mask is optional for causal transformers.
        outputs = self.encoder(payload,
                               mask=mask,
                               is_causal=self.causal)  # (B, 2 * L, D).

        # Remove history tokens and return.
        outputs = PaddedBatch(remove_tokens(outputs), x.seq_lens)  # (B, L, D).
        states = None
        return outputs, states
