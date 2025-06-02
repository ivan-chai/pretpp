from abc import ABC, abstractmethod

import torch
from hotpp.data import PaddedBatch


def add_token_to_the_end(x, timestamps, token):
    b, l, d = x.payload.shape
    last = x.seq_lens  # (B).
    new_lengths = x.seq_lens + 1

    # Add HT to to the end.
    new_x = torch.cat([x.payload, x.payload[:, :1]], 1)  # (B, L + 1, D).
    new_x.scatter_(1, last[:, None, None].expand(b, 1, d), token[None, None].expand(b, 1, d))
    new_x = PaddedBatch(new_x, new_lengths)

    # Duplicate the last timestamp.
    last_ts = timestamps.payload.take_along_dim((timestamps.seq_lens[:, None] - 1).clip(min=0), 1)  # (B, 1).
    new_timestamps = torch.cat([timestamps.payload, timestamps.payload[:, :1]], 1)  # (B, L + 1).
    new_timestamps.scatter_(1, last[:, None], last_ts)
    new_timestamps = PaddedBatch(new_timestamps, new_lengths)
    return new_x, new_timestamps


class HTStrategyBase(ABC, torch.nn.Module):
    """History token strategy."""

    def __init__(self, n_embd):
        super().__init__()
        self.init_token(n_embd)

    def init_token(self, n_embd):
        self.token = torch.nn.Parameter(torch.rand(n_embd))  # (D).

    @abstractmethod
    def select(self, timestamps, embedding=False):
        """Select token positions based on lengths. Use internal state for storage."""
        pass

    @abstractmethod
    def clear_state(self):
        pass

    @abstractmethod
    def insert_tokens(self, x, timestamps):
        """Insert special tokens.

        NOTE. The method affects internal state keeping inserted tokens positions.

        Args:
            x: (B, L, D).
            timestamps: (B, L).
        Returns:
            - A modified input tensor with shape (B, L', D).
            - Modified timestamps with shape (B, L').
        """
        pass

    @abstractmethod
    def make_attention_mask(self):
        """Sample an attention mask for the last input, transformed with `insert_tokens`.

        Returns:
            Attention mask with shape (L', L').
        """
        pass

    def apply(self, x, timestamps):
        """Joined insert_tokens and make_attention_mask method."""
        new_x, new_timestamps = self.insert_tokens(x, timestamps)
        attention_mask = self.make_attention_mask()
        return new_x, new_timestamps, attention_mask

    def forward(self, x, timestamps):
        return self.apply(x, timestamps)

    @abstractmethod
    def extract_outputs(self, x):
        """Remove history tokens from output.

        Args:
            x: (B, L', D).

        Returns:
            PaddedBatch with shape (B, L, D) or (B, D) for embedding mode.
        """
        pass

    def __call__(self, timestamps, embedding=False):
        """Create context for a specific input.

        Args:
            timestamps: PaddedBatch with timestamps (B, L).
            embedding: Whether to use embedding or pretrain strategy.

        Returns:
            Context for a specific input.
        """
        self.select(timestamps, embedding)
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.clear_state()
        if value:
            raise


class FullHTStrategy(HTStrategyBase):
    """Insert history token before each real token.

    Args:
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        apply_probability: The probability of HT usage for each real token.
    """
    def __init__(self, n_embd, apply_probability=0.5, predict="input_tokens"):
        if predict not in {"input_tokens", "history_tokens", "all"}:
            raise ValueError(f"Unknown prediction mode: {predict}")
        super().__init__(n_embd)
        self.apply_probability = apply_probability
        self.predict = predict

    def select(self, timestamps, embedding=False):
        self.seq_lens = timestamps.seq_lens
        self.length = timestamps.shape[1]
        self.embedding = embedding
        self.device = timestamps.device

    def clear_state(self):
        del self.seq_lens
        del self.length
        del self.embedding
        del self.device

    def insert_tokens(self, x, timestamps):
        if self.embedding:
            return add_token_to_the_end(x, timestamps, self.token.to(x.payload.dtype))
        else:
            # Insert before each real token.
            b, l, d = x.payload.shape
            device = x.device
            new_x = torch.stack([x.payload, self.token[None, None].expand(b, l, d)], 2).flatten(1, 2)  # (B, 2 * L, D).
            new_timestamps = timestamps.payload.repeat_interleave(2, 1)  # (B, 2 * L).
            new_lengths = x.seq_lens * 2
        return PaddedBatch(new_x, new_lengths), PaddedBatch(new_timestamps, new_lengths)

    @staticmethod
    def _make_attention_mask_impl(n_summarize):
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

        # Make real tokens masks.
        n_summarize2 = n_summarize * 2  # (L).
        mask = torch.arange(2 * l, device=device)[None] < n_summarize2[:, None]  # (L, 2 * L).
        mask[:, 1::2] = True
        mask.scatter_(1, (n_summarize2 - 1).clip(min=0).unsqueeze(1), False)

        # Make history tokens masks. Disable attention between history tokens.
        ht_mask = torch.zeros_like(mask)
        ht_mask[:, 1::2] = True
        ht_mask[:, 1::2].fill_diagonal_(False)

        # Join masks.
        mask = torch.cat([mask, ht_mask], 1).reshape(2 * l, 2 * l)
        return mask

    def make_attention_mask(self):
        if self.embedding:
            return None  # Simple causal mask.
        else:
            n_summarize = (torch.rand(self.length, device=self.device) * torch.arange(self.length, device=self.device)).round().long()  # (L).
            if torch.rand([]) > self.apply_probability:
                n_summarize.fill_(0)
            return self._make_attention_mask_impl(n_summarize)

    def extract_outputs(self, x):
        if self.embedding:
            return x.payload.take_along_dim(self.seq_lens[:, None, None], 1).squeeze(1)  # (B, D).
        else:
            b, l = x.shape
            if l % 2 != 0:
                raise ValueError("Unexpected input shape")
            if self.predict == "input_tokens":
                new_x = x.payload[:, ::2]
                new_lengths = self.seq_lens
            elif self.predict == "history_tokens":
                new_x = x.payload[:, 1::2]
                new_lengths = self.seq_lens
            else:
                assert self.predict == "all"
                new_x = x
                new_lengths = x.seq_lens
            return PaddedBatch(new_x, new_lengths)
