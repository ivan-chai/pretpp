from abc import ABC, abstractmethod

import torch
from hotpp.data import PaddedBatch
from hotpp.nn import LastAggregator, MeanAggregator


def batch_nonzero(x):
    """Find indices of non-zero elements along second dimension.

    NOTE: the number of non-zero elements must match.

    Args:
        x: Tensor with shape (B, D).

    Returns:
        Indices with shape (B, R).
    """
    nonzero = (x > 0).long()
    n_nonzero = nonzero.sum(1)  # (B).
    if (n_nonzero != n_nonzero[0]).any():
        raise ValueError(f"Different number of non-zero elements in rows: {n_nonzero.tolist()}.")
    order = torch.argsort(nonzero, dim=1, descending=True, stable=True)
    return order[:, :n_nonzero[0]]


def batch_randperm(b, r, device=None):
    return torch.rand(b, r, device=device).argsort(1)


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


def make_ht_attention_mask(length, ht_positions,
                           active_tokens="random", use_ht_mask=None,
                           device=None):
    """Make attention mask for history tokens.

    Args:
        length: The length of input sequence.
        ht_positions: Sorted token indices to insert HT after with shape (B, R).
        active_tokens: Either `random`, `last`, `none` or a tensor with shape (B, L) of HT token indices
            with 0 meaning no HT and R meaning the last HT.
        use_ht_mask: Indicates elements of the batch to use history tokens for.

    Returns:
        Atteniton mask with shape (B, L + R, L + R).
    """
    # Prepare.
    if device is None:
        device = torch.get_default_device()
    l = length
    b, r = ht_positions.shape
    assert r <= l
    ht_positions_mask = torch.zeros(b, l, device=device, dtype=torch.bool).scatter_(1, ht_positions, True)  # (B, L).

    # Generate prefix lengths.
    if active_tokens in {"random", "last"}:
        n_active_tokens = torch.cat([torch.zeros_like(ht_positions_mask[:, :1]), ht_positions_mask[:, :-1]], 1).cumsum(1)  # (B, L).
        if active_tokens == "random":
            active_tokens = (torch.rand(b, l, device=device) * n_active_tokens).round().long()  # (B, L) in [0, R].
        else:
            assert active_tokens == "last"
            active_tokens = n_active_tokens
    elif active_tokens == "none":
        active_tokens = torch.zeros(b, l, device=device, dtype=torch.long)
    assert active_tokens.shape == (b, l)
    if use_ht_mask is not None:
        active_tokens[~use_ht_mask.bool()] = 0

    n_summarize = torch.cat([torch.zeros_like(ht_positions[:, :1]), ht_positions + 1], 1).take_along_dim(active_tokens, 1)  # (B, L)
    selected_tokens = active_tokens - 1  # (B, L).

    # Fill grouped attention mask, with separated real and history tokens blocks:
    # [LL LR]
    # [RL RR]
    # The mask will be sorted after filling.

    mask_ll = (torch.arange(l, device=device)[None, None] < n_summarize[:, :, None]).expand(b, l, l)  # (B, L, L).
    mask_rr = (~torch.eye(r, device=device, dtype=torch.bool))[None].expand(b, r, r)  # (B, R, R).
    mask_lr = torch.ones(b, l, r, device=device, dtype=torch.bool).scatter_(2, selected_tokens.clip(min=0)[:, :, None], False)  # (B, L, R).
    mask_lr = mask_lr.reshape(b * l, r)
    mask_lr[selected_tokens.flatten() == -1, 0] = True
    mask_lr = mask_lr.reshape(b, l, r)
    mask_rl = torch.zeros(r, l, device=device, dtype=torch.bool)[None].expand(b, r, l)  # (B, R, L).

    # Merge and order masks.
    ht_mask = torch.zeros(b, l + r, device=device, dtype=torch.bool).scatter_(1, torch.arange(1, 1 + r, device=device) + ht_positions, True)  # (B, L + R).
    real_mask = ~ht_mask  # (B, L + R).
    order = torch.argsort(torch.cat([batch_nonzero(real_mask), batch_nonzero(ht_mask)], 1), dim=1)  # (B, L + R).

    mask = torch.cat([
        torch.cat([mask_ll, mask_lr], 2),
        torch.cat([mask_rl, mask_rr], 2)
    ], 1)  # (B, L + R, L + R).
    mask = mask.take_along_dim(order[:, :, None], 1).take_along_dim(order[:, None, :], 2)
    return mask


class HTStrategyBase(ABC, torch.nn.Module):
    """History token strategy."""

    def __init__(self, n_embd):
        super().__init__()
        self.init_token(n_embd)

    def init_token(self, n_embd):
        self.token = torch.nn.Parameter(torch.randn(n_embd))  # (D).

    @abstractmethod
    def select(self, timestamps, embedding=False):
        """Select token positions with shape (B, R) based on lengths. Use internal state for storage."""
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
            Attention mask with shape (B, L', L').
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


class HTStrategyImpl(HTStrategyBase):
    """Simple implementation of common strategy algorithms.

    Args:
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last` or `none`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        embedding: Either `end_ht`, `avg_ht`, `avg`, `last`, or `mix_end_ht_avg`.
    """
    def __init__(self, n_embd, apply_probability=0.5, token_selection="random",
                 predict="input_tokens", embedding="end_ht"):
        if token_selection not in {"random", "last", "none"}:
            raise ValueError(f"Unknown token selection mode: {token_selection}")
        if predict not in {"input_tokens", "history_tokens", "all"}:
            raise ValueError(f"Unknown prediction mode: {predict}")
        super().__init__(n_embd)
        self.apply_probability = apply_probability
        self.token_selection = token_selection
        self.predict = predict
        self.embedding_type = embedding

        if embedding in {"end_ht", "last", "mix_end_ht_avg"}:
            self.last_aggregator = LastAggregator()
        if embedding in {"avg", "avg_ht", "mix_end_ht_avg"}:
            self.avg_aggregator = MeanAggregator()

    @abstractmethod
    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (B, R) in the range [0, L).
        """
        pass

    def select(self, timestamps, embedding=False):
        self.seq_lens = timestamps.seq_lens
        self.length = timestamps.shape[1]
        self.embedding = embedding
        self.device = timestamps.device

        if embedding and (self.embedding_type == "avg_ht"):
            self.apply_to_batch = True
        else:
            self.apply_to_batch = not embedding

        if self.apply_to_batch:
            self.after_positions = self.select_positions(self.seq_lens)  # (B, R) in [0, L), sorted.
            r = self.after_positions.shape[1]
            self.positions = torch.arange(1, r + 1, device=self.device) + self.after_positions  # (B, R) in [0, L + R), sorted.

            # Precompute indices.
            b, l = timestamps.shape
            d = len(self.token)
            # Mask with zeros at HT positions.
            self.insert_mask = torch.ones(b, l + r, device=self.device, dtype=torch.long)  # (B, L + R).
            self.insert_mask.scatter_(1, self.positions, 0)  # (B, L + R).
            # Find prefix length for each token.
            last_input = (self.insert_mask.cumsum(1) - 1).clip(min=0)  # (B, L + R).
            # Helper index for HT token insertion from the last position.
            self.index = last_input.scatter(1, self.positions, l)  # (B, L + R).
            # Prefix length for each HT token.
            self.prev_input = last_input.take_along_dim(self.positions, 1)  # (B, R).
            if (self.predict == "history_tokens") or embedding:
                self.output_indices = batch_nonzero(1 - self.insert_mask)  # (B, R).
                self.output_lengths = (self.after_positions < self.seq_lens[:, None]).sum(1)  # (B).
            elif self.predict == "input_tokens":
                self.output_indices = batch_nonzero(self.insert_mask)  # (B, L).
                self.output_lengths = self.seq_lens
            else:
                assert self.predict == "all"

            # Disable HT for some elements.
            self.use_ht_mask = torch.rand(b, device=self.device) < self.apply_probability  # (B).

    def clear_state(self):
        del self.seq_lens
        del self.length
        del self.device
        if not self.embedding:
            del self.positions
            del self.insert_mask
            del self.index
            del self.prev_input
            if (self.predict in {"input_tokens", "history_tokens"}) or self.embedding:
                del self.output_indices
                del self.output_lengths
            del self.use_ht_mask
        del self.embedding
        del self.apply_to_batch

    def insert_tokens(self, x, timestamps):
        if self.embedding and (self.embedding_type != "avg_ht"):
            if self.embedding_type in {"last", "avg"}:
                return x, timestamps
            else:
                assert self.embedding_type in {"end_ht", "mix_end_ht_avg"}
                return add_token_to_the_end(x, timestamps, self.token.to(x.payload.dtype))
        elif self.apply_to_batch:
            b, l = x.shape
            d = len(self.token)
            r = len(self.positions)
            device = x.device

            # Insert tokens.
            extended = torch.cat([x.payload, self.token[None, None].expand(b, 1, d)], 1)  # (B, L + 1, D).
            new_x = extended.take_along_dim(self.index[:, :, None], 1)  # (B, L + R, D).
            extended = torch.cat([timestamps.payload, timestamps.payload[:, :1]], 1)  # (B, L + 1)
            new_timestamps = extended.take_along_dim(self.index, 1)  # (B, L + R).
            prev_ts = timestamps.payload.take_along_dim(self.prev_input, 1)  # (B, R).
            new_timestamps.scatter_(1, self.positions, prev_ts)

            active_tokens = (self.after_positions < x.seq_lens[:, None]).sum(1)  # (B) in [0, R].
            new_lengths = x.seq_lens + active_tokens  # (B) in [0, L + R].
            return PaddedBatch(new_x, new_lengths), PaddedBatch(new_timestamps, new_lengths)
        else:
            return x, timestamps

    def make_attention_mask(self):
        if self.apply_to_batch:
            token_selection = "none" if self.embedding else self.token_selection
            return make_ht_attention_mask(self.length, self.after_positions,
                                          active_tokens=token_selection,
                                          use_ht_mask=self.use_ht_mask,
                                          device=self.device)
        else:
            return None  # Simple causal mask.

    def extract_outputs(self, x):
        if self.embedding:
            if self.embedding_type in {"end_ht", "last"}:
                x = PaddedBatch(x.payload, (x.seq_lens - 2).clip(min=0))
                return self.last_aggregator(x)
            if self.embedding_type == "avg":
                return self.avg_aggregator(x)
            elif self.embedding_type == "mix_end_ht_avg":
                return self.last_aggregator(x) + self.avg_aggregator(PaddedBatch(x.payload, (x.seq_lens - 1).clip(min=0)))
            elif self.embedding_type == "avg_ht":
                outputs = PaddedBatch(x.payload.take_along_dim(self.output_indices[None, :, None], 1), self.output_lengths)  # (B, L, D).
                return self.avg_aggregator(outputs)
            else:
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        elif (not self.apply_to_batch) or (self.predict == "all"):
            return x
        else:
            assert self.predict in {"input_tokens", "history_tokens"}
            return PaddedBatch(x.payload.take_along_dim(self.output_indices[:, :, None], 1), self.output_lengths)  # (B, L, D).


class FullHTStrategy(HTStrategyImpl):
    """Insert history token before each real token.

    Args:
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last` or `none`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        embedding: Either `last`, `avg_ht`, or `mix_last_avg`.
    """
    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (B, R) in the range [0, L).
        """
        r = lengths.max()
        return torch.arange(r, device=lengths.device)[None].expand(len(lengths), r)


class SubsetHTStrategy(HTStrategyImpl):
    """Insert history token before each real token.

    Args:
        frequency: The average fraction of history tokens (use 0 to use single token).
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last`, or `none`.
        token_sampling: Either `uniform`, `bias_end`, or `bias_end_relaxed`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        embedding: Either `last`, `avg_ht`, or `mix_last_avg`.
    """
    def __init__(self, n_embd, frequency=0.1, apply_probability=0.5,
                 token_selection="random", token_sampling="uniform",
                 predict="input_tokens", embedding="last"):
        super().__init__(n_embd,
                         apply_probability=apply_probability,
                         token_selection=token_selection,
                         predict=predict,
                         embedding=embedding)
        self.frequency = frequency
        self.token_sampling = token_sampling

    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (R) in the range [0, L).
        """
        max_length = lengths.max().item()
        max_tokens = max(1, int(round(self.frequency * max_length)))
        if self.token_sampling == "uniform":
            return batch_randperm(len(lengths), max_length, device=self.device)[:, :max_tokens].sort(dim=1)[0]  # (B, R) in [0, L), sorted.
        elif self.token_sampling == "bias_end":
            from_length = max_length // 2
            return from_length + batch_randperm(len(lengths), max_length - from_length, device=self.device)[:, :max_tokens].sort(dim=1)[0]  # (B, R) in [L // 2, L), sorted.
        elif self.token_sampling == "bias_end_relaxed":
            from_length = lengths.sum().item() // (len(lengths) * 2)  # Mean / 2.
            return from_length + batch_randperm(len(lengths), max_length - from_length, device=self.device)[:, :max_tokens].sort(dim=1)[0]  # (R) in [L // 2, L), sorted.
        else:
            raise ValueError(f"Unknown sampling type: {self.token_sampling}")


class FixedHTStrategy(SubsetHTStrategy):
    """Insert history token at specified positions.

    Args:
        positions: The indices of tokens to insert HT after.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        apply_probability: The probability of HT usage for each real token.
    """
    def __init__(self, n_embd, positions, apply_probability=0.5, predict="input_tokens", embed_end=False):
        if predict not in {"input_tokens", "history_tokens", "all"}:
            raise ValueError(f"Unknown prediction mode: {predict}")
        super().__init__(n_embd, frequency=None,
                         apply_probability=apply_probability,
                         predict=predict)
        self.specified_positions = positions
        self.embed_end = embed_end

    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (R) in the range [0, L).
        """
        max_length = lengths.max().item()
        positions = torch.tensor(self.specified_positions, device=self.device, dtype=torch.long).sort()[0]  # (R) in [0, L), sorted.
        positions = positions[positions < max_length]
        positions = positions[None].expand(len(lengths), len(positions))  # (B, R).
        return positions

    def insert_tokens(self, x, timestamps):
        if self.embedding:
            token = self.token.to(x.payload.dtype)
            if self.embed_end:
                return add_token_to_the_end(x, timestamps, token)
            else:
                # Truncate to the first valid position.
                positions = torch.tensor(self.specified_positions, device=self.device, dtype=torch.long).sort()[0]  # (R).
                valid = positions[None].expand(x.shape[0], len(positions)) < x.seq_lens[:, None]  # (B, R).
                valid_sum = valid.cumsum(1)  # (B, R).
                last = torch.logical_and(valid, valid_sum == valid.sum(1)[:, None])
                last_index = last.long().argmax(1)  # (B).
                new_lengths = torch.minimum(positions[last_index] + 1, x.seq_lens)  # (B).
                return add_token_to_the_end(PaddedBatch(x.payload, new_lengths), PaddedBatch(timestamps.payload, new_lengths), token)
        else:
            return super().insert_tokens(x, timestamps)


class LastHTStrategy(HTStrategyImpl):
    """Insert history token at the end of each sequence.

    Args:
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last` or `none`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
    """
    def __init__(self, n_embd, predict="input_tokens"):
        super().__init__(n_embd, apply_probability=1, token_selection="none", predict=predict)

    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (B, R) in the range [0, L).
        """
        return (lengths.unsqueeze(1) - 1).clip(min=0)  # (B, 1).


class MidHTStrategy(HTStrategyImpl):
    """Insert history token at the end of each sequence.

    Args:
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last` or `none`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
    """
    def __init__(self, n_embd, predict="input_tokens"):
        super().__init__(n_embd, apply_probability=1, token_selection="last", predict=predict)

    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (B, R) in the range [0, L).
        """
        assert (lengths % 2 == 0).all()
        return (lengths // 2 - 1).clip(min=0).unsqueeze(1)  # (B, 1).
