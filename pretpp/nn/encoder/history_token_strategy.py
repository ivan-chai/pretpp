from abc import ABC, abstractmethod

import torch
from hotpp.data import PaddedBatch
from hotpp.nn import LastAggregator, MeanAggregator


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


def make_ht_attention_mask(length, ht_positions, active_tokens="random", device=None):
    """Make attention mask for history tokens.

    Args:
        length: The length of input sequence.
        ht_positions: Sorted token indices to insert HT after with shape (R).
        active_tokens: Either `random`, `last`, `none` or a tensor with shape (L) of HT token indices
            with 0 meaning no HT and R meaning the last HT.

    Returns:
        Atteniton mask with shape (L + R, L + R).
    """
    # Prepare.
    if device is None:
        device = torch.get_default_device()
    l = length
    r = len(ht_positions)
    assert r <= l
    ht_positions_mask = torch.zeros(l, device=device, dtype=torch.bool).scatter_(0, ht_positions, True)  # (L).

    # Generate prefix lengths.
    if active_tokens in {"random", "last"}:
        n_active_tokens = torch.cat([torch.zeros_like(ht_positions_mask[:1]), ht_positions_mask[:-1]], 0).cumsum(0)  # (L).
        if active_tokens == "random":
            active_tokens = (torch.rand(l, device=device) * n_active_tokens).round().long()  # (L) in [0, R].
        else:
            assert active_tokens == "last"
            active_tokens = n_active_tokens
    elif active_tokens == "none":
        active_tokens = torch.zeros(l, device=device, dtype=torch.long)
    assert active_tokens.shape == (l,)
    n_summarize = torch.cat([torch.zeros_like(ht_positions[:1]), ht_positions + 1], 0).take_along_dim(active_tokens, 0)
    selected_tokens = active_tokens - 1

    # Fill grouped attention mask, with separated real and history tokens blocks:
    # [LL LR]
    # [RL RR]
    # The mask will be sorted after filling.

    mask_ll = torch.arange(l, device=device)[None] < n_summarize[:, None]
    mask_rr = ~torch.eye(r, device=device, dtype=torch.bool)
    mask_lr = torch.ones(l, r, device=device, dtype=torch.bool).scatter_(1, selected_tokens.clip(min=0)[:, None], False)
    mask_lr[selected_tokens == -1, 0] = True
    mask_rl = torch.zeros(r, l, device=device, dtype=torch.bool)

    # Merge and order masks.
    ht_mask = torch.zeros(l + r, device=device, dtype=torch.bool).scatter_(0, torch.arange(1, 1 + r, device=device) + ht_positions, True)
    real_mask = ~ht_mask
    order = torch.argsort(torch.cat([torch.nonzero(real_mask).squeeze(1), torch.nonzero(ht_mask).squeeze(1)], 0))

    mask = torch.cat([
        torch.cat([mask_ll, mask_lr], 1),
        torch.cat([mask_rl, mask_rr], 1)
    ], 0)
    mask = mask.take_along_dim(order[:, None], 0).take_along_dim(order[None, :], 1)
    return mask


class HTStrategyBase(ABC, torch.nn.Module):
    """History token strategy."""

    def __init__(self, n_embd):
        super().__init__()
        self.init_token(n_embd)

    def init_token(self, n_embd):
        self.token = torch.nn.Parameter(torch.randn(n_embd))  # (D).

    @property
    def causal(self):
        return True

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
            Indices with shape (R) in the range [0, L).
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
            self.apply_to_batch = not embedding and torch.rand([]) < self.apply_probability

        if self.apply_to_batch:
            self.after_positions = self.select_positions(self.seq_lens)  # (R) in [0, L), sorted.
            self.positions = torch.arange(1, len(self.after_positions) + 1, device=self.device) + self.after_positions  # (R) in [0, L + R), sorted.

            # Precompute indices.
            b, l = timestamps.shape
            d = len(self.token)
            r = len(self.positions)
            self.insert_mask = torch.ones(l + r, device=self.device, dtype=torch.long)  # (L + R).
            self.insert_mask.scatter_(0, self.positions, 0)  # (L + R).
            last_input = (self.insert_mask.cumsum(0) - 1).clip(min=0)  # (L + R).
            self.index = last_input.scatter(0, self.positions, l)  # (L + R).
            self.prev_input = last_input.take_along_dim(self.positions, 0)  # (R).
            if (self.predict == "history_tokens") or embedding:
                self.output_indices = (1 - self.insert_mask).nonzero().squeeze(1)  # (L).
                self.output_lengths = (self.after_positions < self.seq_lens[:, None]).sum(1)  # (B).
            elif self.predict == "input_tokens":
                self.output_indices = self.insert_mask.nonzero().squeeze(1)  # (L).
                self.output_lengths = self.seq_lens
            else:
                assert self.predict == "all"

    def clear_state(self):
        del self.seq_lens
        del self.length
        del self.device
        if self.apply_to_batch:
            del self.positions
            del self.insert_mask
            del self.index
            del self.prev_input
            if (self.predict in {"input_tokens", "history_tokens"}) or self.embedding:
                del self.output_indices
                del self.output_lengths
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
            new_x = extended.take_along_dim(self.index[None, :, None], 1)  # (B, L + R, D).
            extended = torch.cat([timestamps.payload, timestamps.payload[:, :1]], 1)  # (B, L + 1)
            new_timestamps = extended.take_along_dim(self.index[None, :], 1)  # (B, L + R).
            prev_ts = timestamps.payload.take_along_dim(self.prev_input[None], 1)  # (B, R).
            new_timestamps.scatter_(1, self.positions[None].expand(b, r), prev_ts)

            active_tokens = (self.after_positions[None] < x.seq_lens[:, None]).sum(1)  # (B) in [0, R].
            new_lengths = x.seq_lens + active_tokens  # (B) in [0, L + R].
            return PaddedBatch(new_x, new_lengths), PaddedBatch(new_timestamps, new_lengths)
        else:
            token_gradient_branch = 0 * self.token.sum()
            new_x = PaddedBatch(x.payload + token_gradient_branch, x.seq_lens)
            return new_x, timestamps

    def make_attention_mask(self):
        if self.apply_to_batch:
            token_selection = "none" if self.embedding else self.token_selection
            return make_ht_attention_mask(self.length, self.after_positions,
                                          active_tokens=token_selection,
                                          device=self.device)
        else:
            return None

    def extract_outputs(self, x):
        if self.embedding:
            if self.embedding_type in {"end_ht", "last"}:
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
            return PaddedBatch(x.payload.take_along_dim(self.output_indices[None, :, None], 1), self.output_lengths)  # (B, L, D).


class FullHTStrategy(HTStrategyImpl):
    """Insert history token before each real token.

    Args:
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last` or `none`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        embedding: Either `end_ht`, `avg_ht`, `avg`, `last`, or `mix_end_ht_avg`.
    """
    def select_positions(self, lengths):
        """Select tokens to insert HT after.

        Args:
            lengths: Input lengths.

        Returns:
            Indices with shape (R) in the range [0, L).
        """
        return torch.arange(lengths.max(), device=lengths.device)


class SubsetHTStrategy(HTStrategyImpl):
    """Insert history token before each real token.

    Args:
        frequency: The average fraction of history tokens (use 0 to use single token).
        apply_probability: The probability of HT usage for each real token.
        token_selection: Either `random`, `last`, or `none`.
        token_sampling: Either `uniform`, `bias_end`, or `bias_end_relaxed`.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
        embedding: Either `end_ht`, `avg_ht`, `avg`, `last`, or `mix_end_ht_avg`.
    """
    def __init__(self, n_embd, frequency=0.1, apply_probability=0.5,
                 token_selection="random", token_sampling="uniform",
                 predict="input_tokens", embedding="end_ht"):
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
            return torch.randperm(max_length, device=self.device)[:max_tokens].sort()[0]  # (R) in [0, L), sorted.
        elif self.token_sampling == "bias_end":
            from_length = max_length // 2
            return from_length + torch.randperm(max_length - from_length, device=self.device)[:max_tokens].sort()[0]  # (R) in [L // 2, L), sorted.
        elif self.token_sampling == "bias_end_relaxed":
            from_length = lengths.sum().item() // (len(lengths) * 2)  # Mean / 2.
            return from_length + torch.randperm(max_length - from_length, device=self.device)[:max_tokens].sort()[0]  # (R) in [L // 2, L), sorted.
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
        return positions

    def insert_tokens(self, x, timestamps):
        if self.embedding:
            token = self.token.to(x.payload.dtype)
            if self.embed_end:
                return add_token_to_the_end(x, timestamps, token)
            else:
                # Truncate to the first valid position.
                positions = torch.tensor(self.specified_positions, device=self.device, dtype=torch.long)  # (R).
                valid = positions[None].expand(x.shape[0], len(positions)) < x.seq_lens[:, None]  # (B, R).
                valid_sum = valid.cumsum(1)  # (B, R).
                last = torch.logical_and(valid, valid_sum == valid.sum(1)[:, None])
                last_index = last.long().argmax(1)  # (B).
                new_lengths = torch.minimum(positions[last_index] + 1, x.seq_lens)  # (B).
                return add_token_to_the_end(PaddedBatch(x.payload, new_lengths), PaddedBatch(timestamps.payload, new_lengths), token)
        else:
            return super().insert_tokens(x, timestamps)


class LastHTStrategy(HTStrategyBase):
    """Insert history token to the end of each sequence.

    Args:
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
    """
    def __init__(self, n_embd, predict="input_tokens"):
        if predict not in {"input_tokens", "history_tokens", "all"}:
            raise ValueError(f"Unknown prediction mode: {predict}")
        super().__init__(n_embd)
        self.predict = predict

    def select(self, timestamps, embedding=False):
        self.seq_lens = timestamps.seq_lens
        self.embedding = embedding
        self.device = timestamps.device

    def clear_state(self):
        del self.seq_lens
        del self.embedding
        del self.device

    def insert_tokens(self, x, timestamps):
        return add_token_to_the_end(x, timestamps, self.token.to(x.payload.dtype))

    def make_attention_mask(self):
        return None  # Simple causal mask.

    def extract_outputs(self, x):
        if self.embedding:
            return x.payload.take_along_dim((x.seq_lens[:, None, None] - 1).clip(min=0), 1).squeeze(1)  # (B, D).
        elif self.predict == "history_tokens":
            payload =  x.payload.take_along_dim((x.seq_lens[:, None, None] - 1).clip(min=0), 1)  # (B, 1, D).
            return PaddedBatch(payload, torch.ones_like(self.seq_lens))
        elif self.predict == "input_tokens":
            return PaddedBatch(x.payload[:, :-1], (x.seq_lens - 1).clip(min=0))
        else:
            assert self.predict == "all"
            return x


class RecMemHTStrategy(HTStrategyImpl):
    """Simulate recurrent-memory transformer.

    Args:
        n_tokens: The number of tokens.
        apply_probability: The probability of HT usage for each real token.
        predict: The type of tokens used for prediction (`input_tokens`, `history_tokens` or `all`).
    """
    def __init__(self, n_embd, n_tokens=3, apply_probability=1.0,
                 predict="input_tokens"):
        super(HTStrategyBase, self).__init__()
        self.n_tokens = n_tokens
        super().__init__(n_embd, apply_probability=apply_probability,
                         predict=predict, embedding="last")

    def init_token(self, n_embd):
        self.token = torch.nn.Parameter(torch.randn(self.n_tokens, n_embd))  # (K, D).

    def select_positions(self, lengths):
        max_length = lengths.max().item()
        offset = torch.randint(0, max_length, size=[]).item()
        return torch.full([self.n_tokens], offset, device=self.device)  # (R) in [0, L), sorted.

    def insert_tokens(self, x, timestamps):
        if self.embedding:
            # Put K tokens to the end.
            token = self.token.to(x.payload.dtype)  # (K, D).
            for i in range(self.n_tokens):
                x, timestamps = add_token_to_the_end(x, timestamps, token[i])
            return x, timestamps
        elif self.apply_to_batch:
            b, l = x.shape
            d = self.token.shape[1]
            r = len(self.positions)
            device = x.device

            # Insert tokens.
            offset = self.positions[0].item()
            new_x = torch.cat([x.payload[:, :offset],
                               self.token[None].expand(b, self.n_tokens, d),
                               x.payload[:, offset:]], 1)  # (B, L + K, D).
            new_timestamps = torch.cat([timestamps.payload[:, :offset],
                                        timestamps.payload[:, offset - 1:offset].expand(b, self.n_tokens),
                                        timestamps.payload[:, offset:]], 1)  # (B, L + K)
            assert new_x.shape[:2] == new_timestamps.shape
            active_tokens = (self.after_positions[None] < x.seq_lens[:, None]).sum(1)  # (B) in [0, R].
            new_lengths = x.seq_lens + active_tokens  # (B) in [0, L + R].
            return PaddedBatch(new_x, new_lengths), PaddedBatch(new_timestamps, new_lengths)
        else:
            token_gradient_branch = 0 * self.token.sum()
            new_x = PaddedBatch(x.payload + token_gradient_branch, x.seq_lens)
            return new_x, timestamps

    def make_attention_mask(self):
        """Use all history tokens, but skip early events."""
        if self.apply_to_batch:
            first = self.after_positions[0]
            total_length = self.n_tokens + self.length
            mask = torch.zeros((total_length, total_length), dtype=torch.bool, device=self.device)
            mask[:first, first + self.n_tokens:] = True
            mask[first + self.n_tokens:, :first] = True
            return mask
        else:
            return None

    def extract_outputs(self, x):
        if self.embedding:
            # Average last K tokens.
            assert (x.seq_lens >= self.n_tokens).all()
            indices = torch.cat([(x.seq_lens[:, None, None] - i) for i in range(1, 1 + self.n_tokens)], 1)  # (B, K, 1).
            return x.payload.take_along_dim(indices, 1).mean(1)  # (B, D).
        else:
            return super().extract_outputs(x)


class LongFormerHTStrategy(HTStrategyBase):
    """Simulate LongFormer (don't insert extra tokens). Predict the average of global tokens.

    Args:
        kernel_size: If not None, use convolution attention mask with a specified kernel size.
        global_frequency: The fraction of global tokens.
    """
    def __init__(self, n_embd, kernel_size=None, global_frequency=0.1):
        super().__init__(n_embd)
        self.kernel_size = kernel_size
        self.global_frequency = global_frequency

    @property
    def causal(self):
        return False

    def init_token(self, n_embd):
        pass

    def select(self, timestamps, embedding=False):
        self.embedding = embedding
        self.device = timestamps.device
        self.length = timestamps.payload.shape[1]
        self.seq_lens = timestamps.seq_lens
        max_length = self.seq_lens.max().item()
        assert max_length > 0
        max_tokens = max(1, int(round(self.global_frequency * max_length)))
        if not self.embedding:
            # Use random locations at train to prevent overfitting.
            global_positions = 1 + torch.randperm(max_length - 1, device=self.device)[:max_tokens - 1].sort()[0]  # (R - 1) in [1, L), sorted.
            global_positions = torch.cat([torch.zeros([1], device=self.device, dtype=torch.long), global_positions])  # (R) in [0, L), sorted.
        else:
            # Use regular locations at inference for preproducibility.
            global_positions = torch.arange(0, max_length, max(1, int(round(max_length / max_tokens))), device=self.device)
        self.global_positions = global_positions

    def clear_state(self):
        del self.embedding
        del self.device
        del self.length
        del self.seq_lens
        del self.global_positions

    def insert_tokens(self, x, timestamps):
        return x, timestamps

    def make_attention_mask(self):
        # Put ones at diagonals.
        conv_mask = torch.ones(self.length, self.length, dtype=torch.bool, device=self.device)
        if self.kernel_size:
            conv_mask = torch.tril(conv_mask, self.kernel_size)
            conv_mask = torch.triu(conv_mask, -self.kernel_size)

        # Add causality to all tokens except global.
        conv_mask = torch.tril(conv_mask)

        # Put ones at global rows and cols.
        glob_mask = torch.zeros(self.length, self.length, dtype=torch.bool, device=self.device)
        glob_mask[self.global_positions] = True
        glob_mask[:, self.global_positions] = True

        # Combine.
        mask = ~torch.logical_or(conv_mask, glob_mask)
        return mask  # Zeros for allowed interractions.

    def extract_outputs(self, x):
        if self.embedding:
            return x.payload.take_along_dim((x.seq_lens[:, None, None] - 1).clip(min=0), 1).squeeze(1)  # (B, D).
        else:
            return x


class NoHTStrategy(HTStrategyBase):
    """Don't use history tokens, extract last token embedding."""
    def __init__(self, n_embd):
        super().__init__(n_embd)

    def init_token(self, n_embd):
        pass

    def select(self, timestamps, embedding=False):
        self.embedding = embedding
        self.device = timestamps.device

    def clear_state(self):
        del self.embedding
        del self.device

    def insert_tokens(self, x, timestamps):
        return x, timestamps

    def make_attention_mask(self):
        return None  # Simple causal mask.

    def extract_outputs(self, x):
        if self.embedding:
            return x.payload.take_along_dim((x.seq_lens[:, None, None] - 1).clip(min=0), 1).squeeze(1)  # (B, D).
        else:
            return x
