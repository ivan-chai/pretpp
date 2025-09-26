import torch
from hotpp.data import PaddedBatch
from hotpp.nn import SimpleTransformer

from .history_token_strategy import add_token_to_the_end


class RecurrentMemoryTransformer(SimpleTransformer):
    """An extension of the transformer model with extra <memory-tokens> and recurrent inference.

    Args:
        chunk_size: The maximum size of input chunk at each iteration.
        n_tokens: The number of tokens.
        multitoken: Whether to use the same history token for all positions or use different tokens.
    """
    def __init__(self, input_size, chunk_size, *, n_tokens=3, multitoken=False, **kwargs):
        super().__init__(input_size, **kwargs)
        self.chunk_size = chunk_size
        self.n_tokens = n_tokens
        self.multitoken = multitoken
        self.token = torch.nn.Parameter(torch.randn(self.n_tokens if self.multitoken else 1, self.n_embd))  # (K, D).

    def forward_chunk(self, embeddings, timestamps, attention_mask=None, memory=None, memory_timestamps=None):
        """Apply recurrent block.
        Args:
            x: Input data with shape (B, L, D) after input projection, but before positional encoding.
            timestamps: Input timestamps with shape (B, L).
            attention_mask: Additional attention mask with shape (L, L) or (B, L, L) which contains True for masked connections.
                The mask will be merged with causal mask if causal transformer is applied.
            memory: If provided, must be memory with shape (B, K, D).
            memory_timestamps: Memory timestamps with shape (B, K).

        Returns:
            A tuple of outputs with shape (B, L, D), new memory with shape (B, K, D), and memory timestamps with shape (B, K).
        """
        if memory is not None:
            # Prepend memory.
            if memory_timestamps is None:
                raise ValueError("Memory timestamps must be provided.")
            assert memory.shape[1] == self.n_tokens
            embeddings = PaddedBatch(torch.cat([memory, embeddings.payload], 1),
                                     embeddings.seq_lens + self.n_tokens)
            timestamps = PaddedBatch(torch.cat([memory_timestamps, timestamps.payload], 1),
                                     timestamps.seq_lens + self.n_tokens)
            if attention_mask is not None:
                full_mask = torch.zeros(list(attention_mask.shape[:-2]) + [self.n_tokens + attention_mask.shape[0],
                                                                           self.n_tokens + attention_mask.shape[1]],
                                        device=attention_mask.device,
                                        dtype=attention_mask.dtype)
                full_mask[..., self.n_tokens:, self.n_tokens:] = attention_mask
                attention_mask = full_mask

        # Append output memory tokens.
        b, _, d = embeddings.payload.shape
        token = self.token.to(embeddings.payload.dtype).expand(self.n_tokens, d)
        for i in range(self.n_tokens):
            embeddings, timestamps = add_token_to_the_end(embeddings, timestamps, token[i])
        if attention_mask is not None:
            full_mask = torch.zeros(list(attention_mask.shape[:-2]) + [self.n_tokens + attention_mask.shape[0],
                                                                       self.n_tokens + attention_mask.shape[1]],
                                    device=attention_mask.device,
                                    dtype=attention_mask.dtype)
            full_mask[..., :-self.n_tokens, :-self.n_tokens] = attention_mask
            attention_mask = full_mask

        embeddings = PaddedBatch(self.positional(embeddings.payload, timestamps.payload),
                                 embeddings.seq_lens)
        if self.rope is not None:
            with self.rope.cache(timestamps.payload):
                outputs, states = self.transform(embeddings, attention_mask=attention_mask)
        else:
            outputs, states = self.transform(embeddings, attention_mask=attention_mask)

        # Extract memory.
        memory_indices = torch.stack([embeddings.seq_lens - self.n_tokens + i for i in range(self.n_tokens)], 1)  # (B, K).
        new_memory = outputs.payload.take_along_dim(memory_indices.unsqueeze(2), 1)  # (B, K, D).
        new_memory_timestamps = timestamps.payload.take_along_dim(memory_indices, 1)  # (B, K).

        # Extract outputs.
        offset = 0 if memory is None else self.n_tokens
        outputs = PaddedBatch(outputs.payload[:, offset:-self.n_tokens], (outputs.seq_lens - offset - self.n_tokens).clip(min=0))
        return outputs, new_memory, new_memory_timestamps

    def forward_recurrent(self, embeddings, timestamps, attention_mask=None):
        chunk_outputs = []
        final_memory = None
        memory = None
        memory_timestamps = None
        for offset in range(0, embeddings.payload.shape[1], self.chunk_size):
            chunk_embeddings = PaddedBatch(embeddings.payload[:, offset:offset + self.chunk_size], (embeddings.seq_lens - offset).clip(min=0, max=self.chunk_size))
            chunk_timestamps = PaddedBatch(timestamps.payload[:, offset:offset + self.chunk_size], (timestamps.seq_lens - offset).clip(min=0, max=self.chunk_size))
            chunk_mask = attention_mask[..., offset:, offset:] if attention_mask is not None else None
            chunk_output, memory, memory_timestamps = self.forward_chunk(chunk_embeddings, chunk_timestamps, chunk_mask,
                                                                         memory, memory_timestamps)
            chunk_outputs.append(chunk_output)
            memory_mask = (chunk_embeddings.seq_lens > 0)[:, None, None]  # (B, 1, 1).
            if final_memory is None:
                final_memory = memory * memory_mask
            else:
                final_memory = torch.where(memory_mask, memory, final_memory)
        outputs = PaddedBatch(torch.cat([output.payload for output in chunk_outputs], 1),
                              embeddings.seq_lens)
        return outputs, final_memory

    def embed(self, x, timestamps):
        embeddings = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).
        _, embeddings = self.forward_recurrent(embeddings, timestamps)  # (B, K, D).
        return embeddings.mean(1)  # (B, D).

    def forward(self, x, timestamps, states=None, return_states=False):
        if return_states:
            raise NotImplementedError("RecurrentMemoryTransformer doesn't support states return.")
        embeddings = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).
        outputs, _ = self.forward_recurrent(embeddings, timestamps)  # (B, K, D).
        states = None
        return outputs, states
