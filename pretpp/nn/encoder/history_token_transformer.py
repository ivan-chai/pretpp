import torch
from hotpp.data import PaddedBatch
from hotpp.nn import SimpleTransformer


class HistoryTokenTransformer(SimpleTransformer):
    """An extension of the transformer model with extra <history-tokens> (HT) for context aggregation.

    Args:
        strategy_partial: History token strategy initializer which accepts embedding dim as input.
        history_token_fraction: The average fraction of batches to apply history token to.
        history_token_locality: The value between 0 and 1 with 0 meaning uniform history token selection
            and 1 for using the last available token.
        embed_layer: The layer to extract HT embeddings from, a list of indices, or `all`. By default, extract
            the output of the final layer.
    """
    def __init__(self, input_size, strategy_partial, embed_layer=None, **kwargs):
        super().__init__(input_size, **kwargs)
        if not self.causal:
            raise NotImplementedError("A history-token transformer must be causal.")
        self.strategy = strategy_partial(self.n_embd)
        if embed_layer == "all":
            embed_layer = list(range(self.n_layer))
        self.embed_layer = embed_layer

    def embed(self, x, timestamps):
        x = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).

        with self.strategy(timestamps, embedding=True) as strategy:
            x, timestamps, attention_mask = strategy.apply(x, timestamps)

            # Extract history token embedding.
            x = PaddedBatch(self.positional(x.payload, timestamps.payload), x.seq_lens)
            if self.embed_layer is not None:
                _, states = self.transform(x, return_states="full", attention_mask=attention_mask)  # N * (B, L, D).
                try:
                    embed_layers = list(self.embed_layer)
                except TypeError:
                    embed_layers = [self.embed_layer]
                all_outputs = []
                for embed_layer in embed_layers:
                    embed_layer = embed_layer % self.n_layer
                    outputs = states[embed_layer]  # (B, L, D).
                    all_outputs.append(outputs)
                outputs = torch.stack(all_outputs, 0).sum(0)  # (B, L, D).
                outputs = PaddedBatch(outputs, x.seq_lens)
            else:
                outputs, _ = self.transform(x, attention_mask=attention_mask)
            embeddings = strategy.extract_outputs(outputs)  # (B, D).
            assert embeddings.ndim == 2
        return embeddings

    def forward(self, x, timestamps, states=None, return_states=False):
        if not self.training:
            # Don't insert history tokens.
            return super().forward(x, timestamps, states=states, return_states=return_states)
        if return_states:
            raise NotImplementedError("HistoryTokenTransformer doesn't support states return.")
        b, l = x.shape
        device = x.device
        x = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).

        with self.strategy(timestamps) as strategy:
            x, timestamps, attention_mask = strategy.apply(x, timestamps)  # (B, L', D).

            # Apply transformer.
            x = PaddedBatch(self.positional(x.payload, timestamps.payload), x.seq_lens)  # (B, L', D).
            outputs, _ = self.transform(x, attention_mask=attention_mask)  # (B, L', D).
            outputs = strategy.extract_outputs(outputs)
        states = None
        return outputs, states
