import torch
from hotpp.data import PaddedBatch
from hotpp.nn import SimpleTransformer


class HistoryTokenTransformer(SimpleTransformer):
    """An extension of the transformer model with extra <history-tokens> (HT) for context aggregation.

    Args:
        strategy_partial: History token strategy initializer which accepts embedding dim as input.
        val_strategy_partial: History token strategy used for validation and testing.
        history_token_fraction: The average fraction of batches to apply history token to.
        history_token_locality: The value between 0 and 1 with 0 meaning uniform history token selection
            and 1 for using the last available token.
        embed_layer: The layer to extract HT embeddings from, a list of indices, or `all`. By default, extract
            the output of the final layer.
    """
    def __init__(self, input_size, strategy_partial, val_strategy_partial=None, embed_layer=None, **kwargs):
        super().__init__(input_size, **kwargs)
        self.strategy = strategy_partial(self.n_embd)
        if self.strategy.causal != self.causal:
            raise NotImplementedError("Strategy and transformer causality mismatch.")
        if val_strategy_partial is not None:
            self.val_strategy = val_strategy_partial(self.n_embd)
            if self.val_strategy.causal != self.causal:
                raise NotImplementedError("Validation strategy and transformer causality mismatch.")
        else:
            self.val_strategy = None
        if embed_layer == "all":
            embed_layer = list(range(self.n_layer))
        self.embed_layer = embed_layer

    def embed(self, x, timestamps):
        x = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).
        x, timestamps = self.add_sos(x, timestamps)

        with self.strategy(timestamps, embedding=True) as strategy:
            x, timestamps, attention_mask = strategy.apply(x, timestamps)

            # Extract history token embedding.
            x = PaddedBatch(self.positional(x.payload, timestamps.payload), x.seq_lens)
            return_states = "full" if self.embed_layer is not None else False
            if self.rope is not None:
                with self.rope.cache(timestamps.payload):
                    outputs, states = self.transform(x,
                                                     return_states=return_states,
                                                     attention_mask=attention_mask)
            else:
                outputs, states = self.transform(x,
                                                 return_states=return_states,
                                                 attention_mask=attention_mask)
            if self.embed_layer is not None:
                # states: N * (B, L, D).
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
            embeddings = strategy.extract_outputs(outputs)  # (B, D).
            assert embeddings.ndim == 2
        return embeddings

    def forward(self, x, timestamps, states=None, return_states=False):
        if self.training:
            mode_strategy = self.strategy
        elif self.val_strategy is None:
            # Don't insert history tokens.
            return super().forward(x, timestamps, states=states, return_states=return_states)
        else:
            mode_strategy = self.val_strategy
        if return_states:
            raise NotImplementedError("HistoryTokenTransformer doesn't support states return.")
        b, l = x.shape
        device = x.device
        x = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).
        x, timestamps = self.add_sos(x, timestamps)

        with mode_strategy(timestamps) as strategy:
            x, timestamps, attention_mask = strategy.apply(x, timestamps)  # (B, L', D).

            # Apply transformer.
            x = PaddedBatch(self.positional(x.payload, timestamps.payload), x.seq_lens)  # (B, L', D).
            if self.rope is not None:
                with self.rope.cache(timestamps.payload):
                    outputs, _ = self.transform(x, attention_mask=attention_mask)  # (B, L', D).
            else:
                outputs, _ = self.transform(x, attention_mask=attention_mask)  # (B, L', D).
            outputs = strategy.extract_outputs(outputs)
        if (self.sos is not None) and isinstance(outputs.payload, dict) and (outputs.payload.get("special_token_mask", None) is not None):
            if outputs.payload["special_token_mask"][:, 0].any():
                raise RuntimeError("Can't remove SOS when history tokens are at the beginning of a sequence.")
        outputs = self.remove_sos(outputs)
        states = None
        return outputs, states
