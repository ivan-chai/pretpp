import math
import torch


class TransformerConfig:
    """Transformer configuration.

    Args:
        angular_embeddings: Use use sine/cosine if it is set to true and trainable positional embeddings otherwise.
    """
    def __init__(self, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None,
                 dropout=0.1, angular_embeddings=True, output_hidden_states=True, causal=False):
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.dropout = dropout
        self.angular_embeddings = angular_embeddings
        self.output_hidden_states = output_hidden_states
        self.causal = causal


class PositionalEncoding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=config.dropout)
        self.angular = config.angular_embeddings
        if config.angular_embeddings:
            position = torch.arange(config.n_positions).unsqueeze(1)  # (L, 1).
            div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))  # (D // 2).
            pe = torch.zeros(config.n_positions, config.n_embd)  # (L, D).
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe, persistent=False)
        else:
            pe = torch.arange(config.n_positions)
            self.register_buffer("pe", pe, persistent=False)
            self.embeddings = torch.nn.Embedding(config.n_positions, config.n_embd)

    def forward(self, x):
        # x: (B, L, D).
        if self.angular:
            x = x + self.pe[:x.shape[1]]
        else:
            x = x + self.embeddings(self.pe[:x.shape[1]])
        return self.dropout(x)


class TransformerOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class Transformer(torch.nn.Module):
    """Simple transformer mimicing HuggingFace interface."""
    def __init__(self, config):
        if not config.output_hidden_states:
            raise ValueError("output_hidden_states must be True")
        super().__init__()
        self.config = config
        layer = torch.nn.TransformerEncoderLayer(d_model=config.n_embd,
                                                 nhead=config.n_head,
                                                 dim_feedforward=config.n_inner,
                                                 dropout=config.dropout,
                                                 batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, config.n_layer)
        self.positional = PositionalEncoding(config)
        if config.causal:
            sa_mask = torch.triu(torch.ones((config.n_positions, config.n_positions), dtype=torch.bool), diagonal=1)
            self.register_buffer("sa_mask", sa_mask)
        else:
            self.sa_mask = None

    def forward(self, inputs_embeds, attention_mask=None):
        b, l, d = inputs_embeds.shape
        sa_mask = self.sa_mask[:l, :l] if self.sa_mask is not None else None
        last_hidden_state = self.encoder(self.positional(inputs_embeds),
                                         mask=sa_mask,
                                         src_key_padding_mask=~attention_mask.bool(),
                                         is_causal=self.config.causal)
        return TransformerOutput(last_hidden_state)
