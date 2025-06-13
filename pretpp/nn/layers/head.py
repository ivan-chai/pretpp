import torch
from hotpp.data import PaddedBatch
from hotpp.nn import Head, ConditionalHead, SimpleTransformer
from .metric import MetricLayer


class IdentityHead(torch.nn.Module):
    """A simple identity head.

    Args:
        input_size: Embedding size.
        output_size: Output dimension.
    """
    def __init__(self, input_size, output_size=None):
        super().__init__()
        if output_size is None:
            output_size = input_size
        elif input_size != output_size:
            raise ValueError("Input and output size must be equal for identity.")
        self._output_size = output_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x):
        if x.payload.shape[-1] != self._output_size:
            raise ValueError(f"Incorrect input size: {x.payload.shape[-1]} != {self._output_size}")
        return x


class NormalizationHead(IdentityHead):
    """L2 normalization head.

    Args:
        input_size: Embedding size.
        output_size: Output dimension.
    """
    def forward(self, x):
        x = super().forward(x)
        return PaddedBatch(torch.nn.functional.normalize(x.payload, dim=-1), x.seq_lens)


class MetricHead(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dims=None,
                 metric_params=None, head_params=None):
        super().__init__()
        if head_params and head_params.get("use_batch_norm", False):
            self.bn = torch.nn.BatchNorm1d(input_size)
        else:
            self.bn = None
        layers = []
        if not hidden_dims:
            layers.append(MetricLayer(input_size, output_size, **(metric_params or {})))
        else:
            layers.append(MetricLayer(input_size, hidden_dims[0], **(metric_params or {})))
            layers.append(Head(hidden_dims[0], output_size, hidden_dims=hidden_dims[1:], **(head_params or {})))
        self.proj = torch.nn.Sequential(*layers)
        self._output_size = output_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x):
        if self.bn is not None:
            # Workaround for a correct BatchNorm computation for variable-lenght sequences.
            x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
            assert x.ndim > 2  # (B, L, *, D).
            shape = list(x.shape)
            x_masked = x[mask]  # (V, *, D).
            v = len(x_masked)
            x_mapped = self.bn(x_masked.flatten(0, -2)).reshape(*x_masked.shape)  # (V, *, D).
            x_new = torch.zeros(*[shape[:-1] + [shape[-1]]], dtype=x_mapped.dtype, device=x_mapped.device)  # (B, L, *, D).
            x_new[mask] = x_mapped
            x = PaddedBatch(x_new, lengths)
        return self.proj(x)


class MetricConditionalHead(torch.nn.Sequential):
    def __init__(self, input_size, output_size, hidden_dims=None,
                 metric_params=None, head_params=None):
        layers = []
        if not hidden_dims:
            layers.append(MetricLayer(input_size, output_size, **(metric_params or {})))
        else:
            layers.append(MetricLayer(input_size, hidden_dims[0], **(metric_params or {})))
            layers.append(ConditionalHead(hidden_dims[0], output_size, hidden_dims=hidden_dims[1:], **(head_params or {})))
        self._output_size = output_size
        super().__init__(*layers)

    @property
    def output_size(self):
        return self._output_size


class TransformerHead(SimpleTransformer):
    def forward(self, x):
        b, l = x.shape
        dummy_timestamps = PaddedBatch(torch.arange(l)[None].float().expand(b, l), x.seq_lens)
        return super().forward(x, dummy_timestamps)[0]
