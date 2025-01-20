import torch
from hotpp.data import PaddedBatch
from hotpp.nn import Head
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


class MetricHead(torch.nn.Sequential):
    def __init__(self, input_size, output_size, hidden_dims=None,
                 metric_params=None, head_params=None):
        layers = []
        if not hidden_dims:
            layers.append(MetricLayer(input_size, output_size, **(metric_params or {})))
        else:
            layers.append(MetricLayer(input_size, hidden_dims[0], **(metric_params or {})))
            layers.append(Head(hidden_dims[0], output_size, hidden_dims=hidden_dims[1:], **(head_params or {})))
        self._output_size = output_size
        super().__init__(*layers)

    @property
    def output_size(self):
        return self._output_size
