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
        # TODO: DEBUG CHECK BATCH NORM CONSISTENCY BETWEEN PRETRAIN / SFT / EMBED.
        #if head_params and head_params.get("use_batch_norm", False):
        #    self.bn = torch.nn.BatchNorm1d(input_size)
        #else:
        #    self.bn = None
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


class CatHead(torch.nn.ModuleList):
    """Concatenate outputs of multiple heads."""
    def __init__(self, input_size, output_size=None, head_partials=None, infer_output_size_index=None):
        if head_partials is None:
            raise ValueError("Need head_partials")
        heads = []
        total_size = 0
        for i, partial in enumerate(head_partials):
            if i == infer_output_size_index:
                heads.append(None)
            else:
                heads.append(partial(input_size))
                total_size += heads[-1].output_size
        if infer_output_size_index is not None:
            if output_size is None:
                raise ValueError("Need output size to infer a missing size")
            heads[infer_output_size_index] = head_partials[infer_output_size_index](input_size, output_size=output_size - total_size)
            total_size = output_size
        if (output_size is not None) and (total_size != output_size):
            raise ValueError(f"Expected output size {output_size}, got {total_size}")
        super().__init__(heads)

    @property
    def output_size(self):
        return sum([h.output_size for h in self])

    def forward(self, x):
        outputs = [h(x) for h in self]
        for output in outputs:
            if (output.seq_lens != x.seq_lens).any():
                raise RuntimeError("Heads output lengths mismatch.")
        return PaddedBatch(torch.cat([output.payload for output in outputs], -1), x.seq_lens)


class StackHead(torch.nn.Sequential):
    """Combine multiple heads vertically."""
    def __init__(self, input_size, output_size=None, head_partials=None):
        if head_partials is None:
            raise ValueError("Need head_partials")
        heads = []
        for i, partial in enumerate(head_partials):
            if i < len(head_partials) - 1:
                heads.append(partial(input_size))
                input_size = heads[-1].output_size
            else:
                heads.append(partial(input_size, output_size=output_size))
        super().__init__(*heads)

    @property
    def output_size(self):
        return self[-1].output_size
