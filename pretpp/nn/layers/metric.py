import torch
from hotpp.data import PaddedBatch


class MetricLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, activation="log_softmax"):
        super().__init__(in_features, out_features, bias=False)
        if activation == "none":
            self.activation = torch.nn.Identity()
        elif activation == "log_softmax":
            self.activation = torch.nn.LogSoftmax(-1)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    @property
    def output_size(self):
        return len(self.weight)

    def encode(self, input):
        payload = torch.nn.functional.normalize(input.payload, dim=-1)
        return PaddedBatch(payload, input.seq_lens)

    def forward(self, input):
        weight = torch.nn.functional.normalize(self.weight, dim=1)
        payload = torch.nn.functional.linear(self.encode(input).payload, weight)
        return PaddedBatch(payload, input.seq_lens)
