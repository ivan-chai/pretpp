import torch
from hotpp.data import PaddedBatch


class MetricLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features,
                 normalize_embedding=True, normalize_centroids=True,
                 scoring="dot", activation="none"):
        super().__init__(in_features, out_features, bias=False)
        self.normalize_embedding = normalize_embedding
        self.normalize_centroids = normalize_centroids
        self.scoring = scoring
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
        payload = input.payload
        if self.normalize_embedding:
            payload = torch.nn.functional.normalize(payload, dim=-1)
        return PaddedBatch(payload, input.seq_lens)

    def forward(self, input):
        weight = self.weight  # (D', D).
        if self.normalize_centroids:
            weight = torch.nn.functional.normalize(weight, dim=1)
        payload = self.encode(input).payload  # (B, L, D).
        if self.scoring == "dot":
            payload = torch.nn.functional.linear(payload, weight)
        elif self.scoring == "l2":
            payload = torch.nn.functional.linear(payload, weight)  # (B, L, D').
            pnorms2 = torch.linalg.norm(payload, dim=-1, keepdim=True).square()  # (B, L, 1).
            wnorms2 = torch.linalg.norm(weight, dim=-1).square()  # (D').
            payload = (- 2 * payload + wnorms2 + pnorms2).sqrt()  # (B, L, D').
        else:
            raise ValueError(f"Unknown scoring: f{self.scoring}")
        payload = self.activation(payload)
        return PaddedBatch(payload, input.seq_lens)
