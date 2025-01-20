import torch


class DownstreamMetric:
    """Finetuned model evaluation on downstream tasks.

    Args:
        classification_fields: The fields to compute accuracy for.
    """
    def __init__(self, classification_fields=None):
        self.classification_fields = classification_fields
        self.reset()

    def reset(self):
        self.n = {}
        self.n_correct = {}
        for field in self.classification_fields:
            self.n[field] = 0
            self.n_correct[field] = 0

    def update(self, predictions, targets):
        """Update metric.

        Args:
            Predictions: PaddedBatch with predictions.
            Predictions: PaddedBatch with targets.
        """
        for field in self.classification_fields:
            prediction = predictions.payload[field]  # (B).
            target = targets.payload[field]  # (B).
            if (prediction.ndim != 1) or (target.ndim != 1):
                raise NotImplementedError("Only global classification is implemented.")
            self.n[field] += len(target)
            self.n_correct[field] += (prediction == target).sum().item()

    def compute(self):
        metrics = {}
        for field in self.classification_fields:
            metrics[f"accuracy-{field}"] = self.n_correct[field] / max(1, self.n[field])
        return metrics
