import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import Metric


class DownstreamMetric(Metric):
    """Finetuned model evaluation on downstream tasks.

    Args:
        classification_fields: The fields to compute accuracy for.
    """
    def __init__(self, classification_fields=None, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.classification_fields = classification_fields
        self.reset()

    def reset(self):
        for field in self.classification_fields:
            # Accuracy.
            self.add_state(f"n_{field}", default=0, dist_reduce_fx="sum")
            self.add_state(f"n_correct_{field}", default=0, dist_reduce_fx="sum")
            # ROC.
            self.add_state(f"labels_{field}", default=[], dist_reduce_fx="cat")
            self.add_state(f"scores_{field}", default=[], dist_reduce_fx="cat")

    def update(self, predictions, scores, targets):
        """Update metric.

        Args:
            predictions: PaddedBatch with predictions (B).
            scores: PaddedBatch with predicted class scores (B, C).
            targets: PaddedBatch with targets (B).
        """
        for field in self.classification_fields:
            target = targets.payload[field]  # (B).
            # Update accuracy.
            prediction = predictions.payload[field]  # (B).
            if (prediction.ndim != 1) or (target.ndim != 1):
                raise NotImplementedError("Only global classification is implemented.")
            attr = f"n_{field}"
            setattr(self, attr, getattr(self, attr) + len(target))
            attr = f"n_correct_{field}"
            setattr(self, attr, getattr(self, attr) + (prediction == target).sum().item())
            # Update ROC.
            score = scores.payload[field]  # (B).
            getattr(self, f"labels_{field}").append(target.cpu())
            getattr(self, f"scores_{field}").append(score.cpu())

    def compute(self):
        metrics = {}
        for field in self.classification_fields:
            metrics[f"accuracy-{field}"] = getattr(self, f"n_correct_{field}") / max(1, getattr(self, f"n_{field}"))
            labels = torch.cat(getattr(self, f"labels_{field}"), 0).numpy()  # (B).
            scores = torch.cat(getattr(self, f"scores_{field}"), 0).numpy()  # (B, C).
            metrics[f"macro-auc-{field}"] = roc_auc_score(labels, scores)
        return metrics
