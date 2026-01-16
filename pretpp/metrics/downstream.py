import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score


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
        self.classes = {}
        self.class_scores = {}
        for field in self.classification_fields:
            self.n[field] = 0
            self.n_correct[field] = 0
            self.classes[field] = list()
            self.class_scores[field] = list()

    def update(self, predictions, scores, targets):
        """Update metric.

        Args:
            Predictions: PaddedBatch with predictions.
            Predictions: PaddedBatch with targets.
        """
        for field in self.classification_fields:
            prediction = predictions.payload[field]  # (B).
            target = targets.payload[field]  # (B).
            score = scores.payload[field]  # (B, C).
            if (prediction.ndim != 1) or (target.ndim != 1):
                raise NotImplementedError("Only global classification is implemented.")
            self.n[field] += len(target)
            self.n_correct[field] += (prediction == target).sum().item()
            self.classes[field].append(target.cpu())
            self.class_scores[field].append(score.cpu())

    def compute(self):
        metrics = {}
        for field in self.classification_fields:
            metrics[f"accuracy-{field}"] = self.n_correct[field] / max(1, self.n[field])
            s = torch.cat(self.class_scores[field])
            y = torch.nn.functional.one_hot(torch.cat(self.classes[field]).long(), s.shape[1])
            metrics[f"macro-roc-auc-{field}"] = roc_auc_score(y.numpy(), s.numpy())
        return metrics
