import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import Metric


class DownstreamMetric(Metric):
    """Finetuned model evaluation on downstream tasks.

    Args:
        classification_fields: The fields to compute accuracy for.
    """
    def __init__(self, classification_fields, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.classification_fields = classification_fields
        self.reset()

    def reset(self):
        for field in self.classification_fields:
            # Accuracy.
            self.add_state(f"n_{field}", default=torch.zeros([], dtype=torch.long), dist_reduce_fx="sum")
            self.add_state(f"n_correct_{field}", default=torch.zeros([], dtype=torch.long), dist_reduce_fx="sum")
            # ROC.
            self.add_state(f"labels_{field}", default=[], dist_reduce_fx="cat")
            self.add_state(f"scores_{field}", default=[], dist_reduce_fx="cat")

    def update(self, predictions, targets):
        """Update metric.

        Args:
            Predictions: PaddedBatch with predictions and logits.
            Predictions: PaddedBatch with targets.
        """
        for field in self.classification_fields:
            target = targets.payload[field]  # (B).
            target_mask = target.isfinite()  # (B).
            if not target_mask.any():
                continue
            target = target[target_mask]

            # Update accuracy.
            prediction = predictions.payload[field][target_mask]  # (B).
            if (prediction.ndim != 1) or (target.ndim != 1):
                raise NotImplementedError("Only global classification is implemented.")
            attr = f"n_{field}"
            setattr(self, attr, getattr(self, attr) + len(target))
            attr = f"n_correct_{field}"
            setattr(self, attr, getattr(self, attr) + (prediction == target).sum().item())

            # Update ROC.
            logits_name = f"logits-{field}"
            if logits_name in predictions.payload:
                logits = predictions.payload[logits_name][target_mask]  # (B, C).
                getattr(self, f"labels_{field}").append(target)
                getattr(self, f"scores_{field}").append(logits.float())

    def compute(self):
        metrics = {}
        for field in self.classification_fields:
            metrics[f"accuracy-{field}"] = getattr(self, f"n_correct_{field}") / max(1, getattr(self, f"n_{field}"))
            scores = getattr(self, f"scores_{field}").cpu().numpy()  # (B, C).
            labels = torch.nn.functional.one_hot(getattr(self, f"labels_{field}").long(), scores.shape[1]).cpu().numpy()  # (B).
            metrics[f"macro-auc-{field}"] = roc_auc_score(labels, scores, average="macro", multi_class="ovr")
        return metrics
