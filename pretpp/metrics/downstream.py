import torch
from torchmetrics import Accuracy, AUROC


class DownstreamMetric(torch.nn.Module):
    """Finetuned model evaluation on downstream tasks.

    Args:
        classification_fields: The fields to compute accuracy for.
    """
    def __init__(self, classification_fields):
        super().__init__()
        self.classification_fields = classification_fields
        self.initialized = False
        self.reset()

    def _initialize_metrics(self, num_classes, device):
        assert len(num_classes) == len(self.classification_fields)
        for field, num_classes in zip(self.classification_fields, num_classes):
            setattr(self, f"accuracy_{field}", Accuracy(task="multiclass", num_classes=num_classes).to(device))
            setattr(self, f"auroc_{field}", AUROC(task="multiclass", num_classes=num_classes, average="macro").to(device))
        self.initialized = True

    def reset(self):
        if not self.initialized:
            return
        for field in self.classification_fields:
            delattr(self, f"accuracy_{field}")
            delattr(self, f"auroc_{field}")
        self.initialized = False

    def update(self, predictions, targets):
        """Update metric.

        Args:
            Predictions: PaddedBatch with predictions and logits.
            Predictions: PaddedBatch with targets.
        """
        if not self.initialized:
            num_classes = [predictions.payload[f"logits-{field}"].shape[1] for field in self.classification_fields]
            self._initialize_metrics(num_classes, predictions.device)
        for field in self.classification_fields:
            target = targets.payload[field]
            target_mask = target.isfinite()
            if not target_mask.any():
                continue
            target = target[target_mask].long()  # (B).

            # Update accuracy.
            prediction = predictions.payload[field][target_mask]  # (B).
            if (prediction.ndim != 1) or (target.ndim != 1):
                raise NotImplementedError("Only global classification is implemented.")
            getattr(self, f"accuracy_{field}").update(prediction, target)

            # Update ROC.
            logits = predictions.payload[f"logits-{field}"][target_mask]  # (B, C).
            getattr(self, f"auroc_{field}").update(logits, target)

    def compute(self):
        if not self.initialized:
            return {}
        metrics = {}
        for field in self.classification_fields:
            metrics[f"accuracy-{field}"] = getattr(self, f"accuracy_{field}").compute()
            metrics[f"macro-auc-{field}"] = getattr(self, f"auroc_{field}").compute()
        return metrics
