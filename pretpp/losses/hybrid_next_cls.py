import torch

from hotpp.data import PaddedBatch
from hotpp.nn import MeanAggregator
from .base import BaseLoss
from .classification import ClassificationLoss
from .next_item import NextItemLoss


class HybridNextClsLoss(BaseLoss):
    """Combines next-item loss with superwised training."""
    def __init__(self, losses, targets, aggregator):
        super().__init__()
        self._next_item = NextItemLoss(losses=losses)
        self._cls = ClassificationLoss(targets=targets)
        self._aggregator = aggregator

    @property
    def aggregate(self):
        return False

    @property
    def input_size(self):
        return self._next_item.input_size + self._cls.input_size

    def prepare_batch(self, inputs, targets=None):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets (unused): Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        # For the next-item loss inputs and targets are the same.
        # Offset is applied in base loss classes.
        return inputs, {"next_item": self._next_item.prepare_batch(inputs, targets)[1],
                        "cls": self._cls.prepare_batch(inputs, targets)[1]}

    def forward(self, outputs, targets):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model outputs with shape (B, L, *, D) or (B, 1, *, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L, *).
            reduction: `mean` or `none`.

        Returns:
            Losses dict and metrics dict.
        """
        next_item_outputs = PaddedBatch(outputs.payload[..., :self._next_item.input_size], outputs.seq_lens)
        next_item_losses, next_item_metrics = self._next_item(next_item_outputs, targets["next_item"])
        cls_outputs = PaddedBatch(outputs.payload[..., self._next_item.input_size:], outputs.seq_lens)
        agg_outputs = PaddedBatch(self._aggregator(cls_outputs).unsqueeze(1),
                                  torch.ones_like(cls_outputs.seq_lens))  # (B, 1, D).
        cls_losses, cls_metrics = self._cls(agg_outputs, targets["cls"])

        next_item_losses = {"next_item_" + k: v for k, v in next_item_losses.items()}
        cls_losses = {"cls_" + k: v for k, v in cls_losses.items()}
        next_item_metrics = {"next_item_" + k: v for k, v in next_item_metrics.items()}
        cls_metrics = {"cls_" + k: v for k, v in cls_metrics.items()}
        return next_item_losses | cls_losses, next_item_metrics | cls_metrics

    def predict(self, outputs):
        cls_outputs = PaddedBatch(outputs.payload[..., self._next_item.input_size:], outputs.seq_lens)
        agg_outputs = PaddedBatch(self._aggregator(cls_outputs).unsqueeze(1),
                                  torch.ones_like(cls_outputs.seq_lens))  # (B, 1, D).
        return self._cls.predict(agg_outputs)
