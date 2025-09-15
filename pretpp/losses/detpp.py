from hotpp.losses import DetectionLoss
from .base import BaseLoss


class DeTPPLoss(DetectionLoss, BaseLoss):
    """Wrapper around HoTPP Detection Loss."""
    @property
    def aggregate(self):
        return False

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
        return inputs, inputs

    def forward(self, targets, outputs=None, embeddings=None):
        """Compute loss and metrics.

        Args:
            targets: Target values, as returned by prepare_batch.
            outputs: Sequential model outputs with shape (B, L, D), when self.aggregate is either False or "both".
            embeddings (unused): Aggregated embeddings with shape (B, D), when self.aggregate is either True or "both".

        Returns:
            Losses dict and metrics dict.
        """
        assert outputs is not None
        return super().forward(targets, outputs, states=None)
