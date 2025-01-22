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

    def forward(self, outputs, targets):
        """Compute loss and metrics.

        Args:
            outputs: Model outputs with shape (B, L, *, D) or (B, 1, *, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L, *).

        Returns:
            Losses dict and metrics dict.
        """
        return super().forward(targets, outputs, states=None)
