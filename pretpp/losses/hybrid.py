import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class HybridLoss(BaseLoss):
    """Combines multiple losses.

    NOTE: Losses must not affect input sequence (masking etc.).

    Args:
        losses: A mapping from the loss name to the loss object.
        prediction_loss: A name of the loss for prediction.
        drop_nans: Exclude nan targets from classification.
    """
    def __init__(self, losses, prediction_loss=None, aggregator=None):
        if any([loss.aggregate for loss in losses.values()]) and (not aggregator):
            raise ValueError("Need aggregator")
        if (prediction_loss is not None) and (prediction_loss not in losses):
            raise ValueError(f"Unknown prediction loss: {prediction_loss}")
        super().__init__()
        self._order = list(sorted(losses))
        self._losses = torch.nn.ModuleDict(losses)
        self._prediction_loss = prediction_loss
        self._aggregator = aggregator

    @property
    def aggregate(self):
        return False

    @property
    def input_size(self):
        return sum([loss.input_size for loss in self._losses.values()])

    def prepare_inference_batch(self, inputs):
        for loss in self._losses.values():
            loss_inputs = loss.prepare_inference_batch(inputs)
            if loss_inputs is not inputs:
                raise RuntimeError("Base losses must not change inputs.")
        return inputs

    def prepare_batch(self, inputs, targets=None):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets (unused): Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        new_targets = {}
        for i, (name, loss) in enumerate(self._losses.items()):
            loss_inputs, loss_targets = loss.prepare_batch(inputs, targets)
            if loss_inputs is not inputs:
                raise RuntimeError("Base losses must not change inputs.")
            new_targets[name] = loss_targets
        return inputs, new_targets

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
        losses = {}
        metrics = {}
        offset = 0
        for name in self._order:
            loss = self._losses[name]
            loss_outputs = PaddedBatch(outputs.payload[..., offset:offset + loss.input_size], outputs.seq_lens)
            if loss.aggregate:
                loss_outputs = PaddedBatch(self._aggregator(loss_outputs).unsqueeze(1),
                                           torch.ones_like(loss_outputs.seq_lens))  # (B, 1, D).
            current_losses, current_metrics = loss(loss_outputs, targets[name])
            losses |= {name + "_" + k: v for k, v in current_losses.items()}
            metrics |= {name + "_" + k: v for k, v in current_metrics.items()}
            offset += loss.input_size
        if offset != outputs.payload.shape[-1]:
            raise RuntimeError("Failed to parse model outputs: dimension mismatch.")
        return losses, metrics

    def predict(self, outputs):
        offset = 0
        for name in self._order:
            if name == self._prediction_loss:
                loss_outputs = PaddedBatch(outputs.payload[..., offset:offset + loss.input_size], outputs.seq_lens)
                break
            loss = self._losses[name]
            offset += loss.input_size
        else:
            assert False
        loss = self._losses[self._prediction_loss]
        if loss.aggregate:
            loss_outputs = PaddedBatch(self._aggregator(loss_outputs).unsqueeze(1),
                                       torch.ones_like(loss_outputs.seq_lens))  # (B, 1, D).
        return loss.predict(loss_outputs)
