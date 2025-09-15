import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class HybridLoss(BaseLoss):
    """Combines multiple losses.

    NOTE: Losses must not affect input sequence (masking etc.).

    Args:
        losses: A list of losses.
        prediction_loss: An index of the prediction loss.
    """
    def __init__(self, losses, prediction_loss=None):
        super().__init__()
        self._losses = torch.nn.ModuleList(losses)
        self._prediction_loss = prediction_loss

    @property
    def aggregate(self):
        return "both"

    @property
    def input_size(self):
        return sum([loss.input_size for loss in self._losses])

    def prepare_inference_batch(self, inputs):
        for loss in self._losses:
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
        for i, loss in enumerate(self._losses):
            loss_inputs, loss_targets = loss.prepare_batch(inputs, targets)
            if loss_inputs is not inputs:
                raise RuntimeError("Base losses must not change inputs.")
            new_targets[i] = loss_targets
        return inputs, new_targets

    def forward(self, targets, outputs=None, embeddings=None):
        """Extract targets and compute loss between predictions and targets.

        Args:
            targets: Target values, as returned by prepare_batch.
            outputs: Sequential model outputs with shape (B, L, D), when self.aggregate is either False or "both".
            embeddings: Aggregated embeddings with shape (B, D), when self.aggregate is either True or "both".

        Returns:
            Losses dict and metrics dict.
        """
        losses = {}
        metrics = {}
        offset = 0
        for i, loss in enumerate(self._losses):
            if i not in targets:
                offset += loss.input_size
                continue
            if outputs is not None:
                loss_outputs = PaddedBatch(outputs.payload[..., offset:offset + loss.input_size],
                                           outputs.seq_lens)
            else:
                loss_outputs = None
            if embeddings is not None:
                loss_embeddings = embeddings[..., offset:offset + loss.input_size]
            else:
                loss_embeddings = None
            current_losses, current_metrics = loss(targets[i], outputs=loss_outputs, embeddings=loss_embeddings)
            losses |= {f"loss_{i}_" + k: v for k, v in current_losses.items()}
            metrics |= {f"loss_{i}_" + k: v for k, v in current_metrics.items()}
            offset += loss.input_size
        gt_dim = (outputs.payload if outputs is not None else embeddings).shape[-1]
        if offset != gt_dim:
            raise RuntimeError("Failed to parse model outputs: dimension mismatch.")
        return losses, metrics

    def predict(self, outputs=None, embeddings=None):
        offset = 0
        for i, loss in enumerate(self._losses):
            if i == self._prediction_loss:
                if outputs is not None:
                    loss_outputs = PaddedBatch(outputs.payload[..., offset:offset + loss.input_size], outputs.seq_lens)
                if embeddings is not None:
                    loss_embeddings = embeddings[..., offset:offset + loss.input_size]
                break
            offset += loss.input_size
        else:
            raise RuntimeError(f"Wrong prediction loss index: {self._prediction_loss}")
        loss = self._losses[self._prediction_loss]
        return loss.predict(outputs=loss_outputs, embeddings=loss_embeddings)

    def get_prediction_targets(self, targets):
        return targets[self._prediction_loss]
