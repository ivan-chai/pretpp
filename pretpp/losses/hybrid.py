import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss


class HybridLoss(BaseLoss):
    """Combines multiple losses.

    NOTE: Losses must not affect input sequence (masking etc.).

    Args:
        losses: A list of losses.
        prediction_loss: An index of the prediction loss.
        truncate: The type of subsequence selection for losses with different input lengths. Either None, `begin`, or `end`.
    """
    def __init__(self, losses, prediction_loss=None, aggregator=None, truncate=None):
        if any([loss.aggregate for loss in losses]) and (not aggregator):
            raise ValueError("Need aggregator")
        super().__init__()
        self._losses = torch.nn.ModuleList(losses)
        self._prediction_loss = prediction_loss
        self._aggregator = aggregator
        self._truncate = truncate

    @property
    def aggregate(self):
        return False

    @property
    def input_size(self):
        return sum([loss.input_size for loss in self._losses])

    def prepare_inference_batch(self, inputs):
        if self._prediction_loss is not None:
            return self._losses[self._prediction_loss].prepare_inference_batch(inputs)
        for loss in self._losses:
            loss_inputs = loss.prepare_inference_batch(inputs)
            if loss_inputs is not inputs:
                raise RuntimeError("Base losses must not change inputs, when prediction_loss is not provided.")
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
            new_targets[f"_loss_{i}_input_length"] = loss_inputs.shape[1]
            if loss_inputs is not inputs:
                if self._truncate is None:
                    raise RuntimeError("Base losses must not change inputs, when 'truncate' is None.")
                if loss_inputs.shape[1] < inputs.shape[1]:
                    raise RuntimeError("Base losses must not truncate sequences.")
                inputs = loss_inputs
            new_targets[i] = loss_targets
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
        special_token_mask = self.get_special_token_mask(outputs)
        outputs = self.unwrap_model_outputs(outputs)
        losses = {}
        metrics = {}
        offset = 0
        for i, loss in enumerate(self._losses):
            loss_outputs = outputs.payload[..., offset:offset + loss.input_size]
            if special_token_mask is None:
                loss_outputs = PaddedBatch({"outputs": loss_outputs}, outputs.seq_lens)
            else:
                loss_outputs = PaddedBatch({"outputs": loss_outputs, "special_token_mask": special_token_mask}, outputs.seq_lens)
            loss_length = targets[f"_loss_{i}_input_length"]
            if (self._truncate is not None) and (loss_outputs.shape[1] != loss_length):
                assert loss_outputs.payload["outputs"].shape[1] > loss_length
                delta = loss_outputs.payload["outputs"].shape[1] - loss_length
                if self._truncate == "begin":
                    loss_outputs = PaddedBatch({k: v[:, :loss_length] for k, v in loss_outputs.payload.items()},
                                               (loss_outputs.seq_lens - delta).clip(min=0))
                else:
                    assert self._truncate == "end"
                    loss_outputs = PaddedBatch({k: v[:, delta:] for k, v in loss_outputs.payload.items()},
                                               (loss_outputs.seq_lens - delta).clip(min=0))
            if loss.aggregate:
                loss_outputs = PaddedBatch(self._aggregator(self.unwrap_model_outputs(loss_outputs)).unsqueeze(1),
                                           torch.ones_like(loss_outputs.seq_lens))  # (B, 1, D).
            current_losses, current_metrics = loss(loss_outputs, targets[i])
            losses |= {f"loss_{i}_" + k: v for k, v in current_losses.items()}
            metrics |= {f"loss_{i}_" + k: v for k, v in current_metrics.items()}
            offset += loss.input_size
        if offset != outputs.payload.shape[-1]:
            raise RuntimeError("Failed to parse model outputs: dimension mismatch.")
        return losses, metrics

    def predict(self, outputs):
        special_token_mask = self.get_special_token_mask(outputs)
        outputs = self.unwrap_model_outputs(outputs)
        offset = 0
        for i, loss in enumerate(self._losses):
            if i == self._prediction_loss:
                loss_outputs = outputs.payload[..., offset:offset + loss.input_size]
                if special_token_mask is None:
                    loss_outputs = PaddedBatch({"outputs": loss_outputs}, outputs.seq_lens)
                else:
                    loss_outputs = PaddedBatch({"outputs": loss_outputs, "special_token_mask": special_token_mask}, outputs.seq_lens)
                break
            offset += loss.input_size
        else:
            raise RuntimeError(f"Wrong prediction loss index: {self._prediction_loss}")
        loss = self._losses[self._prediction_loss]
        if loss.aggregate:
            loss_outputs = PaddedBatch(self._aggregator(self.unwrap_model_outputs(loss_outputs)).unsqueeze(1),
                                       torch.ones_like(loss_outputs.seq_lens))  # (B, 1, D).
        return loss.predict(loss_outputs)

    def get_prediction_targets(self, targets):
        return targets[self._prediction_loss]
