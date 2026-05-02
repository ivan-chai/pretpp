import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss, recursive_map


class HybridLoss(BaseLoss):
    """Combines multiple losses.

    NOTE: Losses must not affect input sequence (masking etc.).

    Args:
        losses: A list of losses.
        prediction_loss: An index of the prediction loss.
    """
    def __init__(self, losses, prediction_loss=None, aggregator=None):
        if any([loss.aggregate for loss in losses]) and (not aggregator):
            raise ValueError("Need aggregator")
        for loss in losses:
            if loss.uses_special_tokens_inside:
                raise RuntimeError("Can't stack losses, that insert special tokens inside.")
        super().__init__()
        self._losses = torch.nn.ModuleList(losses)
        self._prediction_loss = prediction_loss
        self._aggregator = aggregator

    @property
    def structure(self):
        result = []
        for i, loss in enumerate(self._losses):
            if hasattr(loss, "structure"):
                result.append(recursive_map(loss.structure, lambda name: f"loss_{i}_{name}"))
            else:
                raise NotImplementedError(f"Unknown structure for {loss}")
        return result

    @property
    def aggregate(self):
        return False

    @property
    def input_size(self):
        return sum([loss.input_size for loss in self._losses])

    @property
    def special_tokens_start(self):
        """The number of special tokens at the beginning."""
        return sum([loss.special_tokens_start for loss in self._losses])

    @property
    def special_tokens_end(self):
        """The number of special tokens at the beginning."""
        return sum([loss.special_tokens_end for loss in self._losses])

    @property
    def uses_special_tokens_inside(self):
        """Whether the loss uses special tokens except start/end."""
        return any([loss.uses_special_tokens_inside for loss in self._losses])

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
            loss_inputs, loss_targets, loss_global_targets = loss.prepare_batch(inputs, targets)
            if (i > 0) and (loss_inputs.shape[1] != inputs.shape[1] + loss.special_tokens_start + loss.special_tokens_end):
                raise RuntimeError(f"Base losses (except the first one) must not truncate sequences.")
            inputs = loss_inputs
            if targets is None:
                pass
            elif i > 0:
                # Can only subset fields.
                for k, v in loss_global_targets.payload.items():
                    if v is not targets.payload[k]:
                        raise RuntimeError(f"Ony the first loss can change global targets ({i})")
            targets = loss_global_targets
            new_targets[i] = loss_targets
        return inputs, new_targets, targets

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
        offset = outputs.payload.shape[-1]
        for i, loss in reversed(list(enumerate(self._losses))):
            loss_outputs = outputs.payload[..., offset - loss.input_size:offset]
            if special_token_mask is None:
                loss_outputs = PaddedBatch({"outputs": loss_outputs}, outputs.seq_lens)
            else:
                loss_outputs = PaddedBatch({"outputs": loss_outputs, "special_token_mask": special_token_mask}, outputs.seq_lens)
            if loss.aggregate:
                loss_outputs = PaddedBatch(self._aggregator(self.unwrap_model_outputs(loss_outputs)).unsqueeze(1),
                                           torch.ones_like(loss_outputs.seq_lens))  # (B, 1, D).
            current_losses, current_metrics = loss(loss_outputs, targets[i])
            losses |= {f"loss_{i}_" + k: v for k, v in current_losses.items()}
            metrics |= {f"loss_{i}_" + k: v for k, v in current_metrics.items()}

            # Remove special tokens of the current loss.
            if loss.special_tokens_start > 0 or loss.special_tokens_end > 0:
                start = loss.special_tokens_start
                end = outputs.shape[1] - loss.special_tokens_end
                deleted = loss.special_tokens_start + loss.special_tokens_end
                outputs = PaddedBatch({k: v[:, start:end] for k, v in outputs.payload.items()},
                                      (outputs.seq_lens - deleted).clip(min=0))
                if special_token_mask is not None:
                    special_token_mask = special_token_mask[:, start:end]

            offset -= loss.input_size
        if offset != 0:
            raise RuntimeError("Failed to parse model outputs: dimension mismatch.")
        return losses, metrics

    def predict(self, outputs):
        if self._prediction_loss is None:
            raise ValueError("Need prediction loss index")
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
