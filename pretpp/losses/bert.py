import torch

from hotpp.data import PaddedBatch
from .base import BaseLoss
from .mlm import MLMLoss


COLES_TARGET_FIELD = "_coles_target"


class BERTLoss(BaseLoss):
    def __init__(self, mlm_loss, coles_loss):
        super().__init__()
        self.mlm_loss = mlm_loss
        self.coles_loss = coles_loss
        if self.coles_loss.aggregate:
            raise ValueError("CoLES Loss must use CLS token.")
        if not self.coles_loss.cls_token_begin:
            raise ValueError("CoLES Loss must put CLS token at the beginning.")

    @property
    def aggregate(self):
        return False

    @property
    def input_size(self):
        return self.mlm_loss.input_size

    def prepare_batch(self, inputs, targets=None):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets (unused): Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        splits, coles_targets = self.coles_loss.prepare_batch(inputs, targets=targets)
        # Exclude CLS token and apply masking.
        splits_truncated = PaddedBatch({k: splits.payload[k][:, 1:] for k in splits.seq_names},
                                       (splits.seq_lens - 1).clip(min=0))
        mlm_inputs, targets = self.mlm_loss.prepare_batch(splits_truncated)

        # Revert CLS token and built outputs.
        model_inputs = {k: inputs.payload[k] for k in inputs.payload if k not in inputs.seq_names}
        for k, v in mlm_inputs.payload.items():
            if k == self.mlm_loss.timedeltas_field:
                model_inputs[k] = torch.cat([torch.zeros_like(v[:, :1]), v], 1)
            else:
                model_inputs[k] = torch.cat([splits.payload[k][:, :1], v], 1)
        model_inputs = PaddedBatch(model_inputs, mlm_inputs.seq_lens + 1, seq_names=mlm_inputs.seq_names)
        targets.payload[COLES_TARGET_FIELD] = coles_targets
        return model_inputs, targets

    def prepare_inference_batch(self, inputs):
        return self.coles_loss.prepare_inference_batch(inputs)

    def forward(self, outputs, targets):
        """Extract targets and compute loss between predictions and targets.

        Args:
            outputs: Model outputs with shape (B, L, *, D) or (B, 1, *, D).
                Outputs can be dictionary with predictions for particular fields.
            targets: Target features with shape (B, L, *).

        Returns:
            Losses dict and metrics dict.
        """
        payload = dict(targets.payload)
        coles_targets = payload.pop(COLES_TARGET_FIELD)
        assert coles_targets is not None
        outputs_truncated = PaddedBatch(outputs.payload[:, 1:], (outputs.seq_lens - 1).clip(min=0))
        mlm_targets = PaddedBatch(payload, targets.seq_lens, seq_names=targets.seq_names)

        mlm_losses, mlm_metrics = self.mlm_loss(outputs_truncated, mlm_targets)
        coles_losses, coles_metrics = self.coles_loss(outputs, coles_targets)
        losses = {"mlm-" + k: v for k, v in mlm_losses.items()} | {"coles-" + k: v for k, v in coles_losses.items()}
        metrics = {"mlm-" + k: v for k, v in mlm_metrics.items()} | {"coles-" + k: v for k, v in coles_metrics.items()}
        return losses, metrics
