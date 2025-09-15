from abc import ABC, abstractmethod, abstractproperty
import torch


class BaseLoss(torch.nn.Module):
    @abstractmethod
    def prepare_batch(self, inputs, targets):
        """Extract model inputs and targets.

        Args:
            inputs: Input events with shape (B, L, *).
            targets: Targets with shape (B, L) for local recognition or (B) for global recognition.

        Returns:
            Model inputs with shape (B, L', *) and targets with shape (B, L', *).
        """
        pass

    def prepare_inference_batch(self, inputs):
        """Extract model inputs for inference.

        Args:
            inputs: Input events with shape (B, L, *).

        Returns:
            Model inputs with shape (B, L', *).
        """
        return inputs

    @abstractproperty
    def aggregate(self):
        """The booling flag indicating the need of aggregated input, either True, False, or `both`."""
        pass

    @abstractproperty
    def input_size(self):
        pass

    def compute_metrics(self, inputs, outputs, targets):
        """Evaluate batch metrics.

        Args:
            inputs: Model inputs returned by prepare_batch.
            outputs: Model outputs.
            targets: Targets returned by prepare_batch.

        Returns:
            Metrics dictionary.
        """
        return {}

    def forward(self, targets, outputs=None, embeddings=None):
        """Loss computation method.

        Args:
            targets: Target values, as returned by prepare_batch.
            outputs: Sequential model outputs with shape (B, L, D), when self.aggregate is either False or "both".
            embeddings: Aggregated embeddings with shape (B, D), when self.aggregate is either True or "both".

        Returns:
            A tuple of losses dictionary and metrics dictionary.
        """
        raise NotImplementedError("Please, implement the forward method for the loss.")

    def predict(self, outputs=None, embeddings=None):
        raise NotImplementedError("The loss doesn't support prediction.")

    def get_prediction_targets(self, targets):
        return targets
