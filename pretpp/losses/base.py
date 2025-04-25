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
        """The booling flag indicating input is an embedding rather than a sequence."""
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
