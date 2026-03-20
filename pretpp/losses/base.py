from abc import ABC, abstractmethod, abstractproperty
import torch
from hotpp.data import PaddedBatch


def recursive_map(data, func):
    """Recursively applies a function to all leaf nodes in a structure."""
    if isinstance(data, dict):
        return {k: recursive_map(v, func) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return type(data)(recursive_map(item, func) for item in data)
    return func(data)


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

    def predict(self, outputs):
        raise NotImplementedError("The loss doesn't support prediction.")

    def get_prediction_targets(self, targets):
        return targets

    @staticmethod
    def unwrap_model_outputs(outputs):
        """Extract tensor-based model outputs from dictionary-variant PaddedBatch."""
        if not isinstance(outputs.payload, dict):
            return outputs
        if "outputs" not in outputs.payload:
            raise ValueError("Dictionary model outputs must contain 'outputs' field.")
        return PaddedBatch(outputs.payload["outputs"], outputs.seq_lens)

    @staticmethod
    def get_special_token_mask(outputs):
        """Extract special token mask from a dictionary-based PaddedBatch."""
        if not isinstance(outputs.payload, dict):
            return None
        return outputs.payload.get("special_token_mask", None)

    @staticmethod
    def select_embeddings_by_mask(embeddings, mask):
        if mask.ndim != 2:
            raise ValueError("Expected mask with shape (B, L).")
        b, l = mask.shape
        seq_len_mask = embeddings.seq_len_mask.bool()  # (B, L).
        mask = torch.logical_and(mask.bool(), seq_len_mask)  # (B, L).
        mask_lengths = mask.sum(1)  # (B).
        max_length = mask_lengths.max().item()

        # Equalize the number of elements in each raw.
        mask = torch.logical_or(mask, ~seq_len_mask)  # (B, L).
        mask = torch.logical_and(mask, mask.cumsum(1) <= max_length)  # (B, L).

        selected = embeddings.payload[mask].reshape(b, max_length, embeddings.payload.shape[2])  # (B, O, D).
        return PaddedBatch(selected, mask_lengths)
