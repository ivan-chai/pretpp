#!/usr/bin/env python3
from unittest import TestCase, main, mock

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder.history_token_transformer import HistoryTokenTransformer


class _DummyStrategy:
    def __init__(self, extracted, predict, apply_to_batch=True):
        self._extracted = extracted
        self.predict = predict
        self.apply_to_batch = apply_to_batch

    def __call__(self, timestamps):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def apply(self, x, timestamps):
        return x, timestamps, None

    def extract_outputs(self, x):
        return self._extracted


class TestHistoryTokenTransformer(TestCase):
    def _make_model(self, strategy):
        def _remove_sos(x):
            # Simulate sos=None: no-op regardless of payload type.
            return x

        model = object.__new__(HistoryTokenTransformer)
        model.training = True
        model.sos = None
        model.input_projection = lambda x: x
        model.add_sos = lambda x, timestamps: (x, timestamps)
        model.strategy = strategy
        model.positional = lambda x, timestamps: x
        model.rope = None
        model.transform = lambda x, attention_mask=None: (x, None)
        model.remove_sos = mock.Mock(side_effect=_remove_sos)
        return model

    def test_forward_does_not_remove_sos_for_history_token_outputs(self):
        inputs = PaddedBatch(torch.randn(2, 4, 3), torch.tensor([4, 3]))
        timestamps = PaddedBatch(torch.randn(2, 4), torch.tensor([4, 3]))
        # predict=="history_tokens" → extract_outputs returns dict-payload PaddedBatch.
        extracted = PaddedBatch(
            {"outputs": torch.randn(2, 2, 3),
             "special_token_mask": torch.ones(2, 2, dtype=torch.bool)},
            torch.tensor([2, 1])
        )
        strategy = _DummyStrategy(extracted=extracted, predict="history_tokens")
        model = self._make_model(strategy)

        outputs, _ = model.forward(inputs, timestamps)

        # remove_sos is always called (sos=None makes it a no-op).
        model.remove_sos.assert_called_once()
        # Dict payload is preserved.
        self.assertIsInstance(outputs.payload, dict)
        self.assertIn("special_token_mask", outputs.payload)

    def test_forward_removes_sos_for_input_token_outputs(self):
        inputs = PaddedBatch(torch.randn(2, 4, 3), torch.tensor([4, 3]))
        timestamps = PaddedBatch(torch.randn(2, 4), torch.tensor([4, 3]))
        # predict=="input_tokens" → extract_outputs returns plain PaddedBatch.
        extracted = PaddedBatch(torch.randn(2, 4, 3), torch.tensor([4, 3]))
        strategy = _DummyStrategy(extracted=extracted, predict="input_tokens")
        model = self._make_model(strategy)

        outputs, _ = model.forward(inputs, timestamps)

        model.remove_sos.assert_called_once()
        self.assertNotIsInstance(outputs.payload, dict)

    def test_forward_does_not_remove_sos_for_all_outputs(self):
        inputs = PaddedBatch(torch.randn(1, 3, 2), torch.tensor([3]))
        timestamps = PaddedBatch(torch.randn(1, 3), torch.tensor([3]))
        # predict=="all" → extract_outputs returns dict payload with special_token_mask.
        special_token_mask = torch.tensor([[False, True, False, True, False]])
        extracted = PaddedBatch(
            {"outputs": torch.randn(1, 5, 2), "special_token_mask": special_token_mask},
            torch.tensor([5])
        )
        strategy = _DummyStrategy(extracted=extracted, predict="all")
        model = self._make_model(strategy)

        outputs, _ = model.forward(inputs, timestamps)

        # remove_sos is always called (sos=None makes it a no-op).
        model.remove_sos.assert_called_once()
        self.assertIsInstance(outputs.payload, dict)
        self.assertIn("special_token_mask", outputs.payload)
        self.assertEqual(outputs.payload["special_token_mask"].tolist(), special_token_mask.tolist())


if __name__ == "__main__":
    main()
