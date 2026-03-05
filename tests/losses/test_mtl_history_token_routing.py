#!/usr/bin/env python3
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.losses import MAELoss
from pretpp.losses import ClassificationLoss, NextItemLoss


class TestMTLHistoryTokenRouting(TestCase):
    def test_classification_uses_only_history_tokens(self):
        loss = ClassificationLoss(
            targets={"target": {"num_classes": 2, "cast": True}},
            apply_to_tokens="special",
        )

        logits = torch.tensor([
            [
                [10.0, 0.0],
                [0.0, 10.0],
                [10.0, 0.0],
                [0.0, 10.0],
            ]
        ])
        outputs = PaddedBatch(
            {"outputs": logits, "special_token_mask": torch.tensor([[0, 1, 0, 1]], dtype=torch.bool)},
            torch.tensor([4]),
            seq_names={"outputs", "special_token_mask"},
        )
        targets = PaddedBatch({"target": torch.tensor([1.0])}, torch.tensor([4]), seq_names=set())

        losses, metrics = loss(outputs, targets)
        expected = torch.nn.functional.cross_entropy(logits[:, [1, 3]].flatten(0, -2), torch.tensor([1, 1]))

        self.assertAlmostEqual(losses["target"].item(), expected.item(), places=6)
        self.assertAlmostEqual(metrics["batch-accuracy-target"].item(), 1.0, places=6)

    def test_next_item_uses_only_regular_tokens(self):
        loss = NextItemLoss(
            losses={"value": MAELoss()},
            apply_to_tokens="regular",
        )

        inputs = PaddedBatch({"value": torch.tensor([[1.0, 2.0, 3.0]])}, torch.tensor([3]), seq_names={"value"})
        outputs = PaddedBatch(
            {
                "outputs": torch.tensor([[[2.0], [99.0], [3.0], [99.0], [0.0]]]),
                "special_token_mask": torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.bool),
            },
            torch.tensor([5]),
            seq_names={"outputs", "special_token_mask"},
        )

        losses, _ = loss(outputs, inputs)

        self.assertAlmostEqual(losses["value"].item(), 0.0, places=6)


if __name__ == "__main__":
    main()
