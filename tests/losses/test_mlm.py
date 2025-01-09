#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.losses import TimeMAELoss, CrossEntropyLoss
from pretpp.losses import MLMLoss
from pretpp.losses.mlm import EVAL_MASK_FIELD


class TestMLMLoss(TestCase):
    def test_augmentation(self):
        lengths = [
            4,
            3,
            0
        ]
        timestamps = [
            [0, 1.5, 3, 4],
            [5, 6.5, 7, -1],
            [5, 5, 5, 5]
        ]
        labels = [
            [1, 0, 0, 0],
            [3, 0, 1, 0],
            [0, 0, 0, 0]
        ]
        index = [
            0,
            1,
            2
        ]
        inputs = PaddedBatch({"timestamps": torch.tensor(timestamps),
                              "labels": torch.tensor(labels),
                              "index": torch.tensor(index)
                              },
                             torch.tensor(lengths).long(),
                             seq_names=["timestamps", "labels"])

        # Check no augmentation.
        loss = MLMLoss(losses={"timestamps": TimeMAELoss(),
                               "labels": CrossEntropyLoss(num_classes=5)},
                       mask_token={"timestamps": -1,
                                   "labels": 4},
                       eval_fraction=0.3,
                       mask_prob=0,
                       random_prob=0)
        model_inputs, targets = loss.prepare_batch(inputs)
        self.assertEqual(set(model_inputs.payload), {"timestamps", "labels", "index"})
        self.assertEqual(set(targets.payload), {"timestamps", "labels", EVAL_MASK_FIELD})
        self.assertEqual(model_inputs.seq_lens.tolist(), [3, 2, 0])
        self.assertEqual(targets.seq_lens.tolist(), [3, 2, 0])
        for key in model_inputs.seq_names:
            self.assertTrue(model_inputs.payload[key].allclose(inputs.payload[key][:, 1:]))
            self.assertTrue(targets.payload[key].allclose(inputs.payload[key][:, :-1]))

        # Check masking.
        loss = MLMLoss(losses={"timestamps": TimeMAELoss(),
                               "labels": CrossEntropyLoss(num_classes=5)},
                       mask_token={"timestamps": -1,
                                   "labels": 4},
                       eval_fraction=0.3,
                       mask_prob=0.6,
                       random_prob=0.3)
        model_inputs, targets = loss.prepare_batch(inputs)
        # TODO: more tests.


if __name__ == "__main__":
    main()
