#!/usr/bin/env python3
import math
from collections import Counter
from unittest import TestCase, main

import numpy as np
import torch

from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector

from hotpp.data import PaddedBatch
from hotpp.losses import TimeMAELoss, CrossEntropyLoss
from pretpp.losses import ColesLoss


class TestColesLoss(TestCase):
    def test_prepare_batch(self):
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
        ids = [
            0,
            1,
            0
        ]
        index = [
            0,
            1,
            2
        ]
        inputs = PaddedBatch({"timestamps": torch.tensor(timestamps),
                              "id": torch.tensor(ids),
                              "index": torch.tensor(index)},
                             torch.tensor(lengths).long(),
                             seq_names=["timestamps"])

        loss = ColesLoss(embedding_dim=16,
                         n_splits=2,
                         min_length=1,
                         max_length=3,
                         coles_loss=ContrastiveLoss(
                             margin=0.5,
                             sampling_strategy=HardNegativePairSelector(neg_count=2)
                         ))
        model_inputs, targets = loss.prepare_batch(inputs, None)
        self.assertEqual(len(model_inputs), 3 * 2)  # BS x NSplits.
        self.assertEqual(Counter(targets.tolist()), {0: 4, 1: 2})
        for i in range(len(model_inputs)):
            i_orig = model_inputs.payload["index"][i].item()
            ts_inp = model_inputs.payload["timestamps"][i].numpy()
            ts_orig = np.asarray(timestamps[i_orig])
            self.assertLessEqual(len(ts_inp), 3)
            self.assertGreaterEqual(len(ts_inp), 1)
            is_subsequence = False
            for j in range(1 + len(ts_orig) - len(ts_inp)):
                if np.linalg.norm(ts_orig[j:j + len(ts_inp)] - ts_inp) < 1e-6:
                    is_subsequence = True
                    break
            self.assertTrue(is_subsequence)


if __name__ == "__main__":
    main()
