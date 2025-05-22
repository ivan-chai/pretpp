#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder.history_token_transformer import HistoryTokenTransformer, insert_tokens, sample_mask


class TestHistoryTokenTransformer(TestCase):
    def test_insert_tokens(self):
        embeddings = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]).reshape(3, 4, 1)
        timestamps = torch.tensor([
            [0, 1.5, 3, 4],
            [5, 6.5, 7, -1],
            [5, 5, 5, 5]
        ])
        positions = torch.tensor([
            [1, 3],
            [0, 1],
            [3, 5]  # Require sorting.
        ])
        token = torch.tensor([-1])
        new_embeddings, new_timestamps, source_indices = insert_tokens(embeddings, timestamps, positions, token)

        gt_embeddings = torch.tensor([
            [0, -1, 1, -1, 2, 3],
            [-1, -1, 4, 5, 6, 7],
            [8, 9, 10, -1, 11, -1]
        ]).reshape(3, 6, 1)
        self.assertTrue((new_embeddings == gt_embeddings).all())

        gt_timestamps = torch.tensor([
            [0, 0, 1.5, 1.5, 3, 4],
            [5, 5, 5, 6.5, 7, -1],
            [5, 5, 5, 5, 5, 5]
        ])
        self.assertTrue((new_timestamps - gt_timestamps).abs().max() < 1e-6)

        gt_indices = torch.tensor([
            [0, 2, 4, 5],
            [2, 3, 4, 5],
            [0, 1, 2, 4]
        ])
        self.assertTrue((source_indices == gt_indices).all())

    def test_sample_mask(self):
        positions = torch.tensor([1, 3])
        l = 4
        mask = sample_mask(positions, l)  # (6, 6).
        mask_gt = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],  # History token.
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],  # History token.
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ])
        self.assertTrue((mask == mask_gt).all())


if __name__ == "__main__":
    main()
