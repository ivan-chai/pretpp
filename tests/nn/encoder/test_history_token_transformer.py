#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder.history_token_transformer import HistoryTokenTransformer, insert_tokens, remove_tokens, make_mask, sample_mask


class TestHistoryTokenTransformer(TestCase):
    def test_insert_remove_tokens(self):
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
        token = torch.tensor([-1])
        new_embeddings, new_timestamps = insert_tokens(embeddings, timestamps, token)

        gt_embeddings = torch.tensor([
            [0, -1, 1, -1, 2, -1, 3, -1],
            [4, -1, 5, -1, 6, -1, 7, -1],
            [8, -1, 9, -1, 10, -1, 11, -1]
        ]).reshape(3, 8, 1)
        self.assertTrue((new_embeddings == gt_embeddings).all())

        gt_timestamps = torch.tensor([
            [0, 0, 1.5, 1.5, 3, 3, 4, 4],
            [5, 5, 6.5, 6.5, 7, 7, -1, -1],
            [5, 5, 5, 5, 5, 5, 5, 5]
        ])
        self.assertTrue((new_timestamps - gt_timestamps).abs().max() < 1e-6)

        reverted_embeddings = remove_tokens(new_embeddings)
        self.assertTrue((reverted_embeddings == embeddings).all())

    def test_make_mask(self):
        n_summarize = torch.tensor([
            0, 2, 1
        ])
        mask = make_mask(n_summarize)

        mask_gt = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],  # History token.
            [1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],  # History token.
            [1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0]   # History token.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

    def test_sample_mask(self):
        mask = sample_mask(3, locality=1)
        mask_gt = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],  # History token.
            [1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],  # History token.
            [1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]   # History token.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

        for _ in range(8):
            mask = sample_mask(3, locality=0)
            self.assertEqual(mask.shape, (6, 6))
            self.assertTrue((~mask[1::2]).all())
            self.assertTrue((~mask.diag()).all())


if __name__ == "__main__":
    main()
