#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder import HistoryTokenTransformer, FullHTStrategy


class TestFullHTStrategy(TestCase):
    def test_insert_remove_tokens(self):
        strategy = FullHTStrategy(1)
        strategy.token.data.fill_(-1)
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
        lengths = torch.tensor([4, 4, 4])
        with strategy(PaddedBatch(timestamps, lengths)) as s:
            new_embeddings, new_timestamps = s.insert_tokens(PaddedBatch(embeddings, lengths),
                                                             PaddedBatch(timestamps, lengths))
            gt_embeddings = torch.tensor([
                [0, -1, 1, -1, 2, -1, 3, -1],
                [4, -1, 5, -1, 6, -1, 7, -1],
                [8, -1, 9, -1, 10, -1, 11, -1]
            ]).reshape(3, 8, 1)
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())

            gt_timestamps = torch.tensor([
                [0, 0, 1.5, 1.5, 3, 3, 4, 4],
                [5, 5, 6.5, 6.5, 7, 7, -1, -1],
                [5, 5, 5, 5, 5, 5, 5, 5]
            ])
            self.assertTrue((new_timestamps.payload - gt_timestamps).abs().max() < 1e-6)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == embeddings).all())

    def test_make_attention_mask(self):
        n_summarize = torch.tensor([
            0, 2, 1
        ])
        mask = FullHTStrategy._make_attention_mask_impl(n_summarize)

        mask_gt = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1],  # History token.
            [1, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],  # History token.
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 0]   # History token.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

    def test_sample_mask(self):
        timestamps = torch.tensor([
            [0, 1.5, 3],
            [5, 6.5, 7],
            [5, 5, 5]
        ])
        lengths = torch.tensor([
            3, 3, 3
        ])
        timestamps = PaddedBatch(timestamps, lengths)
        with FullHTStrategy(1)(timestamps) as s:
            for _ in range(8):
                mask = s.make_attention_mask()
                self.assertEqual(mask.shape, (6, 6))
                self.assertTrue((~mask[1::2, ::2]).all())
                self.assertTrue((~mask.diag()).all())


if __name__ == "__main__":
    main()
