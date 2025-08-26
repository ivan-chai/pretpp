#!/usr/bin/env python3
# Test common strategies:
# - "apply" method (including "apply_probability" parameter)
# - "extract_outputs" method (including "predict" mode)

import math
from contextlib import contextmanager
from unittest import TestCase, main, mock

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder import HistoryTokenTransformer
from pretpp.nn.encoder.history_token_strategy import LastHTStrategy


class TestHistoryTokenTransformer(TestCase):
    def setUp(self):
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
        lengths = torch.tensor([4, 3, 4])
        self.embeddings = PaddedBatch(embeddings, lengths)
        self.timestamps = PaddedBatch(timestamps, lengths)

    def test_add_avg(self):
        model = HistoryTokenTransformer(1,
                                        strategy_partial=lambda dim: LastHTStrategy(dim),
                                        n_embd=1, n_head=1, causal=True, add_avg=True)
        model.strategy.token.data.fill_(-1)

        # Mock transformer.
        del model.input_projection
        model.input_projection = torch.nn.Identity()
        del model.positional
        model.positional = lambda embeddings, timestamps: embeddings

        gt_embeddings = torch.tensor([
            [6/4, 0, 1, 2, 3, -1],
            [15/3, 4, 5, 6, -1, 7],
            [38/4, 8, 9, 10, 11, -1]
        ]).reshape(3, 6, 1)

        gt_timestamps = torch.tensor([
            [0, 0, 1.5, 3, 4, 4],
            [5, 5, 6.5, 7, 7, -1],
            [5, 5, 5, 5, 5, 5]
        ])

        def transform(embeddings, *args, **kwargs):
            self.assertEqual(embeddings.seq_lens.tolist(), [6, 5, 6])
            self.assertTrue(torch.logical_or(embeddings.payload == gt_embeddings, ~embeddings.seq_len_mask.unsqueeze(2)).all())
            return (embeddings, None)
        model.transform = transform
        model(self.embeddings, self.timestamps)


if __name__ == "__main__":
    main()
