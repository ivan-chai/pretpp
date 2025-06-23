#!/usr/bin/env python3
# Test common strategies:
# - "apply" method (including "apply_probability" parameter)
# - "extract_outputs" method (including "predict" mode)

import math
from contextlib import contextmanager
from unittest import TestCase, main, mock

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder import FullHTStrategy, SubsetHTStrategy, FixedHTStrategy
from pretpp.nn.encoder.history_token_strategy import make_ht_attention_mask


class TestMakeHTAttentionMask(TestCase):
    def test_make_attention_mask(self):
        l = 4
        ht_positions = torch.tensor([0, 2, 3])

        # Fixed HT tokens.
        active_tokens = torch.tensor([0, 1, 0, 2])
        mask = make_ht_attention_mask(l, ht_positions, active_tokens=active_tokens)

        mask_gt = torch.tensor([
            [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
            [0, 0, 0, 0, 1, 0, 1],  # History token.
            [1, 0, 0, 0, 1, 0, 1],  # 1 active token.
            [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
            [0, 1, 0, 0, 0, 0, 1],  # History token.
            [1, 1, 1, 1, 0, 0, 1],  # 2 active tokens.
            [0, 1, 0, 0, 1, 0, 0]   # History token.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

        # Last HT token.
        mask = make_ht_attention_mask(l, ht_positions, active_tokens="last")

        mask_gt = torch.tensor([
            [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
            [0, 0, 0, 0, 1, 0, 1],  # History token.
            [1, 0, 0, 0, 1, 0, 1],  # 1 active token.
            [1, 0, 0, 0, 1, 0, 1],  # 1 active tokens.
            [0, 1, 0, 0, 0, 0, 1],  # History token.
            [1, 1, 1, 1, 0, 0, 1],  # 2 active tokens.
            [0, 1, 0, 0, 1, 0, 0]   # History token.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

        # No HT tokens.
        mask = make_ht_attention_mask(l, ht_positions, active_tokens="none")

        mask_gt = torch.tensor([
            [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
            [0, 0, 0, 0, 1, 0, 1],  # History token.
            [0, 1, 0, 0, 1, 0, 1],  # 0 active token.
            [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
            [0, 1, 0, 0, 0, 0, 1],  # History token.
            [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
            [0, 1, 0, 0, 1, 0, 0]   # History token.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())


class TestHTStrategy(TestCase):
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

    @mock.patch("torch.rand")
    @mock.patch("torch.randn")
    def test_full_strategy(self, mock_randn, mock_rand):
        n_active_tokens = torch.arange(4)
        mock_randn.side_effect = [
            torch.tensor([-1.0]),  # Token embedding.
        ]
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor(0.2),  # < apply_probability = 0.5.
            torch.tensor([0, 0, 2, 1]) / n_active_tokens.clip(min=1),  # Active tokens.
            # Case 2: don't apply.
            torch.tensor(0.8),  # > apply_probability = 0.5.
        ]

        strategy = FullHTStrategy(1)

        gt_embeddings = torch.tensor([
            [0, -1, 1, -1, 2, -1, 3, -1],
            [4, -1, 5, -1, 6, -1, 7, -1],
            [8, -1, 9, -1, 10, -1, 11, -1]
        ]).reshape(3, 8, 1)

        gt_timestamps = torch.tensor([
            [0, 0, 1.5, 1.5, 3, 3, 4, 4],
            [5, 5, 6.5, 6.5, 7, 7, -1, -1],
            [5, 5, 5, 5, 5, 5, 5, 5]
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [8, 6, 8])
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())
            self.assertTrue((new_timestamps.payload - gt_timestamps).abs().max() < 1e-6)

            gt_attention_mask = torch.tensor([
                [0, 1, 0, 1, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 1, 0, 1, 0, 1],  # History token.
                [0, 1, 0, 1, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 0, 1, 0, 1],  # History token.
                [1, 1, 1, 0, 0, 1, 0, 1],  # 2 active tokens.
                [0, 1, 0, 1, 0, 0, 0, 1],  # History token.
                [1, 0, 0, 1, 0, 1, 0, 1],  # 1 active token.
                [0, 1, 0, 1, 0, 1, 0, 0]   # History token.
            ])
            self.assertTrue((attention_mask == gt_attention_mask).all())

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 2: don't apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [4, 3, 4])
            self.assertTrue((new_embeddings.payload == self.embeddings.payload).all())
            self.assertTrue((new_timestamps.payload - self.timestamps.payload).abs().max() < 1e-6)

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 3: embedding.
        with strategy(self.timestamps, embedding=True) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])

            gt_embeddings = torch.tensor([
                [0, 1, 2, 3, -1],
                [4, 5, 6, -1, 7],
                [8, 9, 10, 11, -1]
            ]).reshape(3, 5, 1)
            self.assertTrue(torch.logical_or(new_embeddings.payload == gt_embeddings, ~new_embeddings.seq_len_mask.unsqueeze(2)).all())

            gt_timestamps = torch.tensor([
                [0, 1.5, 3, 4, 4],
                [5, 6.5, 7, 7, -1],
                [5, 5, 5, 5, 5]
            ])
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings - (-1)).abs().max() < 1e-6)

    @mock.patch("torch.rand")
    @mock.patch("torch.randperm")
    @mock.patch("torch.randn")
    def test_subset_strategy(self, mock_randn, mock_randperm, mock_rand):
        mock_randn.side_effect = [
            torch.tensor([-1.0]),  # Token embedding.
        ]
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor(0.2),  # < apply_probability = 0.5.
            # Case 2: don't apply.
            torch.tensor(0.8),  # > apply_probability = 0.5.
            # Case 3: embedding (unused).
            torch.tensor(0.8),
        ]
        mock_randperm.side_effect = [
            # Case 1: apply.
            torch.tensor([1, 3]),  # Selected positions.
            # Case 2: don't apply.
        ]
        strategy = SubsetHTStrategy(1, frequency=0.5)


        gt_embeddings = torch.tensor([
            [0, 1, -1, 2, 3, -1],
            [4, 5, -1, 6, 7, -1],
            [8, 9, -1, 10, 11, -1]
        ]).reshape(3, 6, 1)

        gt_timestamps = torch.tensor([
            [0, 1.5, 1.5, 3, 4, 4],
            [5, 6.5, 6.5, 7, -1, -1],
            [5, 5, 5, 5, 5, 5]
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [6, 4, 6])
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())
            self.assertTrue((new_timestamps.payload - gt_timestamps).abs().max() < 1e-6)

            gt_attention_mask = torch.tensor([
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ])
            self.assertTrue((attention_mask == gt_attention_mask).all())

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 2: don't apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [4, 3, 4])
            self.assertTrue((new_embeddings.payload == self.embeddings.payload).all())
            self.assertTrue((new_timestamps.payload - self.timestamps.payload).abs().max() < 1e-6)

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 3: embedding.
        with strategy(self.timestamps, embedding=True) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])

            gt_embeddings = torch.tensor([
                [0, 1, 2, 3, -1],
                [4, 5, 6, -1, 7],
                [8, 9, 10, 11, -1]
            ]).reshape(3, 5, 1)
            self.assertTrue(torch.logical_or(new_embeddings.payload == gt_embeddings, ~new_embeddings.seq_len_mask.unsqueeze(2)).all())

            gt_timestamps = torch.tensor([
                [0, 1.5, 3, 4, 4],
                [5, 6.5, 7, 7, -1],
                [5, 5, 5, 5, 5]
            ])
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings - (-1)).abs().max() < 1e-6)

    @mock.patch("torch.rand")
    @mock.patch("torch.randn")
    def test_fixed_strategy(self, mock_randn, mock_rand):
        mock_randn.side_effect = [
            torch.tensor([-1.0]),  # Token embedding.
        ]
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor(0.2),  # < apply_probability = 0.5.
            # Case 2: don't apply.
            torch.tensor(0.8),  # > apply_probability = 0.5.
            # Case 3: embedding (unused).
            torch.tensor(0.8),
        ]
        strategy = FixedHTStrategy(1, positions=[1, 3])

        gt_embeddings = torch.tensor([
            [0, 1, -1, 2, 3, -1],
            [4, 5, -1, 6, 7, -1],
            [8, 9, -1, 10, 11, -1]
        ]).reshape(3, 6, 1)

        gt_timestamps = torch.tensor([
            [0, 1.5, 1.5, 3, 4, 4],
            [5, 6.5, 6.5, 7, -1, -1],
            [5, 5, 5, 5, 5, 5]
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [6, 4, 6])
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())
            self.assertTrue((new_timestamps.payload - gt_timestamps).abs().max() < 1e-6)

            gt_attention_mask = torch.tensor([
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ])
            self.assertTrue((attention_mask == gt_attention_mask).all())

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 2: don't apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [4, 3, 4])
            self.assertTrue((new_embeddings.payload == self.embeddings.payload).all())
            self.assertTrue((new_timestamps.payload - self.timestamps.payload).abs().max() < 1e-6)

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 3: embedding.
        with strategy(self.timestamps, embedding=True) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 3, 5])

            gt_embeddings = torch.tensor([
                [0, 1, 2, 3, -1],
                [4, 5, -1, 6, 7],
                [8, 9, 10, 11, -1]
            ]).reshape(3, 5, 1)
            self.assertTrue(torch.logical_or(new_embeddings.payload == gt_embeddings, ~new_embeddings.seq_len_mask.unsqueeze(2)).all())

            gt_timestamps = torch.tensor([
                [0, 1.5, 3, 4, 4],
                [5, 6.5, 6.5, 7, -1],
                [5, 5, 5, 5, 5]
            ])
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings - (-1)).abs().max() < 1e-6)


if __name__ == "__main__":
    main()
