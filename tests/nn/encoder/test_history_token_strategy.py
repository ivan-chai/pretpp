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
        b = 3
        l = 4
        ht_positions = torch.tensor([
            [0, 2, 3],
            [0, 1, 2],
            [0, 2, 3],
        ])
        r = len(ht_positions)

        # Fixed HT tokens.
        active_tokens = torch.tensor([0, 1, 0, 2])
        mask = make_ht_attention_mask(l, ht_positions, active_tokens=active_tokens[None].repeat(b, 1))
        self.assertEqual(mask.shape, (b, l + r, l + r))

        mask_gt = torch.tensor([
            [
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 0, 1, 0, 1],  # History token.
                [1, 0, 0, 0, 1, 0, 1],  # 1 active token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 0, 0, 1],  # History token.
                [1, 1, 1, 1, 0, 0, 1],  # 2 active tokens.
                [0, 1, 0, 0, 1, 0, 0]   # History token.
            ],  # 0, 2, 3.
            [
                [0, 1, 0, 1, 0, 1, 0],  # 0 active tokens.
                [0, 0, 0, 1, 0, 1, 0],  # History token.
                [1, 0, 0, 1, 0, 1, 0],  # 1 active token.
                [0, 1, 0, 0, 0, 1, 0],  # History token.
                [0, 1, 0, 1, 0, 1, 0],  # 0 active tokens.
                [0, 1, 0, 1, 0, 0, 0],  # History token.
                [1, 1, 1, 0, 0, 1, 0]   # 2 active tokens.
            ],  # 0, 1, 2.
            [
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 0, 1, 0, 1],  # History token.
                [1, 0, 0, 0, 1, 0, 1],  # 1 active token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 0, 0, 1],  # History token.
                [1, 1, 1, 1, 0, 0, 1],  # 2 active tokens.
                [0, 1, 0, 0, 1, 0, 0]   # History token.
            ]  # 0, 2, 3.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

        # Last HT token.
        mask = make_ht_attention_mask(l, ht_positions, active_tokens="last")
        self.assertEqual(mask.shape, (b, l + r, l + r))

        mask_gt = torch.tensor([
            [
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 0, 1, 0, 1],  # History token.
                [1, 0, 0, 0, 1, 0, 1],  # 1 active token.
                [1, 0, 0, 0, 1, 0, 1],  # 1 active tokens.
                [0, 1, 0, 0, 0, 0, 1],  # History token.
                [1, 1, 1, 1, 0, 0, 1],  # 2 active tokens.
                [0, 1, 0, 0, 1, 0, 0]   # History token.
            ],  # 0, 2, 3.
            [
                [0, 1, 0, 1, 0, 1, 0],  # 0 active tokens.
                [0, 0, 0, 1, 0, 1, 0],  # History token.
                [1, 0, 0, 1, 0, 1, 0],  # 1 active token.
                [0, 1, 0, 0, 0, 1, 0],  # History token.
                [1, 1, 1, 0, 0, 1, 0],  # 2 active tokens.
                [0, 1, 0, 1, 0, 0, 0],  # History token.
                [1, 1, 1, 1, 1, 0, 0]   # 3 active tokens.
            ],  # 0, 1, 2.
            [
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 0, 1, 0, 1],  # History token.
                [1, 0, 0, 0, 1, 0, 1],  # 1 active token.
                [1, 0, 0, 0, 1, 0, 1],  # 1 active tokens.
                [0, 1, 0, 0, 0, 0, 1],  # History token.
                [1, 1, 1, 1, 0, 0, 1],  # 2 active tokens.
                [0, 1, 0, 0, 1, 0, 0]   # History token.
            ]  # 0, 2, 3.
        ]).bool()
        self.assertTrue((mask == mask_gt).all())

        # No HT tokens.
        mask = make_ht_attention_mask(l, ht_positions, active_tokens="none")
        self.assertEqual(mask.shape, (b, l + r, l + r))

        mask_gt = torch.tensor([
            [
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 0, 1, 0, 1],  # History token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 0, 0, 1],  # History token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 1, 0, 0]   # History token.
            ],  # 0, 2, 3.
            [
                [0, 1, 0, 1, 0, 1, 0],  # 0 active tokens.
                [0, 0, 0, 1, 0, 1, 0],  # History token.
                [0, 1, 0, 1, 0, 1, 0],  # 0 active token.
                [0, 1, 0, 0, 0, 1, 0],  # History token.
                [0, 1, 0, 1, 0, 1, 0],  # 0 active tokens.
                [0, 1, 0, 1, 0, 0, 0],  # History token.
                [0, 1, 0, 1, 0, 1, 0]   # 0 active tokens.
            ],  # 0, 1, 2.
            [
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 0, 0, 0, 1, 0, 1],  # History token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 0, 0, 1],  # History token.
                [0, 1, 0, 0, 1, 0, 1],  # 0 active tokens.
                [0, 1, 0, 0, 1, 0, 0]   # History token.
            ],  # 0, 2, 3.
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
    def test_full_strategy(self, mock_rand):
        n_active_tokens = torch.arange(4)
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor([-1.0]),  # Token embedding.
            torch.tensor([0.2, 0.3, 0.1]),  # < apply_probability = 0.5.
            torch.tensor([0, 0, 2, 1])[None].repeat(3, 1) / n_active_tokens.clip(min=1),  # Active tokens.
            # Case 2: embedding.
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

        # Case 2: embedding.
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
    def test_subset_strategy(self, mock_rand):
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor([-1.0]),  # Token embedding.
            torch.tensor([
                [1, 0.1, 1, 0.2],  # put 1 and 3 at the first place.
                [0, 0.1, 0.2, 1],  # put 0 and 1 at the first place
                [1, 0.1, 1, 0.2],  # put 1 and 3 at the first place.
            ]),  # batch_randperm, put 1 and 3 at the first place.
            torch.tensor([0.2, 0.1, 0.3]),  # < apply_probability = 0.5.
            # Case 2: embedding.
        ]
        strategy = SubsetHTStrategy(1, frequency=0.5)

        gt_embeddings = torch.tensor([
            [0, 1, -1, 2, 3, -1],  # 1, 3.
            [4, -1, 5, -1, 6, 7],  # 0, 1.
            [8, 9, -1, 10, 11, -1]  # 1, 3.
        ]).reshape(3, 6, 1)

        gt_timestamps = torch.tensor([
            [0, 1.5, 1.5, 3, 4, 4],  # 1, 3.
            [5, 5, 6.5, 6.5, 7, -1],  # 0, 1.
            [5, 5, 5, 5, 5, 5]  # 1, 3.
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [6, 5, 6])
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())
            self.assertTrue((new_timestamps.payload - gt_timestamps).abs().max() < 1e-6)

            gt_attention_mask = torch.tensor([
                [
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ],
                [
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],  # History token.
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0],  # History token.
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                ],
                [
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ],
            ])
            self.assertTrue((attention_mask == gt_attention_mask).all())

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 2: embedding.
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
    def test_fixed_strategy(self, mock_rand):
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor([-1.0]),  # Token embedding.
            torch.tensor([0.2, 0.6, 0.3]),  # Disable 0.6, as it > apply_probability = 0.5.
            # Case 2: embedding.
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
                [
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ],
                [
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ],  # Disabled.
                [
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],  # History token.
                    [1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],  # History token.
                ],
            ])
            self.assertTrue((attention_mask == gt_attention_mask).all())

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings.payload == self.embeddings.payload).all())

        # Case 2: embedding.
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
