#!/usr/bin/env python3
# Test common strategies:
# - "apply" method (including "apply_probability" parameter)
# - "extract_outputs" method (including "predict" mode)

import math
from contextlib import contextmanager
from unittest import TestCase, main, mock

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder import FullHTStrategy, SubsetHTStrategy, FixedHTStrategy, LastHTStrategy, NoHTStrategy
from pretpp.nn.encoder.history_token_strategy import LongFormerHTStrategy, make_ht_attention_mask


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

    @mock.patch("torch.randn")
    def test_last_strategy(self, mock_randn):
        mock_randn.side_effect = [
            torch.tensor([-1.0]),  # Token embedding.
        ]
        strategy = LastHTStrategy(1)

        gt_embeddings = torch.tensor([
            [0, 1, 2, 3, -1],
            [4, 5, 6, -1, 7],
            [8, 9, 10, 11, -1]
        ]).reshape(3, 5, 1)

        gt_timestamps = torch.tensor([
            [0, 1.5, 3, 4, 4],
            [5, 6.5, 7, 7, -1],
            [5, 5, 5, 5, 5]
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            self.assertTrue(torch.logical_or(new_embeddings.payload == gt_embeddings, ~new_embeddings.seq_len_mask.unsqueeze(2)).all())
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue(torch.logical_or(reverted_embeddings.payload == self.embeddings.payload, ~self.embeddings.seq_len_mask.unsqueeze(2)).all())

        # Case 2: embedding.
        with strategy(self.timestamps, embedding=True) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            self.assertTrue(torch.logical_or(new_embeddings.payload == gt_embeddings, ~new_embeddings.seq_len_mask.unsqueeze(2)).all())
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertTrue((reverted_embeddings - (-1)).abs().max() < 1e-6)

    def test_no_strategy(self):
        strategy = NoHTStrategy(1)

        gt_embeddings = torch.tensor([
            [3],
            [6],
            [11]
        ]).reshape(3, 1)

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), self.embeddings.seq_lens.tolist())
            self.assertAlmostEqual((new_embeddings.payload - self.embeddings.payload).abs().max().item(), 0)
            self.assertAlmostEqual((new_timestamps.payload - self.timestamps.payload).abs().max().item(), 0)

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertAlmostEqual((reverted_embeddings.payload - self.embeddings.payload).abs().max().item(), 0)

        # Case 2: embedding.
        with strategy(self.timestamps, embedding=True) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), self.embeddings.seq_lens.tolist())
            self.assertAlmostEqual((new_embeddings.payload - self.embeddings.payload).abs().max().item(), 0)
            self.assertAlmostEqual((new_timestamps.payload - self.timestamps.payload).abs().max().item(), 0)

            self.assertTrue(attention_mask is None)

            reverted_embeddings = s.extract_outputs(new_embeddings)
            self.assertAlmostEqual((reverted_embeddings - gt_embeddings).abs().max().item(), 0)


class TestSpecialTokenMask(TestCase):
    """Tests for special_token_mask correctness across all strategies."""

    def setUp(self):
        payload = torch.arange(12).reshape(3, 4, 1).float()
        timestamps = torch.zeros(3, 4)
        lengths = torch.tensor([4, 3, 4])
        self.embeddings = PaddedBatch(payload, lengths)
        self.timestamps = PaddedBatch(timestamps, lengths)

    # --- HTStrategyImpl (via FixedHTStrategy), predict == "all" ---

    @mock.patch("torch.rand")
    @mock.patch("torch.randn")
    def test_fixed_strategy_predict_all_special_token_mask_shape(self, mock_randn, mock_rand):
        mock_randn.return_value = torch.tensor([-1.0])
        mock_rand.return_value = torch.tensor(0.2)  # apply

        strategy = FixedHTStrategy(1, positions=[1, 3], predict="all")
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, _ = s.apply(self.embeddings, self.timestamps)
            outputs = s.extract_outputs(new_embeddings)

        self.assertIsInstance(outputs.payload, dict)
        mask = outputs.payload["special_token_mask"]
        # Shape must be (B, L+R).
        b, l_plus_r = new_embeddings.shape
        self.assertEqual(mask.shape, (b, l_plus_r))

    @mock.patch("torch.rand")
    @mock.patch("torch.randn")
    def test_fixed_strategy_predict_all_special_token_mask_values(self, mock_randn, mock_rand):
        """HT positions should be True, real token positions False."""
        mock_randn.return_value = torch.tensor([-1.0])
        mock_rand.return_value = torch.tensor(0.2)  # apply

        # positions=[1, 3]: HT inserted after positions 1 and 3.
        # New sequence layout: [r0, r1, HT, r2, r3, HT] → HT at indices 2 and 5.
        strategy = FixedHTStrategy(1, positions=[1, 3], predict="all")
        with strategy(self.timestamps) as s:
            new_embeddings, _, _ = s.apply(self.embeddings, self.timestamps)
            outputs = s.extract_outputs(new_embeddings)

        mask = outputs.payload["special_token_mask"]
        # All batch rows share the same HT positions (at indices 2 and 5).
        expected = torch.tensor([False, False, True, False, False, True])
        for b in range(mask.shape[0]):
            self.assertEqual(mask[b].tolist(), expected.tolist(), f"Row {b} mismatch")

    @mock.patch("torch.rand")
    @mock.patch("torch.randn")
    def test_fixed_strategy_predict_all_no_apply(self, mock_randn, mock_rand):
        """When apply_to_batch=False, extract_outputs returns the input unchanged (no dict)."""
        mock_randn.return_value = torch.tensor([-1.0])
        mock_rand.return_value = torch.tensor(0.8)  # don't apply

        strategy = FixedHTStrategy(1, positions=[1, 3], predict="all")
        with strategy(self.timestamps) as s:
            new_embeddings, _, _ = s.apply(self.embeddings, self.timestamps)
            outputs = s.extract_outputs(new_embeddings)

        # No HT tokens inserted → plain PaddedBatch returned.
        self.assertIsInstance(outputs, PaddedBatch)
        self.assertNotIsInstance(outputs.payload, dict)

    # --- LastHTStrategy ---

    @mock.patch("torch.randn")
    def test_last_strategy_predict_history_tokens_no_name_error(self, mock_randn):
        """Bug: len(payload) was undefined. Should not raise."""
        mock_randn.return_value = torch.tensor([-1.0])
        strategy = LastHTStrategy(1, predict="history_tokens")

        with strategy(self.timestamps) as s:
            new_emb, new_ts, _ = s.apply(self.embeddings, self.timestamps)
            outputs = s.extract_outputs(new_emb)

        # Returns 1 HT token per sequence.
        self.assertIsInstance(outputs.payload, dict)
        self.assertEqual(outputs.payload["outputs"].shape[1], 1)
        self.assertEqual(outputs.seq_lens.tolist(), [1, 1, 1])
        # All returned tokens are special.
        mask = outputs.payload["special_token_mask"]
        self.assertTrue(mask.all())

    @mock.patch("torch.randn")
    def test_last_strategy_predict_all_special_token_mask(self, mock_randn):
        """HT appended at end → last valid position should be True."""
        mock_randn.return_value = torch.tensor([-1.0])
        strategy = LastHTStrategy(1, predict="all")

        with strategy(self.timestamps) as s:
            new_emb, new_ts, _ = s.apply(self.embeddings, self.timestamps)
            outputs = s.extract_outputs(new_emb)

        self.assertIsInstance(outputs.payload, dict)
        mask = outputs.payload["special_token_mask"]
        # seq_lens should be original + 1 (includes HT token).
        self.assertEqual(outputs.seq_lens.tolist(), new_emb.seq_lens.tolist())
        # HT is at position seq_lens-1 for each row.
        for b, sl in enumerate(outputs.seq_lens.tolist()):
            self.assertTrue(mask[b, sl - 1].item(), f"Row {b}: HT position should be True")
            self.assertFalse(mask[b, :sl - 1].any().item(), f"Row {b}: real positions should be False")

    # --- LongFormerHTStrategy ---

    def test_longformer_strategy_special_token_mask_shape_and_values(self):
        """Global positions should be marked True in all batch rows."""
        strategy = LongFormerHTStrategy(global_frequency=0.5)

        # Use embedding=True for deterministic global_positions.
        with strategy(self.timestamps, embedding=False) as s:
            # Override global_positions with known values for determinism.
            s.global_positions = torch.tensor([0, 2])
            new_emb, _, _ = s.apply(self.embeddings, self.timestamps)
            outputs = s.extract_outputs(new_emb)

        self.assertIsInstance(outputs.payload, dict)
        mask = outputs.payload["special_token_mask"]
        b, l = self.embeddings.shape
        self.assertEqual(mask.shape, (b, l))

        # Positions 0 and 2 should be True in every row.
        for row in range(b):
            self.assertTrue(mask[row, 0].item(), f"Row {row}, pos 0 should be True")
            self.assertTrue(mask[row, 2].item(), f"Row {row}, pos 2 should be True")
            self.assertFalse(mask[row, 1].item(), f"Row {row}, pos 1 should be False")
            self.assertFalse(mask[row, 3].item(), f"Row {row}, pos 3 should be False")


if __name__ == "__main__":
    main()
