#!/usr/bin/env python3
# Test common strategies:
# - "apply" method (including "apply_probability" parameter)
# - "extract_outputs" method (including "predict" mode)

import math
from contextlib import contextmanager
from unittest import TestCase, main, mock

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.encoder import FullHTStrategy, SingleHTStrategy, SubsetHTStrategy, FixedHTStrategy, LastHTStrategy, NoHTStrategy
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
    def test_subset_strategy_single_token(self, mock_randn, mock_randperm, mock_rand):
        """With frequency=0 a single history token is inserted at a random position."""
        mock_randn.side_effect = [
            torch.tensor([-1.0]),  # Token embedding.
        ]
        mock_randperm.side_effect = [
            # Case 1: apply.
            torch.tensor([1]),  # Selected position.
            # Cases 2 and 3 don't select positions.
        ]
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor(0.2),  # < apply_probability = 0.5.
            torch.tensor([0, 0, 0.6, 0.2]),  # Active tokens (only tokens after the HT can use it).
            # Case 2: don't apply.
            torch.tensor(0.8),  # > apply_probability = 0.5.
            # Case 3: embedding (unused).
            torch.tensor(0.8),
        ]
        strategy = SubsetHTStrategy(1, frequency=0)

        gt_embeddings = torch.tensor([
            [0, 1, -1, 2, 3],
            [4, 5, -1, 6, 7],
            [8, 9, -1, 10, 11]
        ]).reshape(3, 5, 1)

        gt_timestamps = torch.tensor([
            [0, 1.5, 1.5, 3, 4],
            [5, 6.5, 6.5, 7, -1],
            [5, 5, 5, 5, 5]
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            # Layout: [r0, r1, HT, r2, r3].
            gt_attention_mask = torch.tensor([
                [0, 0, 1, 0, 0],  # 0 active tokens.
                [0, 0, 1, 0, 0],  # 0 active tokens.
                [0, 0, 0, 0, 0],  # History token.
                [1, 1, 0, 0, 0],  # 1 active token.
                [0, 0, 1, 0, 0],  # 0 active tokens.
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

    @mock.patch("torch.randperm")
    @mock.patch("torch.randn")
    def test_subset_strategy_token_selection(self, mock_randn, mock_randperm):
        """`token_selection` controls HT usage by each real token."""
        mock_randn.return_value = torch.tensor([-1.0])  # Token embedding.
        mock_randperm.return_value = torch.tensor([1])  # Selected position.

        # Layout: [r0, r1, HT, r2, r3].
        strategy = SubsetHTStrategy(1, frequency=0, apply_probability=1.0, token_selection="last")
        with strategy(self.timestamps) as s:
            new_embeddings, _, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            # Both r2 and r3 use the HT and skip the summarized prefix.
            self.assertEqual(attention_mask[3].tolist(), [True, True, False, False, False])
            self.assertEqual(attention_mask[4].tolist(), [True, True, False, False, False])

        strategy = SubsetHTStrategy(1, frequency=0, apply_probability=1.0, token_selection="none")
        with strategy(self.timestamps) as s:
            new_embeddings, _, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            # Nobody uses the HT, so all tokens attend to the full prefix.
            self.assertEqual(attention_mask[3].tolist(), [False, False, True, False, False])
            self.assertEqual(attention_mask[4].tolist(), [False, False, True, False, False])

    @mock.patch("torch.randn")
    def test_subset_strategy_single_token_positions(self, mock_randn):
        """With frequency=0 the position is sampled uniformly, one token per batch."""
        mock_randn.return_value = torch.tensor([-1.0])  # Token embedding.
        strategy = SubsetHTStrategy(1, frequency=0, apply_probability=1.0)

        counts = torch.zeros(4)
        for _ in range(1000):
            with strategy(self.timestamps) as s:
                self.assertEqual(len(s.after_positions), 1)
                self.assertTrue((s.after_positions >= 0).all() and (s.after_positions < 4).all())
                counts[s.after_positions] += 1
                new_embeddings, _, _ = s.apply(self.embeddings, self.timestamps)
                # Exactly one extra token for sequences longer than the selected position.
                gt_lens = self.embeddings.seq_lens + (s.after_positions[0] < self.embeddings.seq_lens)
                self.assertEqual(new_embeddings.seq_lens.tolist(), gt_lens.tolist())
        self.assertTrue((counts > 150).all(), f"Non-uniform positions: {counts.tolist()}")

    @mock.patch("torch.rand")
    @mock.patch("torch.randint")
    @mock.patch("torch.randn")
    def test_single_strategy(self, mock_randn, mock_randint, mock_rand):
        """SingleHTStrategy inserts a single history token at a random position."""
        mock_randn.side_effect = [
            torch.tensor([-1.0]),  # Token embedding.
        ]
        mock_randint.side_effect = [
            # Case 1: apply.
            torch.tensor([1]),  # Selected position.
            # Cases 2 and 3 don't select positions.
        ]
        mock_rand.side_effect = [
            # Case 1: apply.
            torch.tensor(0.2),  # < apply_probability = 0.5.
            torch.tensor([0, 0, 0.6, 0.2]),  # Active tokens (only tokens after the HT can use it).
            # Case 2: don't apply.
            torch.tensor(0.8),  # > apply_probability = 0.5.
            # Case 3: embedding (unused).
            torch.tensor(0.8),
        ]
        strategy = SingleHTStrategy(1)

        gt_embeddings = torch.tensor([
            [0, 1, -1, 2, 3],
            [4, 5, -1, 6, 7],
            [8, 9, -1, 10, 11]
        ]).reshape(3, 5, 1)

        gt_timestamps = torch.tensor([
            [0, 1.5, 1.5, 3, 4],
            [5, 6.5, 6.5, 7, -1],
            [5, 5, 5, 5, 5]
        ])

        # Case 1: apply.
        with strategy(self.timestamps) as s:
            new_embeddings, new_timestamps, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            self.assertTrue((new_embeddings.payload == gt_embeddings).all())
            self.assertTrue(torch.logical_or((new_timestamps.payload - gt_timestamps).abs() < 1e-6, ~new_embeddings.seq_len_mask).all())

            # Layout: [r0, r1, HT, r2, r3].
            gt_attention_mask = torch.tensor([
                [0, 0, 1, 0, 0],  # 0 active tokens.
                [0, 0, 1, 0, 0],  # 0 active tokens.
                [0, 0, 0, 0, 0],  # History token.
                [1, 1, 0, 0, 0],  # 1 active token.
                [0, 0, 1, 0, 0],  # 0 active tokens.
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

    @mock.patch("torch.randint")
    @mock.patch("torch.randn")
    def test_single_strategy_token_selection(self, mock_randn, mock_randint):
        """`token_selection` controls HT usage by each real token."""
        mock_randn.return_value = torch.tensor([-1.0])  # Token embedding.
        mock_randint.return_value = torch.tensor([1])  # Selected position.

        # Layout: [r0, r1, HT, r2, r3].
        strategy = SingleHTStrategy(1, apply_probability=1.0, token_selection="last")
        with strategy(self.timestamps) as s:
            new_embeddings, _, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            # Both r2 and r3 use the HT and skip the summarized prefix.
            self.assertEqual(attention_mask[3].tolist(), [True, True, False, False, False])
            self.assertEqual(attention_mask[4].tolist(), [True, True, False, False, False])

        strategy = SingleHTStrategy(1, apply_probability=1.0, token_selection="none")
        with strategy(self.timestamps) as s:
            new_embeddings, _, attention_mask = s.apply(self.embeddings, self.timestamps)
            self.assertEqual(new_embeddings.seq_lens.tolist(), [5, 4, 5])
            # Nobody uses the HT, so all tokens attend to the full prefix.
            self.assertEqual(attention_mask[3].tolist(), [False, False, True, False, False])
            self.assertEqual(attention_mask[4].tolist(), [False, False, True, False, False])

    @mock.patch("torch.randn")
    def test_single_strategy_positions(self, mock_randn):
        """The position is sampled uniformly, one token per batch."""
        mock_randn.return_value = torch.tensor([-1.0])  # Token embedding.
        strategy = SingleHTStrategy(1, apply_probability=1.0)

        counts = torch.zeros(4)
        for _ in range(1000):
            with strategy(self.timestamps) as s:
                self.assertTrue(0 <= s.after_position < 4)
                counts[s.after_position] += 1
                new_embeddings, _, _ = s.apply(self.embeddings, self.timestamps)
                # Exactly one extra token for sequences longer than the selected position.
                gt_lens = self.embeddings.seq_lens + (s.after_position < self.embeddings.seq_lens)
                self.assertEqual(new_embeddings.seq_lens.tolist(), gt_lens.tolist())
        self.assertTrue((counts > 150).all(), f"Non-uniform positions: {counts.tolist()}")

    def test_single_strategy_matches_subset(self):
        """SingleHTStrategy must be a drop-in replacement for SubsetHTStrategy with frequency=0."""
        b, l, d = 5, 9, 3
        embeddings = PaddedBatch(torch.randn(b, l, d), torch.tensor([9, 7, 4, 1, 0]))
        timestamps = PaddedBatch(torch.arange(l).float()[None] + torch.arange(b).float()[:, None],
                                 embeddings.seq_lens)

        def run(strategy, position, is_subset, active_tokens, embedding):
            # Pin the sampled position and the set of active tokens for both implementations.
            name, value = ("torch.randperm", torch.tensor([position])) if is_subset else ("torch.randint", torch.tensor([position]))
            with mock.patch(name, return_value=value):
                with mock.patch("torch.rand", side_effect=[torch.tensor(0.0), active_tokens.clone()]):
                    with strategy(timestamps, embedding=embedding) as s:
                        new_embeddings, new_timestamps, attention_mask = s.apply(embeddings, timestamps)
                        return new_embeddings, new_timestamps, attention_mask, s.extract_outputs(new_embeddings)

        def assert_equal(name, subset, single):
            if isinstance(subset, torch.Tensor) or (subset is None):
                self.assertEqual(subset is None, single is None, name)
                if subset is not None:
                    self.assertTrue((subset == single).all(), name)
                return
            self.assertEqual(subset.seq_lens.tolist(), single.seq_lens.tolist(), name)
            payloads = subset.payload if isinstance(subset.payload, dict) else {"": subset.payload}
            for key, value in payloads.items():
                other = single.payload[key] if key else single.payload
                mask = subset.seq_len_mask
                while mask.ndim < value.ndim:
                    mask = mask.unsqueeze(-1)
                same = (value == other) if value.dtype == torch.bool else ((value - other).abs() < 1e-6)
                self.assertTrue(torch.logical_or(same, ~mask).all(), name + key)

        for position in range(l):
            active_tokens = torch.rand(l)
            cases = [dict(token_selection=selection, predict=predict, use_attention_sink=sink)
                     for selection in ["random", "last", "none"]
                     for predict in ["input_tokens", "history_tokens", "all"]
                     for sink in [False, True]]
            cases += [dict(embedding=embedding_type)
                      for embedding_type in ["end_ht", "avg_ht", "avg", "last", "mix_end_ht_avg"]]
            for kwargs in cases:
                embedding = "embedding" in kwargs
                subset = SubsetHTStrategy(d, frequency=0, apply_probability=1.0, **kwargs)
                single = SingleHTStrategy(d, apply_probability=1.0, **kwargs)
                with torch.no_grad():
                    single.token.copy_(subset.token)
                results = [run(subset, position, True, active_tokens, embedding),
                           run(single, position, False, active_tokens, embedding)]
                for field, *values in zip(["embeddings", "timestamps", "attention_mask", "outputs"], *results):
                    assert_equal(f"position={position} {kwargs} {field}: ", *values)

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
