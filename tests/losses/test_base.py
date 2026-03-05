#!/usr/bin/env python3
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from pretpp.losses.base import BaseLoss


class TestSelectEmbeddingsByMask(TestCase):
    def _make_embeddings(self, payload, lengths):
        return PaddedBatch(torch.tensor(payload, dtype=torch.float), torch.tensor(lengths))

    def test_basic(self):
        # B=2, L=4, D=2
        payload = [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [13, 14], [0, 0]],
        ]
        lengths = [4, 3]
        embeddings = self._make_embeddings(payload, lengths)

        # Select positions 0,2 from row 0; position 1 from row 1
        mask = torch.tensor([
            [True, False, True, False],
            [False, True, False, False],
        ])

        result = BaseLoss.select_embeddings_by_mask(embeddings, mask)

        self.assertEqual(result.payload.shape, (2, 2, 2))
        self.assertTrue(result.seq_lens.tolist() == [2, 1])

        # Row 0: positions 0 and 2 → [1,2] and [5,6]
        torch.testing.assert_close(result.payload[0, :2], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
        # Row 1: position 1 → [11,12]
        torch.testing.assert_close(result.payload[1, :1], torch.tensor([[11.0, 12.0]]))

    def test_mask_respects_seq_len(self):
        # Mask True at a padding position — should be ignored
        payload = [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [0, 0], [0, 0]],
        ]
        lengths = [4, 2]
        embeddings = self._make_embeddings(payload, lengths)

        # Row 1 mask has True at positions 2,3 which are padding
        mask = torch.tensor([
            [True, False, False, False],
            [False, False, True, True],
        ])

        result = BaseLoss.select_embeddings_by_mask(embeddings, mask)

        self.assertEqual(result.seq_lens.tolist(), [1, 0])
        torch.testing.assert_close(result.payload[0, :1], torch.tensor([[1.0, 2.0]]))

    def test_uniform_selection(self):
        # All rows select the same number of elements
        payload = [
            [[1, 0], [2, 0], [3, 0]],
            [[4, 0], [5, 0], [6, 0]],
        ]
        lengths = [3, 3]
        embeddings = self._make_embeddings(payload, lengths)

        mask = torch.tensor([
            [True, True, False],
            [False, True, True],
        ])

        result = BaseLoss.select_embeddings_by_mask(embeddings, mask)

        self.assertEqual(result.payload.shape, (2, 2, 2))
        self.assertEqual(result.seq_lens.tolist(), [2, 2])
        torch.testing.assert_close(result.payload[0], torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
        torch.testing.assert_close(result.payload[1], torch.tensor([[5.0, 0.0], [6.0, 0.0]]))

    def test_invalid_mask_ndim(self):
        payload = [[[1, 2]]]
        embeddings = self._make_embeddings(payload, [1])
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            BaseLoss.select_embeddings_by_mask(embeddings, mask)


if __name__ == "__main__":
    main()
