#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.losses import TimeMAELoss, CrossEntropyLoss
from pretpp.losses import MLMLoss
from pretpp.losses.mlm import EVAL_MASK_FIELD


class TestMLMLoss(TestCase):
    def test_disabled_augmentation(self):
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
        labels = [
            [1, 0, 0, 0],
            [3, 0, 1, 0],
            [0, 0, 0, 0]
        ]
        index = [
            0,
            1,
            2
        ]
        inputs = PaddedBatch({"timestamps": torch.tensor(timestamps),
                              "labels": torch.tensor(labels),
                              "index": torch.tensor(index)},
                             torch.tensor(lengths).long(),
                             seq_names=["timestamps", "labels"])

        # Check no augmentation.
        loss = MLMLoss(losses={"timestamps": TimeMAELoss(),
                               "labels": CrossEntropyLoss(num_classes=5)},
                       mask_token={"timestamps": -1,
                                   "labels": 4},
                       eval_fraction=0.4,
                       mask_prob=0,
                       random_prob=0)
        model_inputs, targets = loss.prepare_batch(inputs)
        self.assertEqual(set(model_inputs.payload), {"timestamps", "labels", "index"})
        self.assertEqual(set(targets.payload), {"timestamps", "labels", EVAL_MASK_FIELD})
        self.assertEqual(model_inputs.seq_lens.tolist(), [3, 2, 0])
        self.assertEqual(targets.seq_lens.tolist(), [3, 2, 0])
        for key in model_inputs.seq_names:
            self.assertTrue(model_inputs.payload[key].allclose(inputs.payload[key][:, 1:]))
            self.assertTrue(targets.payload[key].allclose(inputs.payload[key][:, :-1]))

    def test_augmentation(self):
        b = 500
        l = 1000
        max_dt = 10
        nc = 5
        inputs = PaddedBatch({"timestamps": (torch.rand(b, l) * max_dt).cumsum(1),
                              "labels": torch.randint(0, nc, [b, l])},
                             torch.randint(0, 100, [b]))
        mask_token = {"timestamps": -1,
                      "labels": nc}
        loss = MLMLoss(losses={"timestamps": TimeMAELoss(),
                               "labels": CrossEntropyLoss(num_classes=nc)},
                       mask_token=mask_token,
                       eval_fraction=0.4,
                       mask_prob=0.6,
                       random_prob=0.3)
        model_inputs, targets = loss.prepare_batch(inputs)
        self.assertEqual(set(model_inputs.payload), {"timestamps", "labels"})
        self.assertEqual(set(targets.payload), {"timestamps", "labels", EVAL_MASK_FIELD})
        self.assertEqual(model_inputs.seq_lens.tolist(), (inputs.seq_lens - 1).clip(min=0).tolist())
        self.assertEqual(targets.seq_lens.tolist(), (inputs.seq_lens - 1).clip(min=0).tolist())
        mask = inputs.seq_len_mask
        inputs_mask = mask[:, 1:]
        atol = 0.01
        for key in model_inputs.seq_names:
            model_input = model_inputs.payload[key]
            equal_fraction = model_input.isclose(inputs.payload[key][:, 1:])[inputs_mask].float().mean().item()
            equal_prob = 1 - 0.6 * 0.4 - 0.3 * 0.4
            self.assertGreater(equal_fraction, equal_prob - atol)

            masking_mask = model_input.isclose(torch.tensor([mask_token[key]], dtype=model_input.dtype))
            masking_fraction = masking_mask[inputs_mask].float().mean().item()
            self.assertAlmostEqual(masking_fraction, 0.6 * 0.4, delta=atol)

            orig_values = set(inputs.payload[key][mask].unique().tolist())
            new_values = set(model_input[inputs_mask].unique().tolist())
            self.assertEqual(new_values - orig_values, {mask_token[key]})

            self.assertTrue(targets.payload[key].allclose(inputs.payload[key][:, :-1]))

    def test_per_field_mlm(self):
        b = 500
        l = 1000
        max_dt = 10
        nc = 1000000
        inputs = PaddedBatch({"timestamps": (torch.rand(b, l) * max_dt).cumsum(1),
                              "labels": torch.randint(0, nc, [b, l])},
                             torch.randint(0, 100, [b]))
        mask_token = {"timestamps": -1,
                      "labels": nc}
        field_mask_probs = {"timestamps": 0.3,
                            "labels": 0.6}
        loss = MLMLoss(losses={"timestamps": TimeMAELoss(),
                               "labels": CrossEntropyLoss(num_classes=nc)},
                       mask_token=mask_token,
                       eval_fraction=0.4,
                       mask_prob=0.6,
                       random_prob=0.3,
                       field_mask_probs=field_mask_probs)
        model_inputs, targets = loss.prepare_batch(inputs)
        self.assertEqual(set(model_inputs.payload), {"timestamps", "labels"})
        self.assertEqual(set(targets.payload), {"timestamps", "labels", EVAL_MASK_FIELD})
        self.assertEqual(model_inputs.seq_lens.tolist(), (inputs.seq_lens - 1).clip(min=0).tolist())
        self.assertEqual(targets.seq_lens.tolist(), (inputs.seq_lens - 1).clip(min=0).tolist())
        mask = inputs.seq_len_mask
        inputs_mask = mask[:, 1:]
        atol = 0.01
        for key in model_inputs.seq_names:
            model_input = model_inputs.payload[key]
            equal_fraction = model_input.isclose(inputs.payload[key][:, 1:])[inputs_mask].float().mean().item()
            equal_prob = 1 - (0.6 + 0.3) * 0.4 * field_mask_probs[key]
            self.assertAlmostEqual(equal_fraction, equal_prob, delta=atol)

            masking_mask = model_input.isclose(torch.tensor([mask_token[key]], dtype=model_input.dtype))
            masking_fraction = masking_mask[inputs_mask].float().mean().item()
            self.assertAlmostEqual(masking_fraction, 0.6 * 0.4 * field_mask_probs[key], delta=atol)

    def test_convergence(self):
        lengths = [
            4,
            3,
            0
        ]
        timestamps = [
            [0, 1.5, 3.5, 4],
            [5, 6.5, 7, -1],
            [5, 5, 5, 5]
        ]
        labels = [
            [1, 4, 0, 0],
            [3, 0, 1, 0],
            [0, 0, 0, 0]
        ]
        inputs = PaddedBatch({"timestamps": torch.tensor(timestamps),
                              "labels": torch.tensor(labels)},
                             torch.tensor(lengths).long())

        loss = MLMLoss(losses={"timestamps": TimeMAELoss(),
                               "labels": CrossEntropyLoss(num_classes=5)},
                       mask_token={"timestamps": -1,
                                   "labels": 4})
        timestamps_prediction = torch.randn(3, 4, requires_grad=True)
        labels_prediction = torch.randn(3, 4, 5, requires_grad=True)
        optimizer = torch.optim.Adam([timestamps_prediction, labels_prediction], lr=0.01)
        for step in range(1000):
            _, targets = loss.prepare_batch(inputs)
            values, _ = loss({"timestamps": timestamps_prediction[:, 1:, None], "labels": labels_prediction[:, 1:]}, targets)
            value = sum(values.values())
            if step == 0:
                print("Init losses:", values)
                print("Init loss:", value.item())
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
        print("Final losses:", values)
        print("Final loss:", value.item())

        mask = inputs.seq_len_mask[:, 1:-1]
        deltas = inputs.payload["timestamps"]
        deltas = deltas[:, 1:] - deltas[:, :-1]
        delta = torch.linalg.norm((timestamps_prediction[:, 1:-1] - deltas[:, :-1])[mask])
        self.assertLess(delta, 0.1)
        self.assertEqual(labels_prediction.argmax(-1)[:, 1:-1][mask].tolist(), inputs.payload["labels"][:, 1:-1][mask].tolist())


if __name__ == "__main__":
    main()
