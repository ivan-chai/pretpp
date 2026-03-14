#!/usr/bin/env python3
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from pretpp.nn.layers.head import StoreGradients, IdentityHead


class TestStoreGradients(TestCase):
    def _make_batch(self, b=2, l=3, d=4):
        payload = torch.randn(b, l, d, requires_grad=True)
        lengths = torch.tensor([l] * b)
        return PaddedBatch(payload, lengths), payload

    def _make_module(self, d=4):
        return StoreGradients(IdentityHead(d))

    def test_forward_is_identity(self):
        module = self._make_module()
        batch, payload = self._make_batch()
        out = module(batch)
        torch.testing.assert_close(out.payload, payload)
        self.assertTrue((out.seq_lens == batch.seq_lens).all())

    def test_forward_returns_copy_of_batch(self):
        module = self._make_module()
        batch, _ = self._make_batch()
        out = module(batch)
        self.assertIsNot(out, batch)

    def test_input_grads_none_before_backward(self):
        module = self._make_module()
        self.assertIsNone(module.input_grads)
        batch, _ = self._make_batch()
        module(batch)
        self.assertIsNone(module.input_grads)

    def test_input_grads_stored_after_backward(self):
        module = self._make_module()
        batch, payload = self._make_batch()
        out = module(batch)
        loss = out.payload.sum()
        loss.backward()
        self.assertIsNotNone(module.input_grads)
        # gradient of sum w.r.t. each element is 1
        torch.testing.assert_close(module.input_grads, torch.ones_like(payload))

    def test_input_grads_match_upstream_gradient(self):
        module = self._make_module()
        batch, payload = self._make_batch(b=2, l=3, d=4)
        out = module(batch)
        # Use a non-trivial upstream gradient
        upstream = torch.randn_like(out.payload)
        out.payload.backward(upstream)
        torch.testing.assert_close(module.input_grads, upstream)

    def test_gradient_flows_to_input(self):
        module = self._make_module()
        batch, payload = self._make_batch()
        out = module(batch)
        loss = out.payload.sum()
        loss.backward()
        self.assertIsNotNone(payload.grad)
        torch.testing.assert_close(payload.grad, torch.ones_like(payload))

    def test_seq_lens_preserved(self):
        module = self._make_module(d=5)
        lengths = torch.tensor([3, 2, 1])
        payload = torch.randn(3, 3, 5, requires_grad=True)
        batch = PaddedBatch(payload, lengths)
        out = module(batch)
        self.assertTrue((out.seq_lens == lengths).all())

    def test_double_backward_raises(self):
        module = self._make_module()
        batch, _ = self._make_batch()
        out = module(batch)
        out.payload.sum().backward(retain_graph=True)
        with self.assertRaises(NotImplementedError):
            out.payload.sum().backward(retain_graph=True)

    def test_second_forward_after_reset_works(self):
        module = self._make_module()
        batch, payload = self._make_batch()
        out = module(batch)
        out.payload.sum().backward()
        first_grads = module.input_grads.clone()

        # Reset and run again
        module.input_grads = None
        batch2, payload2 = self._make_batch()
        out2 = module(batch2)
        upstream2 = torch.randn_like(out2.payload)
        out2.payload.backward(upstream2)
        torch.testing.assert_close(module.input_grads, upstream2)
        self.assertFalse(torch.equal(module.input_grads, first_grads))


if __name__ == "__main__":
    main()
