#!/usr/bin/env python3
import torch
from unittest import TestCase, main

from pretpp.optim import CorrHPOptimizer


def f1(x):
    return (x - 5) ** 2


def f2(x):
    return (x + 3) ** 2


def loss(x, alpha, beta):
    return alpha * f1(x) + beta * f2(x)


def downstream(x):
    return 0.3 * f1(x) + 0.9 * f2(x)


class TestCorrHPOptimizer(TestCase):
    def test_gradient(self):
        torch.manual_seed(0)
        x = torch.nn.Parameter(torch.randn([]))
        alpha = torch.nn.Parameter(torch.rand([]))
        beta = torch.nn.Parameter(torch.rand([]))
        optimizer = CorrHPOptimizer([{"params": [alpha, beta]},
                                    {"params": [x]}],
                                    torch.optim.Adam,
                                    normalize_weights="softmax",
                                    lr=0.01)

        w1, w2 = torch.nn.functional.softmax(torch.stack([alpha, beta]), 0)
        grad = 2 * w1 * (x - 5) + 2 * w2 * (x + 3)
        new_x = (x + grad).detach() - grad
        downstream(new_x).backward()
        alpha_grad_gt = alpha.grad.item()
        beta_grad_gt = beta.grad.item()

        def closure(down, alpha, beta, final=False):
            optimizer.zero_grad()
            if down > 0:
                v = down * downstream(x)
            else:
                v = 0
            if alpha > 0 or beta > 0:
                v = v + loss(x, alpha, beta)
            v.backward()

        def mock_step(closure):
            closure()
            self.assertAlmostEqual(alpha.grad.item(), alpha_grad_gt, places=3)
            self.assertAlmostEqual(beta.grad.item(), beta_grad_gt, places=3)

        optimizer.base_optimizer.step = mock_step
        optimizer.hpo_step(closure)

    def test_optimizer(self):
        torch.manual_seed(0)
        for normalize in ["none", "sigmoid", "softmax"]:
            x = torch.nn.Parameter(torch.randn([]))
            alpha = torch.nn.Parameter(torch.rand([]))
            beta = torch.nn.Parameter(torch.rand([]))
            optimizer = CorrHPOptimizer([{"params": [alpha, beta]},
                                        {"params": [x]}],
                                        torch.optim.Adam,
                                        normalize_weights=normalize,
                                        lr=0.01)

            for step in range(1000):
                def closure(down, alpha, beta, final=False):
                    optimizer.zero_grad()
                    if down > 0:
                        v = down * downstream(x)
                    else:
                        v = 0
                    if alpha > 0 or beta > 0:
                        v = v + loss(x, alpha, beta)
                    v.backward()
                optimizer.hpo_step(closure)
            final_downstream_loss = downstream(x).item()
            self.assertAlmostEqual(final_downstream_loss, 14.4, places=3)


if __name__ == "__main__":
    main()
