#!/usr/bin/env python3
import math
import torch
from unittest import TestCase, main

from pretpp.optim import CorrHPOptimizer, HPO_STAGE_DOWNSTREAM, HPO_STAGE_FINAL
from pretpp.optim.corrhpo import find_closest_unit_norm


def f1(x):
    return (x - 5) ** 2


def f2(x):
    return (x + 3) ** 2


def loss(x, alpha, beta):
    return alpha * f1(x) + beta * f2(x)


def downstream(x):
    return 0.3 * f1(x) + 0.9 * f2(x)


class TestCorrHPOptimizer(TestCase):
    def test_find_closest_unit_norm(self):
        torch.manual_seed(0)
        # Test random.
        dim = 5
        v = torch.eye(dim).double()
        c = torch.randn(dim, dim).double()
        c = c @ c.T
        v = c @ v
        weights_gt = torch.randn(dim).double()
        weights_gt /= torch.linalg.norm(weights_gt)
        target = v @ weights_gt

        weights = find_closest_unit_norm(v, target)
        self.assertAlmostEqual(torch.linalg.norm(weights - weights_gt).item(), 0, places=2)

        # Test small vector weights.
        # We expect that zero vector will have larger weight.
        dim = 5
        v = torch.eye(dim).double()
        v[0, 0] = 0  # Eye with zero vector.
        weights_gt = torch.randn(dim).double()
        weights_gt /= torch.linalg.norm(weights_gt) * 10
        weights_gt[0] = 1 - (weights_gt[1:] ** 2).sum()
        target = v @ weights_gt

        weights = find_closest_unit_norm(v, target)
        self.assertAlmostEqual(torch.linalg.norm(weights - weights_gt).item(), 0, places=2)

    def test_gradient(self):
        torch.manual_seed(0)
        x = torch.nn.Parameter(torch.randn([]))
        alpha = torch.nn.Parameter(torch.rand([]))
        beta = torch.nn.Parameter(torch.rand([]))

        def closure(down, free, alpha, beta, stage=None):
            optimizer.zero_grad()
            if down > 0:
                v = down * downstream(x)
            else:
                v = 0
            if alpha > 0 or beta > 0:
                v = v + loss(x, alpha, beta)
            v.backward()

        for parametrization in ["sigmoid", "tanh", "abs"]:
            for normalization in ["none", "sum", "norm"]:
                optimizer = CorrHPOptimizer([{"params": [alpha, beta]},
                                             {"params": [x]}],
                                            torch.optim.Adam,
                                            weights_parametrization=parametrization,
                                            weights_normalization=normalization,
                                            eps=0,
                                            lr=0)

                logits = torch.stack([alpha, beta])
                if parametrization == "sigmoid":
                    weights = torch.sigmoid(logits)
                elif parametrization == "tanh":
                    weights = torch.tanh(logits)
                elif parametrization == "abs":
                    weights = torch.abs(logits)
                else:
                    assert parametrization == "linear"
                if normalization == "sum":
                    weights = weights / weights.sum() * len(weights)
                elif normalization == "norm":
                    weights = weights / torch.linalg.norm(weights) * math.sqrt(len(weights))
                else:
                    assert normalization == "none"
                w1, w2 = weights

                grad = 2 * w1 * (x - 5) + 2 * w2 * (x + 3)
                new_x = (x + grad).detach() - grad
                alpha.grad = None
                beta.grad = None
                downstream(new_x).backward()
                alpha_grad_gt = alpha.grad.item()
                beta_grad_gt = beta.grad.item()

                def mock_step(closure):
                    closure()
                    self.assertAlmostEqual(alpha.grad.item(), alpha_grad_gt, places=3)
                    self.assertAlmostEqual(beta.grad.item(), beta_grad_gt, places=3)

                optimizer.base_optimizer.step = mock_step
                optimizer.hpo_step(closure)

    def test_optimizer(self):
        torch.manual_seed(0)
        for algorithm in ["sgd", "closed-form-fast", "closed-form"]:
            for normalization in (["none", "sum", "norm"] if algorithm == "sgd" else ["none", "norm"]):
                for parametrization in (["sigmoid", "tanh", "abs"] if algorithm == "sgd" else ["linear"]):
                    x = torch.nn.Parameter(torch.randn([]))
                    alpha = torch.nn.Parameter(torch.rand([]))
                    beta = torch.nn.Parameter(torch.rand([]))
                    optimizer = CorrHPOptimizer([{"params": [alpha, beta]},
                                                {"params": [x]}],
                                                torch.optim.SGD,
                                                algorithm=algorithm,
                                                weights_parametrization=parametrization,
                                                weights_normalization=normalization,
                                                clip_hp_grad=None if parametrization == "sigmoid" else 0.1,
                                                lr=0.01)

                    for step in range(2000):
                        def closure(down, free, alpha, beta, stage=None):
                            optimizer.zero_grad()
                            if down > 0:
                                v = down * downstream(x)
                            else:
                                v = 0
                            if alpha > 0 or beta > 0 or free > 0:
                                v = v + loss(x, alpha, beta)
                            v.backward()
                        optimizer.hpo_step(closure)
                    final_downstream_loss = downstream(x).item()
                    try:
                        self.assertAlmostEqual(final_downstream_loss, 14.4, places=1)
                    except AssertionError:
                        print(f"Test failed for {algorithm} {parametrization} {normalization}")
                        raise


if __name__ == "__main__":
    main()
