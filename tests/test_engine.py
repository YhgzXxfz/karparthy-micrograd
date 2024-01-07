import unittest

import torch

from core.engine import Value


class EngineTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_sanity_check(self):
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.tanh() + z * x
        h = (z * z).tanh()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        x = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
        z = 2 * x + 2 + x
        q = z.tanh() + z * x
        h = (z * z).tanh()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # forward pass went well
        self.assertEqual(ymg.data, ypt.data.item())
        # backward pass went well
        self.assertEqual(xmg.grad, xpt.grad.item())

    def test_more_ops(self):
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).tanh()
        d += 3 * d + (b - a).tanh()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        a = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
        b = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).tanh()
        d = d + 3 * d + (b - a).tanh()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        # forward pass went well
        self.assertTrue((gmg.data - gpt.data.item()) < tol)
        # backward pass went well
        self.assertTrue(abs(a.grad - a.grad.item()) < tol)
        self.assertTrue(abs(b.grad - b.grad.item()) < tol)
