import unittest

import torch

from ouro_pipeline.projection_utils import project_activations
from ouro_pipeline.subspace import update_basis


torch.manual_seed(7)


class LoopDepthMathTests(unittest.TestCase):
    def test_01_local_projection(self):
        u, _ = torch.linalg.qr(torch.randn(6, 2))
        e = torch.randn(4, 6) @ (torch.eye(6) - u @ u.T)
        self.assertLess(float((e @ u).abs().max()), 1e-5)

    def test_03_invertible_nullity(self):
        w = torch.randn(5, 5) + 3 * torch.eye(5)
        x = torch.randn(5, 3)
        self.assertEqual(int(torch.linalg.matrix_rank(x)), int(torch.linalg.matrix_rank(w @ x)))

    def test_04_path_intersection_shrinks(self):
        w, x = torch.randn(5, 5), torch.randn(5, 1)
        rank1 = torch.linalg.matrix_rank(x)
        rank3 = torch.linalg.matrix_rank(torch.cat([x, w @ x, w @ w @ x], dim=1))
        self.assertGreaterEqual(int(rank3), int(rank1))

    def test_05_delayed_effect(self):
        w = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 1., 1.]])
        e = torch.diag(torch.tensor([0., 0., 0.2]))
        x = torch.tensor([0., 1., 0.])
        self.assertTrue(torch.allclose((w + e) @ x, w @ x))
        self.assertFalse(torch.allclose((w + e) @ (w @ x), w @ (w @ x)))

    def test_06_noncommutative_power_identity(self):
        w, e = torch.randn(4, 4), torch.randn(4, 4) * 0.1
        rhs = sum(torch.linalg.matrix_power(w + e, 2 - j) @ e @ torch.linalg.matrix_power(w, j) for j in range(3))
        self.assertTrue(torch.allclose(torch.linalg.matrix_power(w + e, 3) - torch.linalg.matrix_power(w, 3), rhs, atol=1e-5))

    def test_07_column_vectorization_jacobian(self):
        a, e, b = torch.randn(2, 3), torch.randn(3, 4), torch.randn(4, 2)
        left = (a @ e @ b).T.reshape(-1)
        vec_e = e.T.reshape(-1)
        right = torch.kron(b.T.contiguous(), a.contiguous()) @ vec_e
        self.assertTrue(torch.allclose(left, right, atol=1e-5))

    def test_08_cross_loop_cancellation(self):
        w = torch.diag(torch.tensor([1., -1.]))
        d = torch.tensor([[0., 1.], [1., 0.]])
        self.assertTrue(torch.allclose(w @ d + d @ w, torch.zeros(2, 2)))
        self.assertTrue(torch.allclose((w + 0.1 * d) @ (w + 0.1 * d) - w @ w, 0.01 * torch.eye(2), atol=1e-6))

    def test_09_nonlinear_recurrence_finite_difference(self):
        w, h = torch.randn(3, 3) * 0.2, torch.randn(3)
        direction, eps = torch.randn(3, 3), 1e-4
        def run(weight):
            state = h
            for _ in range(2):
                state = torch.tanh(weight @ state)
            return state
        _, jvp = torch.autograd.functional.jvp(run, w, direction)
        finite = (run(w + eps * direction) - run(w - eps * direction)) / (2 * eps)
        self.assertTrue(torch.allclose(jvp, finite, atol=2e-3, rtol=2e-3))

    def test_10_ideal_projection_descent(self):
        j = torch.randn(2, 6)
        projector = torch.eye(6) - torch.linalg.pinv(j) @ j
        gradient = torch.randn(6)
        delta = -0.01 * projector @ gradient
        self.assertLess(float(gradient @ delta), 1e-7)
        self.assertLess(float(torch.linalg.vector_norm(j @ delta)), 1e-5)

    def test_11_projection_is_orthogonal_to_basis(self):
        u, _ = torch.linalg.qr(torch.randn(6, 2))
        projected = project_activations(torch.randn(3, 6), u)
        self.assertLess(float(torch.linalg.vector_norm(projected @ u)), 1e-5)

    def test_12_incremental_basis_preserves_history(self):
        old, _ = torch.linalg.qr(torch.randn(8, 2))
        new, report = update_basis(torch.randn(32, 8), old, max_rank=5, residual_tolerance=0.0)
        self.assertEqual(report["new_rank"], 5)
        overlap = torch.linalg.svdvals(old.T @ new)
        self.assertTrue(torch.allclose(overlap, torch.ones_like(overlap), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
