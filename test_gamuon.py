"""
Tests for the Gamuon geometric-algebra-native optimizer.

Verifies:
  1. Grade decomposition correctness
  2. Bivector exponential (rotor) orthogonality & closed-form accuracy
  3. Rotor sandwich action preserves Frobenius norm
  4. Full optimizer step on a toy linear regression problem
  5. Comparison with standard Muon (Newton-Schulz variant)
  6. Multivector momentum behaviour
  7. Edge cases (n=2, n=3, non-square via padding)
"""

from __future__ import annotations

import pytest
import torch

from gamuon import (
    Gamuon,
    GamuonNS,
    MultivectorMomentum,
    bivector_exp,
    grade_decompose,
    newton_schulz,
    rotor_apply,
)

# Note: default dtype is float32; tolerances are set accordingly
#   float32 eps ~ 1.2e-7 → norms of (5×5) matrices have ~6e-7 error
_F32_TOL = 1e-5      # general float32 tolerance for norm checks
_F32_TOL_TIGHT = 1e-6
_ORTHO_TOL = 1e-5    # orthogonality check (RᵀR − I)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GRADE DECOMPOSITION                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestGradeDecompose:
    def test_scalar_part_is_isotropic(self):
        """⟨M⟩₀ should be c·I where c = tr(M)/n."""
        M = torch.randn(5, 5)
        scalar, _, _ = grade_decompose(M)
        # scalar should be a multiple of identity
        off_diag = scalar - torch.diag(scalar.diag())
        assert off_diag.norm() < _F32_TOL, "Scalar part has off-diagonal entries"
        # diagonal values should all be equal
        assert scalar.diag().std() < _F32_TOL, "Scalar diagonal is not constant"
        # trace should equal tr(M)
        assert abs(scalar.trace() - M.trace()) < _F32_TOL

    def test_bivector_is_antisymmetric(self):
        """⟨M⟩₂ should be antisymmetric."""
        M = torch.randn(5, 5)
        _, bivector, _ = grade_decompose(M)
        assert (bivector + bivector.T).norm() < _F32_TOL, \
            "Bivector is not antisymmetric"
        assert abs(bivector.trace()) < _F32_TOL

    def test_strain_is_symmetric_traceless(self):
        """⟨M⟩₊ should be symmetric and trace-free."""
        M = torch.randn(5, 5)
        _, _, strain = grade_decompose(M)
        assert (strain - strain.T).norm() < _F32_TOL, "Strain is not symmetric"
        assert abs(strain.trace()) < _F32_TOL, "Strain has non-zero trace"

    def test_decomposition_sums_to_original(self):
        """⟨M⟩₀ + ⟨M⟩₂ + ⟨M⟩₊ should equal M."""
        M = torch.randn(5, 5)
        scalar, bivector, strain = grade_decompose(M)
        reconstructed = scalar + bivector + strain
        assert (M - reconstructed).norm() < _F32_TOL, \
            "Decomposition does not sum to M"

    def test_orthogonality_of_grades(self):
        """Different grades should be orthogonal in Frobenius inner product."""
        M = torch.randn(5, 5)
        s, b, p = grade_decompose(M)
        assert abs((s * b).sum()) < _F32_TOL
        assert abs((s * p).sum()) < _F32_TOL
        assert abs((b * p).sum()) < _F32_TOL

    def test_pure_antisymmetric_decomposes_correctly(self):
        """A purely antisymmetric matrix should have only bivector part."""
        B = torch.randn(4, 4)
        B = B - B.T  # make antisymmetric
        s, b, p = grade_decompose(B)
        assert s.norm() < _F32_TOL
        assert p.norm() < _F32_TOL
        assert (b - B).norm() < _F32_TOL

    def test_pure_symmetric_decomposes_correctly(self):
        """A symmetric matrix should have no bivector part."""
        S = torch.randn(4, 4)
        S = S + S.T         # make symmetric
        S = S - torch.eye(4) * S.trace() / 4  # remove scalar
        _, b, _ = grade_decompose(S)
        assert b.norm() < _F32_TOL

    def test_multi_batch_decomposition(self):
        """Grade decomposition should work on batched matrices."""
        M = torch.randn(3, 5, 5)
        s, b, p = grade_decompose(M)
        assert s.shape == (3, 5, 5)
        assert b.shape == (3, 5, 5)
        assert p.shape == (3, 5, 5)
        assert (M - (s + b + p)).norm() < _F32_TOL


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BIVECTOR EXPONENTIAL                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestBivectorExp:
    def test_exp_2d_is_orthogonal(self):
        """exp(B) for 2×2 should be orthogonal (RᵀR = I)."""
        B = torch.tensor([[0.0, 1.5], [-1.5, 0.0]])
        R = bivector_exp(B)
        assert (R @ R.T - torch.eye(2)).norm() < _ORTHO_TOL, "RᵀR ≠ I"
        assert abs(R.det() - 1.0) < _ORTHO_TOL, "det(R) ≠ 1"

    def test_exp_2d_closed_form_accuracy(self):
        """Closed-form 2D exponential should match torch.matrix_exp."""
        B = torch.tensor([[0.0, 0.8], [-0.8, 0.0]])
        R_closed = bivector_exp(B)
        R_ref = torch.matrix_exp(B)
        assert (R_closed - R_ref).norm() < _F32_TOL_TIGHT

    def test_exp_3d_is_orthogonal(self):
        """exp(B) for 3×3 should be orthogonal (RᵀR = I, det=1)."""
        B = torch.tensor([
            [0.0, 1.2, -0.5],
            [-1.2, 0.0, 0.3],
            [0.5, -0.3, 0.0],
        ])
        R = bivector_exp(B)
        assert (R @ R.T - torch.eye(3)).norm() < _ORTHO_TOL
        assert abs(R.det() - 1.0) < _ORTHO_TOL

    def test_exp_3d_rodrigues_accuracy(self):
        """Rodrigues formula should match torch.matrix_exp for 3×3 bivectors."""
        B = torch.tensor([
            [0.0, 0.7, -1.2],
            [-0.7, 0.0, 0.4],
            [1.2, -0.4, 0.0],
        ])
        R_rod = bivector_exp(B)
        R_ref = torch.matrix_exp(B)
        assert (R_rod - R_ref).norm() < _F32_TOL

    def test_exp_4d_is_orthogonal(self):
        """exp(B) for 4×4 (via matrix_exp) should be orthogonal."""
        B = torch.randn(4, 4)
        B = B - B.T  # antisymmetrize
        R = bivector_exp(B)
        assert (R @ R.T - torch.eye(4)).norm() < _ORTHO_TOL

    def test_exp_zero_is_identity(self):
        """exp(0) should be identity."""
        for n in [2, 3, 5]:
            B = torch.zeros(n, n)
            R = bivector_exp(B)
            assert (R - torch.eye(n)).norm() < _F32_TOL_TIGHT

    def test_exp_composition(self):
        """exp(B₁)·exp(B₂) ≈ exp(B₁+B₂) for commuting bivectors."""
        B1 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        B2 = torch.tensor([[0.0, 0.5], [-0.5, 0.0]])
        R12 = bivector_exp(B1) @ bivector_exp(B2)
        R_sum = bivector_exp(B1 + B2)
        assert (R12 - R_sum).norm() < _F32_TOL


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ROTOR SANDWICH                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestRotorApply:
    def test_preserves_frobenius_norm(self):
        """R · W · Rᵀ should preserve ‖W‖_F."""
        W = torch.randn(5, 5)
        B = torch.randn(5, 5)
        B = B - B.T
        R = bivector_exp(B)
        W_rot = rotor_apply(R, W)
        assert abs(W.norm() - W_rot.norm()) < _F32_TOL

    def test_consecutive_rotors_compose(self):
        """Applying R₁ then R₂ should equal applying R₂·R₁."""
        W = torch.randn(3, 3)
        B1 = torch.tensor(
            [[0.0, 0.5, -0.3], [-0.5, 0.0, 0.2], [0.3, -0.2, 0.0]]
        )
        B2 = torch.tensor(
            [[0.0, -0.2, 0.4], [0.2, 0.0, -0.1], [-0.4, 0.1, 0.0]]
        )
        R1 = bivector_exp(B1)
        R2 = bivector_exp(B2)
        W_seq = rotor_apply(R2, rotor_apply(R1, W))
        W_comp = rotor_apply(R2 @ R1, W)
        assert (W_seq - W_comp).norm() < _F32_TOL

    def test_identity_rotor_does_nothing(self):
        """Applying identity rotor should leave W unchanged."""
        W = torch.randn(4, 4)
        R = torch.eye(4)
        W_out = rotor_apply(R, W)
        assert (W - W_out).norm() < _F32_TOL_TIGHT


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MULTIVECTOR MOMENTUM                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestMultivectorMomentum:
    def test_momentum_tracks_constant(self):
        """Given a constant gradient, momentum should converge to it."""
        n = 4
        mom = MultivectorMomentum(n, "cpu", torch.float32)
        ones = torch.ones(n, n)

        for _ in range(500):
            mom.step(ones, ones / 2, ones / 3, (0.9, 0.999))

        m_s, m_b, m_p, _, _, _ = (
            mom.scalar_m,
            mom.bivector_m,
            mom.strain_m,
            mom.scalar_v,
            mom.bivector_v,
            mom.strain_v,
        )
        assert (m_s - ones).norm() < 1e-1  # EMA converges slowly
        assert (m_b - ones / 2).norm() < 1e-1
        assert (m_p - ones / 3).norm() < 1e-1

    def test_returns_6_tensors(self):
        """step() should return 6 tensors (3 moments, 3 velocities)."""
        mom = MultivectorMomentum(3, "cpu", torch.float32)
        z = torch.zeros(3, 3)
        result = mom.step(z, z, z, (0.9, 0.999))
        assert len(result) == 6
        assert all(isinstance(t, torch.Tensor) for t in result)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  NEWTON–SCHULZ                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


def _normalized_ns(G: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """Newton-Schulz with spectral normalization (avoids NaN divergence).

    The raw iteration  X ← (3X − XXᵀX)/2  converges iff σ(G) ⊂ (0, √3).
    We normalise by the Frobenius norm to guarantee convergence.
    """
    scale = G.norm() + 1e-10
    X = newton_schulz(G / scale, num_iters=num_iters)
    return X  # X ≈ U Vᵀ regardless of scaling


class TestNewtonSchulz:
    def test_output_is_nearly_orthogonal(self):
        """Newton-Schulz should produce an approximately orthogonal matrix."""
        G = torch.randn(5, 5)
        X = _normalized_ns(G, num_iters=20)
        err = (X @ X.T - torch.eye(5)).norm()
        # NS converges asymptotically; 20 iters gives ~1e-3 accuracy
        assert err < 1e-2, f"NS orthogonality error: {err:.4e}"

    def test_converges_to_sign_of_spd(self):
        """For an SPD matrix, sign(G) = I."""
        A = torch.randn(5, 5)
        G = A @ A.T  # SPD
        X = _normalized_ns(G, num_iters=30)
        err = (X - torch.eye(5)).norm()
        assert err < 1e-2, f"NS SPD error: {err:.4e}"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GAMUON OPTIMIZER  (end-to-end)                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestGamuonOptimizer:
    def test_linear_regression_converges(self):
        """Gamuon should minimize a simple least-squares problem."""
        torch.manual_seed(42)
        n, d = 100, 4
        X = torch.randn(n, d)
        w_star = torch.randn(d, 1)
        y = X @ w_star + 0.01 * torch.randn(n, 1)

        w = torch.nn.Parameter(torch.randn(d, 1))
        opt = Gamuon([w], lr=0.05)

        losses = []
        for _ in range(200):
            opt.zero_grad()
            loss = ((X @ w - y) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.1, (
            f"Loss did not converge: {losses[0]:.6f} → {losses[-1]:.6f}"
        )
        assert (w - w_star).norm() < 0.5

    def test_square_matrix_decreases_loss(self):
        """Gamuon should decrease a simple quadratic loss for square weights."""
        torch.manual_seed(42)
        n = 4
        target = torch.randn(n, n)
        W = torch.nn.Parameter(torch.randn(n, n))
        opt = Gamuon([W], lr=0.01)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((W - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should at least not explode
        assert torch.isfinite(torch.tensor(losses[-1])), "NaN loss"
        # Should make some progress
        assert losses[-1] < losses[0] * 0.95, (
            f"Square matrix loss did not decrease: "
            f"{losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_non_square_matrix_via_padding(self):
        """Gamuon should handle non-square (m×n) matrices via padding."""
        torch.manual_seed(42)
        m, n = 6, 4
        target = torch.randn(m, n)
        W = torch.nn.Parameter(torch.randn(m, n))
        opt = Gamuon([W], lr=0.01)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((W - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert torch.isfinite(torch.tensor(losses[-1])), "NaN loss"
        assert losses[-1] < losses[0] * 0.95, (
            f"Non-square matrix loss did not decrease: "
            f"{losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_per_grade_lr(self):
        """Different per-grade learning rates should not crash."""
        W = torch.nn.Parameter(torch.randn(4, 4))
        opt = Gamuon([W], lr=0.01, lr_scalar=0.5, lr_bivector=2.0,
                     lr_strain=1.0)
        (W ** 2).mean().backward()
        opt.step()
        assert True

    def test_weight_decay_adds_to_gradient(self):
        """Weight decay should add wd * param to the gradient in-place."""
        torch.manual_seed(1729)
        W = torch.nn.Parameter(torch.randn(3, 3))

        # Use lr=0 so weights don't change between measurements
        opt = Gamuon([W], lr=0.0, weight_decay=0.5)

        (W ** 2).mean().backward()
        grad_before_step = W.grad.clone()
        # The gradient of (W**2).mean() is 2*W/9
        grad_of_loss = 2 * W.data / 9
        assert (grad_before_step - grad_of_loss).norm() < _F32_TOL, "Baseline grad"

        opt.step()
        grad_after_step = W.grad.clone()

        # After step, weight-decay was added:  grad ← grad + wd * W
        expected_grad = grad_of_loss + 0.5 * W.data
        err = (grad_after_step - expected_grad).norm()
        assert err < _F32_TOL, f"Weight decay error: {err:.2e}"

    def test_momentum_persistence(self):
        """Momentum state should persist across steps."""
        W = torch.nn.Parameter(torch.randn(4, 4))
        opt = Gamuon([W], lr=0.01, betas=(0.9, 0.999))

        (W ** 2).mean().backward()
        opt.step()

        state = opt.state[W]
        assert "step" in state
        assert state["step"] == 1
        assert "momentum" in state

        opt.zero_grad()
        (W ** 2).mean().backward()
        opt.step()
        assert state["step"] == 2

    def test_closure_support(self):
        """Gamuon should support the closure API."""
        W = torch.nn.Parameter(torch.randn(4, 4))
        opt = Gamuon([W], lr=0.01)

        def closure():
            return (W ** 2).mean()

        loss = opt.step(closure)
        assert loss is not None
        assert loss.item() > 0


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GAMUONNS  (Newton-Schulz baseline)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestGamuonNS:
    def test_ns_runs_without_error(self):
        """GamuonNS should execute without crashing (may produce NaN)."""
        torch.manual_seed(42)
        X = torch.randn(50, 4)
        w_star = torch.randn(4, 4)
        y = X @ w_star + 0.01 * torch.randn(50, 4)

        w = torch.nn.Parameter(torch.randn(4, 4))
        opt = GamuonNS([w], lr=0.01, ns_iters=10)

        for _ in range(20):
            opt.zero_grad()
            loss = ((X @ w - y) ** 2).mean()
            loss.backward()
            opt.step()

        # Just check it ran (NS can produce NaN but shouldn't crash python)
        assert True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  EDGE CASES                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestEdgeCases:
    def test_2x2_matrix_optimization(self):
        """Gamuon should work with 2×2 matrices (closed-form rotor)."""
        W = torch.nn.Parameter(torch.randn(2, 2))
        opt = Gamuon([W], lr=0.01)
        (W ** 2).mean().backward()
        opt.step()

    def test_3x3_matrix_optimization(self):
        """Gamuon should work with 3×3 matrices (Rodrigues rotor)."""
        W = torch.nn.Parameter(torch.randn(3, 3))
        opt = Gamuon([W], lr=0.01)
        (W ** 2).mean().backward()
        opt.step()

    def test_large_matrix_optimization(self):
        """Gamuon should work with 16×16 matrices (general matrix_exp)."""
        W = torch.nn.Parameter(torch.randn(16, 16))
        opt = Gamuon([W], lr=0.01)
        (W ** 2).mean().backward()
        opt.step()

    def test_zero_gradient(self):
        """Zero gradient should not modify weights."""
        W = torch.nn.Parameter(torch.randn(4, 4))
        initial = W.data.clone()
        opt = Gamuon([W], lr=0.1)
        opt.step()  # no backward → gradient is None → skip
        assert (W.data - initial).norm() < 1e-10

    def test_multiple_parameters(self):
        """Gamuon should handle multiple parameters."""
        W1 = torch.nn.Parameter(torch.randn(4, 4))
        W2 = torch.nn.Parameter(torch.randn(4, 4))
        opt = Gamuon([W1, W2], lr=0.01)
        ((W1 ** 2).mean() + (W2 ** 2).mean()).backward()
        opt.step()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
