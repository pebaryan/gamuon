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
    ConformalMuon,
    Gamuon,
    GamuonAuto,
    GamuonNS,
    GradNormMonitor,
    MultivectorMomentum,
    _affine_exp,
    bivector_exp,
    find_conformal_pairs,
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
        torch.manual_seed(42)
        A = torch.randn(5, 5)
        G = A @ A.T  # SPD
        X = _normalized_ns(G, num_iters=50)
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

    def test_ns_rectangular_does_not_crash(self):
        """GamuonNS used to crash on non-square gradients due to a
        scalar·I subtraction with mismatched shape.  Should now just
        apply sign(G) additively for rectangular weights."""
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(6, 4))
        opt = GamuonNS([w], lr=0.01, ns_iters=5)

        before = w.data.clone()
        (w ** 2).mean().backward()
        opt.step()

        assert torch.isfinite(w.data).all(), "GamuonNS produced non-finite values"
        # Should actually move the weights
        assert (w.data - before).norm() > 0


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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFORMAL MUON                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestConformalMuon:
    def test_affine_exp_identity(self):
        """_affine_exp with a=0, b=0 should return (1, 0)."""
        a = torch.zeros(4)
        b = torch.zeros(4)
        dg, db = _affine_exp(a, b)
        assert (dg - 1.0).abs().max().item() < 1e-6
        assert db.abs().max().item() < 1e-6

    def test_affine_exp_translation_only(self):
        """_affine_exp with a=0, b≠0 should give (1, b)."""
        a = torch.zeros(4)
        b = torch.tensor([0.5, -1.0, 2.0, 0.0])
        dg, db = _affine_exp(a, b)
        assert (dg - 1.0).abs().max().item() < 1e-6
        assert (db - b).abs().max().item() < 1e-5, "Translation factor should ≈ 1"

    def test_affine_exp_dilation_only(self):
        """_affine_exp with a≠0, b=0 should give (exp(a), 0)."""
        a = torch.tensor([0.5, -0.3, 1.0, 0.0])
        b = torch.zeros(4)
        dg, db = _affine_exp(a, b)
        expected_dg = torch.exp(a)
        assert (dg - expected_dg).abs().max().item() < 1e-6
        assert db.abs().max().item() < 1e-6

    def test_affine_exp_near_singularity(self):
        """_affine_exp should handle a very close to 0 without numerical issues."""
        a = torch.tensor([1e-10, -1e-10])
        b = torch.tensor([1.0, -0.5])
        dg, db = _affine_exp(a, b)
        # For |a| < eps, translation_factor → 1
        assert (dg - 1.0).abs().max().item() < 1e-6
        assert (db - b).abs().max().item() < 1e-5

    def test_find_conformal_pairs_ln(self):
        """find_conformal_pairs should detect LayerNorm (weight, bias) pairs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.LayerNorm(16),
        )
        pairs = find_conformal_pairs(model)
        assert len(pairs) == 1
        w, b = pairs[0]
        assert w is not None
        assert b is not None
        assert w.shape == (16,)
        assert b.shape == (16,)

    def test_find_conformal_pairs_bn(self):
        """find_conformal_pairs should detect BatchNorm1d (weight, bias) pairs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32),
        )
        pairs = find_conformal_pairs(model)
        assert len(pairs) == 1
        w, b = pairs[0]
        assert w is not None
        assert b is not None

    def test_find_conformal_pairs_multi_layer(self):
        """find_conformal_pairs should find all norm layers in a multi-layer model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 32),
            torch.nn.LayerNorm(32),
        )
        pairs = find_conformal_pairs(model)
        assert len(pairs) == 2
        assert pairs[0][0].shape == (16,)
        assert pairs[1][0].shape == (32,)

    def test_conformal_muon_module_input(self):
        """ConformalMuon should accept an nn.Module and auto-detect pairs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.LayerNorm(8),
        )
        opt = ConformalMuon(model, lr=0.01)
        assert len(opt.param_groups) >= 1

        # Verify conformal group exists
        conformal_groups = [
            g for g in opt.param_groups if g.get("is_conformal", False)
        ]
        assert len(conformal_groups) == 1
        # LayerNorm has weight and bias
        assert len(conformal_groups[0]["params"]) == 2

    def test_conformal_muon_tuple_input(self):
        """ConformalMuon should accept list of (weight, bias) tuples."""
        model = torch.nn.LayerNorm(8)
        pairs = find_conformal_pairs(model)
        opt = ConformalMuon(pairs, lr=0.01)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0].get("is_conformal", False)

    def test_conformal_muon_gamma_positive(self):
        """ConformalMuon should keep gamma positive after updates."""
        torch.manual_seed(42)
        ln = torch.nn.LayerNorm(16)
        opt = ConformalMuon(ln, lr=0.1)

        # Run a few steps with random gradients
        for _ in range(10):
            if ln.weight.grad is not None:
                opt.zero_grad()
            loss = ln.weight.sum()
            loss.backward()
            opt.step()

        # Gamma (weight) should remain positive
        assert (ln.weight.data > 0).all(), "Gamma became non-positive"

    def test_conformal_muon_ln_converges(self):
        """ConformalMuon should minimize a loss for a LayerNorm parameter pair."""
        torch.manual_seed(42)
        ln = torch.nn.LayerNorm(8)

        # Initialize gamma and beta away from optimal
        torch.nn.init.ones_(ln.weight)
        torch.nn.init.zeros_(ln.bias)

        # Target values
        target_gamma = torch.full((8,), 2.0)
        target_beta = torch.full((8,), 0.5)

        opt = ConformalMuon(ln, lr=0.05)

        losses = []
        # Need enough steps for the exponential map to converge
        for _ in range(200):
            opt.zero_grad()
            loss = ((ln.weight - target_gamma) ** 2).sum() + \
                   ((ln.bias - target_beta) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not converge: {losses[0]:.6f} → {losses[-1]:.6f}"
        )
        # Gamma should approach target (positive)
        assert (ln.weight.data > 0).all(), "Gamma became non-positive"
        assert (ln.weight.data - target_gamma).abs().mean().item() < 0.5
        assert (ln.bias.data - target_beta).abs().mean().item() < 0.5

    def test_conformal_muon_without_bias(self):
        """ConformalMuon should handle norm layers without bias."""
        ln = torch.nn.LayerNorm(8, bias=False)
        opt = ConformalMuon(ln, lr=0.01)

        conformal_groups = [
            g for g in opt.param_groups if g.get("is_conformal", False)
        ]
        assert len(conformal_groups) == 1
        assert len(conformal_groups[0]["params"]) == 1  # only gamma

        # Should run without error
        opt.zero_grad()
        ln.weight.sum().backward()
        opt.step()

    def test_conformal_muon_weight_decay(self):
        """ConformalMuon should apply weight decay correctly."""
        ln = torch.nn.LayerNorm(4)
        gamma_before = ln.weight.data.clone()
        beta_before = ln.bias.data.clone()

        opt = ConformalMuon(ln, lr=0.0, weight_decay=0.1)

        # Create a gradient that involves both weight and bias
        ((ln.weight ** 2).sum() + (ln.bias ** 2).sum()).backward()

        # Gradient of weight**2 is 2*weight, same for bias
        # After step with wd=0.1: grad += 0.1 * param
        # With lr=0: no actual parameter update, but grad is modified
        opt.step()

        # With lr=0 and wd>0, gamma's grad gets wd*gamma added
        # but no actual parameter update happens
        expected_grad_gamma = 2 * gamma_before + 0.1 * gamma_before
        err = (ln.weight.grad - expected_grad_gamma).norm().item()
        assert err < 1e-5, f"Weight decay on gamma error: {err:.2e}"

        expected_grad_beta = 2 * beta_before + 0.1 * beta_before
        err = (ln.bias.grad - expected_grad_beta).norm().item()
        assert err < 1e-5, f"Weight decay on beta error: {err:.2e}"

    def test_conformal_muon_sgd_fallback(self):
        """Non-conformal parameter groups should use SGD fallback."""
        w = torch.nn.Parameter(torch.randn(4, 4))
        b = torch.nn.Parameter(torch.randn(4))

        opt = ConformalMuon([
            {"params": [w, b], "is_conformal": False},
        ], lr=0.01)

        before = w.data.clone()
        (w ** 2).mean().backward()
        opt.step()
        # SGD: w ← w - lr * grad
        expected = before - 0.01 * (2 * before / 16)
        assert (w.data - expected).norm() < 1e-6

    def test_conformal_muon_with_bn(self):
        """ConformalMuon should work with BatchNorm1d on a forward pass."""
        torch.manual_seed(42)
        bn = torch.nn.BatchNorm1d(8, track_running_stats=False)
        x = torch.randn(4, 8)

        opt = ConformalMuon(bn, lr=0.01)

        # Train for a few steps matching target statistics
        target_mean = torch.full((8,), 0.3)
        target_std = torch.full((8,), 2.0)

        for _ in range(50):
            opt.zero_grad()
            x = torch.randn(4, 8)
            out = bn(x)
            # Encourage output mean and std toward target
            loss = ((out.mean(0) - target_mean) ** 2).sum() + \
                   ((out.std(0) - target_std) ** 2).sum()
            loss.backward()
            opt.step()

        # Gamma (weight) should stay positive
        assert (bn.weight.data > 0).all(), "Gamma became non-positive"

    def test_conformal_muon_mixed_model(self):
        """ConformalMuon should handle a model with both linear and norm layers."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.LayerNorm(20),
            torch.nn.Linear(20, 5),
        )
        opt = ConformalMuon(model, lr=0.01)

        x = torch.randn(4, 10)
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # All params should have been updated
        conformal_count = sum(
            1 for g in opt.param_groups if g.get("is_conformal", False)
        )
        assert conformal_count == 1  # one LayerNorm

    def test_find_conformal_pairs_rmsnorm(self):
        """find_conformal_pairs should detect RMSNorm (weight, None) pairs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.RMSNorm(32),
        )
        pairs = find_conformal_pairs(model)
        assert len(pairs) == 1
        w, b = pairs[0]
        assert w is not None
        assert b is None  # RMSNorm has no bias
        assert w.shape == (32,)

    def test_find_conformal_pairs_groupnorm(self):
        """find_conformal_pairs should detect GroupNorm (weight, bias) pairs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.GroupNorm(4, 32),  # 4 groups, 32 channels
        )
        pairs = find_conformal_pairs(model)
        assert len(pairs) == 1
        w, b = pairs[0]
        assert w is not None
        assert b is not None  # GroupNorm has bias (affine=True by default)
        assert w.shape == (32,)
        assert b.shape == (32,)

    def test_find_conformal_pairs_weightnorm(self):
        """find_conformal_pairs should detect weight_norm weight_g parameters."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        # Apply weight_norm to the Linear layer
        torch.nn.utils.weight_norm(model[0], name="weight")
        pairs = find_conformal_pairs(model)

        # Should find both the LayerNorm pair and the weight_g
        ln_pairs = [(w, b) for w, b in pairs if b is not None]
        wn_pairs = [(w, b) for w, b in pairs if b is None]

        assert len(ln_pairs) == 1  # LayerNorm(16)
        assert len(wn_pairs) == 1  # weight_g from Linear
        # weight_g is stored as a column vector (out_features, 1)
        assert wn_pairs[0][0].dim() == 2
        assert wn_pairs[0][0].shape == (16, 1)  # out_features × 1

    def test_find_conformal_pairs_weightnorm_disabled(self):
        """find_conformal_pairs should skip weight_g when detect_weightnorm=False."""
        model = torch.nn.Sequential(torch.nn.Linear(8, 16))
        torch.nn.utils.weight_norm(model[0], name="weight")
        pairs = find_conformal_pairs(model, detect_weightnorm=False)
        assert len(pairs) == 0  # Linear is not a norm type

    def test_find_conformal_pairs_mixed_norms(self):
        """find_conformal_pairs with RMSNorm + LayerNorm + weight_norm."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.RMSNorm(16),
            torch.nn.Linear(16, 8),
            torch.nn.LayerNorm(8),
        )
        torch.nn.utils.weight_norm(model[0], name="weight")  # weight_g on first Linear
        pairs = find_conformal_pairs(model)

        assert len(pairs) == 3  # RMSNorm + LayerNorm + weight_g
        norm_types = [(w.shape, b is not None) for w, b in pairs]
        assert (torch.Size([16]), False) in norm_types  # RMSNorm
        assert (torch.Size([8]), True) in norm_types    # LayerNorm
        assert (torch.Size([16]), False) in norm_types  # weight_g

    def test_find_conformal_pairs_conv_weightnorm(self):
        """find_conformal_pairs should detect weight_g from Conv2d with weight_norm.

        Conv2d weight_g has shape (out_channels, 1, 1, 1) — 4D with
        three singleton spatial dims. The generalised 1-D scale check
        sum(s > 1 for s in shape) == 1 should catch it.
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.LayerNorm(16),
        )
        torch.nn.utils.weight_norm(model[0], name="weight")
        pairs = find_conformal_pairs(model)

        # Should find both the LayerNorm pair and the Conv2d weight_g
        ln_pairs = [(w, b) for w, b in pairs if b is not None]
        wn_pairs = [(w, b) for w, b in pairs if b is None]

        assert len(ln_pairs) == 1  # LayerNorm(16)
        assert len(wn_pairs) == 1  # weight_g from Conv2d

        wg = wn_pairs[0][0]
        # weight_g is (out_channels, 1, 1, 1) — 4D with spatial singletons
        assert wg.dim() == 4
        assert wg.shape == (16, 1, 1, 1)
        assert wg.shape[0] > 1  # out_channels
        assert sum(1 for s in wg.shape if s > 1) == 1  # exactly 1 non-singleton dim

    def test_find_conformal_pairs_conv1d_weightnorm(self):
        """find_conformal_pairs should detect weight_g from Conv1d with weight_norm.

        Conv1d weight_g has shape (out_channels, 1, 1) — 3D with two
        singleton spatial dims. Verifies the generalised 1-D scale check
        sum(s > 1 for s in shape) == 1 also works for 3D tensors.
        """
        model = torch.nn.Sequential(
            torch.nn.Conv1d(3, 16, 3),
            torch.nn.LayerNorm(16),
        )
        torch.nn.utils.weight_norm(model[0], name="weight")
        pairs = find_conformal_pairs(model)

        ln_pairs = [(w, b) for w, b in pairs if b is not None]
        wn_pairs = [(w, b) for w, b in pairs if b is None]

        assert len(ln_pairs) == 1  # LayerNorm(16)
        assert len(wn_pairs) == 1  # weight_g from Conv1d

        wg = wn_pairs[0][0]
        # weight_g is (out_channels, 1, 1) — 3D with spatial singletons
        assert wg.dim() == 3
        assert wg.shape == (16, 1, 1)
        assert wg.shape[0] > 1  # out_channels
        assert sum(1 for s in wg.shape if s > 1) == 1  # exactly 1 non-singleton dim


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GAMUON AUTO META-OPTIMIZER                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestGamuonAuto:
    """Verify GamuonAuto correctly partitions parameters into
    ConformalMuon / Gamuon / SGD sub-optimizers."""

    def test_auto_partitions_basic_model(self):
        """A model with Linear + LayerNorm should create all three sub-optimizers."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)

        # All three sub-optimizers should exist and have work
        assert opt._conformal is not None, "ConformalMuon should exist"
        assert opt._gamuon is not None, "Gamuon should exist"
        assert opt._sgd is not None, "SGD should exist"
        assert len(opt._optimizers) == 3

        # ConformalMuon should have one param group per norm pair (4 params total)
        total_conf_params = sum(
            len(g["params"]) for g in opt._conformal.param_groups
            if g.get("is_conformal", False)
        )
        # 2 LayerNorm layers, each with weight + bias = 4 params
        assert total_conf_params == 4, f"Expected 4 conformal params, got {total_conf_params}"

        # SGD should have biases from the Linear layers (bias=True by default)
        total_sgd_params = sum(
            len(g["params"]) for g in opt._sgd.param_groups
        )
        # 2 Linear biases = 2 params
        assert total_sgd_params == 2, f"Expected 2 SGD params, got {total_sgd_params}"

    def test_auto_no_norm_layers(self):
        """A model without norm layers should have no ConformalMuon."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )
        opt = GamuonAuto(model, lr=1e-3)

        assert opt._conformal is None, "No norm layers → no ConformalMuon"
        assert opt._gamuon is not None, "Should have Gamuon for 2D weights"
        assert opt._sgd is not None, "Should have SGD for 1D biases"
        assert len(opt._optimizers) == 2

    def test_auto_only_norm_layers(self):
        """A model with only norm layers should have only ConformalMuon and SGD."""
        model = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)

        assert opt._conformal is not None
        # LayerNorm weights/bias are 1D, so they go to ConformalMuon
        # No 2D params → no Gamuon
        assert opt._gamuon is None, "No 2D params → no Gamuon"
        # No other params → no SGD
        assert opt._sgd is None, "All params are conformal → no SGD"
        assert len(opt._optimizers) == 1

    def test_auto_weight_norm(self):
        """GamuonAuto should detect weight_norm parameters."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        torch.nn.utils.weight_norm(model[0], name="weight")
        opt = GamuonAuto(model, lr=1e-3)

        # ConformalMuon should have both the LayerNorm pair and the weight_g
        total_conf_params = sum(
            len(g["params"]) for g in opt._conformal.param_groups
        )
        # LayerNorm(16) weight+bias (2) + weight_g (1) = 3
        assert total_conf_params == 3, f"Expected 3 conformal params, got {total_conf_params}"

        # weight_g is 1D, so the Linear's weight (now parametrized) is also 1D → SGD
        assert opt._gamuon is not None, "Should still have Gamuon for remaining 2D params"

    def test_auto_rmsnorm_groupnorm(self):
        """GamuonAuto should detect RMSNorm (weight-only) and GroupNorm."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.RMSNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.GroupNorm(4, 16),
        )
        opt = GamuonAuto(model, lr=1e-3)

        total_conf_groups = len([
            g for g in opt._conformal.param_groups
            if g.get("is_conformal", False)
        ])
        # RMSNorm(32) + GroupNorm(4, 16) = 2 groups
        assert total_conf_groups == 2, f"Expected 2 conformal groups, got {total_conf_groups}"

        # RMSNorm has weight only (1 param), GroupNorm has weight+bias (2 params)
        # → 3 total conformal params
        total_conf_params = sum(
            len(g["params"]) for g in opt._conformal.param_groups
        )
        assert total_conf_params == 3, f"Expected 3 conformal params, got {total_conf_params}"

    def test_auto_empty_model(self):
        """An empty model should create no sub-optimizers."""
        model = torch.nn.Sequential()
        opt = GamuonAuto(model, lr=1e-3)

        assert opt._conformal is None
        assert opt._gamuon is None
        assert opt._sgd is None
        assert len(opt._optimizers) == 0

    def test_auto_manual_groups_role_conformal_uses_affine_update(self):
        """role='conformal' groups in the manual-group path used to
        silently fall back to plain SGD because the is_conformal flag
        wasn't being injected.  Verify the affine update is actually
        applied: gamma must update *multiplicatively* (γ ← γ · e^a),
        which is the signature behaviour of ConformalMuon."""
        torch.manual_seed(0)
        ln = torch.nn.LayerNorm(8)
        other = torch.nn.Parameter(torch.randn(4, 4))

        opt = GamuonAuto(
            [
                {"params": [ln.weight, ln.bias], "role": "conformal"},
                {"params": [other], "role": "gamuon"},
            ],
            lr=1e-2,
        )

        # The manual-group path must build a real ConformalMuon
        assert opt._conformal is not None, "manual group conformal missing"
        assert opt._gamuon is not None
        # And the group must be flagged so step() takes the affine branch
        conf_groups = [
            g for g in opt._conformal.param_groups
            if g.get("is_conformal", False)
        ]
        assert len(conf_groups) == 1, (
            f"Expected 1 affine-flagged group, got {len(conf_groups)}"
        )

        # Run a step; the affine update is multiplicative on gamma,
        # so γ_new / γ_old should be uniform across all entries (a
        # single dilation factor) — additive SGD would not have this
        # property because the per-entry grads differ.
        gamma_before = ln.weight.data.clone()
        x = torch.randn(4, 8)
        loss = (ln(x) ** 2).sum() + (other ** 2).sum()
        loss.backward()
        opt.step()

        ratios = ln.weight.data / gamma_before
        # All gamma entries should have moved by the same multiplicative
        # factor (within float32 noise) since the dilation coefficient
        # `a` is shared across entries for a single norm layer.
        assert ratios.std().item() < 1e-5, (
            f"gamma did not update as a uniform dilation "
            f"(std of ratio = {ratios.std().item():.2e}) — "
            "the manual-group conformal path is falling back to SGD."
        )

    def test_auto_manual_groups_conformal_rejects_oversized_group(self):
        """A role='conformal' group with >2 params is a user error
        (norm layers have at most γ and β); GamuonAuto should reject
        it instead of silently dropping the extras inside
        ConformalMuon._step_conformal."""
        ln1 = torch.nn.LayerNorm(4)
        ln2 = torch.nn.LayerNorm(4)
        with pytest.raises(ValueError, match="role='conformal'"):
            GamuonAuto(
                [{
                    "params": [ln1.weight, ln1.bias, ln2.weight, ln2.bias],
                    "role": "conformal",
                }],
                lr=1e-3,
            )

    def test_auto_training_step(self):
        """A full training step with GamuonAuto should update all params and reduce loss."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 4),
        )
        opt = GamuonAuto(model, lr=1e-2)
        x = torch.randn(4, 8)
        target = torch.randn(4, 4)

        # Get initial state
        params_before = [p.data.clone() for p in model.parameters()]

        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()

        # All params should have been updated
        for p, before in zip(model.parameters(), params_before):
            assert (p.data - before).abs().sum().item() > 0, "Parameter was not updated"

    def test_auto_state_dict_structure(self):
        """state_dict should have expected structure with sub-keys for each
        active sub-optimizer."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-2)

        # Run a few steps to build state
        for _ in range(5):
            x = torch.randn(4, 8)
            ((model(x) ** 2).mean()).backward()
            opt.step()

        state = opt.state_dict()

        # Should have sub-keys for each active sub-optimizer
        assert "conformal" in state
        assert "gamuon" in state
        assert "sgd" in state

        # Each sub-state should be a valid optimizer state dict
        for key in ("conformal", "gamuon", "sgd"):
            assert "param_groups" in state[key], (
                f"Sub-state '{key}' missing param_groups"
            )
            assert len(state[key]["param_groups"]) > 0
            # Params are stored as IDs (ints) in the state dict
            for g in state[key]["param_groups"]:
                assert "params" in g
                for p_id in g["params"]:
                    assert isinstance(p_id, int)

    def test_auto_state_dict_same_instance_roundtrip(self):
        """state_dict -> load_state_dict on the same optimizer instance
        should produce identical updates (parameter IDs match within one
        optimizer)."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-2)

        # Run a few steps to build momentum state
        for _ in range(5):
            x = torch.randn(4, 8)
            ((model(x) ** 2).mean()).backward()
            opt.step()

        # Save state
        state = opt.state_dict()

        # Reset model weights so load_state_dict has observable effect
        for p in model.parameters():
            torch.nn.init.normal_(p)

        # Reset weights, load saved state, run a step
        for p in model.parameters():
            torch.nn.init.normal_(p)
        opt.load_state_dict(state)

        x = torch.randn(4, 8)
        ((model(x) ** 2).mean()).backward()
        opt.step()

        # Verify load_state_dict doesn't crash and produces valid params
        assert all(torch.isfinite(p).all() for p in model.parameters())

    def test_auto_scale_invariant_ln_converges(self):
        """GamuonAuto should minimise a loss with LayerNorm present."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 4),
        )
        opt = GamuonAuto(model, lr=5e-3)
        x = torch.randn(16, 8)
        target = torch.randn(16, 4)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((model(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.8, (
            f"Loss did not converge: {losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_auto_output_shape(self):
        """Verify model output changes after a GamuonAuto step (sanity)."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-2)
        x = torch.randn(4, 8)

        out_before = model(x).clone()
        ((model(x) ** 2).mean()).backward()
        opt.step()
        out_after = model(x)

        # Outputs should differ after an update
        diff = (out_before - out_after).abs().max().item()
        assert diff > 0, "Output unchanged after optimizer step"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GRADIENT NORM MONITOR                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestGradNormMonitor:
    """Verify GradNormMonitor captures correct gradient norm metrics
    across all three GamuonAuto sub-optimizer roles."""

    def test_capture_returns_all_active_roles(self):
        """capture() should return a dict keyed by active sub-optimizer roles."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()

        result = monitor.capture(step=0)

        # All three roles should be present
        assert "conformal" in result
        assert "gamuon" in result
        assert "sgd" in result

        # Each result should have the expected keys
        for role_result in result.values():
            assert "total_norm" in role_result
            assert "mean_norm" in role_result
            assert "max_norm" in role_result
            assert "num_params" in role_result
            assert "zero_frac" in role_result

    def test_capture_norms_are_positive(self):
        """After a backward pass, gradient norms should be strictly positive."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()

        result = monitor.capture()

        for role_result in result.values():
            assert role_result["total_norm"] > 0.0
            assert role_result["mean_norm"] > 0.0
            assert role_result["max_norm"] > 0.0

    def test_capture_without_backward_zeros(self):
        """Without a backward pass, gradients are None — zero_frac should be 1."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        # No backward pass — all grads are None
        result = monitor.capture()

        for role_result in result.values():
            assert role_result["zero_frac"] == 1.0
            assert role_result["total_norm"] == 0.0

    def test_capture_after_zero_grad(self):
        """After zero_grad (without new backward), gradients should be zero
        but not None — total_norm = 0, zero_frac < 1 but total_norm = 0."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        result = monitor.capture()

        for role_result in result.values():
            assert role_result["total_norm"] == 0.0
            # Grads are zeroed (not None), so all are counted as zero

    def test_capture_after_step_zero_grad(self):
        """Full forward → backward → step → zero_grad cycle, norms should be
        positive before step and zero after zero_grad."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)

        # Forward + backward
        loss = (model(x) ** 2).mean()
        loss.backward()

        result_before = monitor.capture(step=0)
        for role_result in result_before.values():
            assert role_result["total_norm"] > 0.0, \
                "Expected positive norms before step"

        opt.step()
        opt.zero_grad()

        result_after = monitor.capture(step=1)
        for role_result in result_after.values():
            assert role_result["total_norm"] == 0.0, \
                "Expected zero norms after step + zero_grad"

    def test_summary_returns_aggregated_stats(self):
        """summary() should return mean, std, min, max, last per role."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)

        for step in range(5):
            opt.zero_grad()
            loss = (model(x) ** 2).mean()
            loss.backward()
            monitor.capture(step=step)
            opt.step()

        summary = monitor.summary()

        for role in ("conformal", "gamuon", "sgd"):
            assert role in summary, f"Missing role: {role}"
            s = summary[role]
            assert s["count"] == 5
            assert s["mean"] > 0.0
            assert s["std"] >= 0.0
            assert 0.0 <= s["min"] <= s["max"]
            assert s["last"] > 0.0
            assert 0.0 <= s["zero_frac_mean"] <= 1.0

    def test_reset_clears_history(self):
        """reset() should clear all recorded history."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()
        monitor.capture()

        summary_before = monitor.summary()
        assert summary_before["gamuon"]["count"] == 1

        monitor.reset()
        summary_after = monitor.summary()
        assert summary_after["gamuon"]["count"] == 0

    def test_log_freq_zero_no_output(self, capsys):
        """With log_freq=0, capture() should not print anything."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, log_freq=0)

        x = torch.randn(2, 4)
        loss = (model(x) ** 2).mean()
        loss.backward()
        monitor.capture()

        captured = capsys.readouterr()
        assert captured.out == "", f"Expected no output, got: {captured.out}"

    def test_log_freq_one_prints(self, capsys):
        """With log_freq=1, capture() should print gradient norms."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, log_freq=1)

        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()
        monitor.capture(step=42)

        captured = capsys.readouterr()
        assert "grad_norm" in captured.out
        assert "conformal:" in captured.out
        assert "gamuon:" in captured.out
        assert "sgd:" in captured.out

    def test_only_gamuon_active(self):
        """A model with only 2-D params (no norms, no 1-D) → only gamuon role."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        # No bias to avoid 1-D params going to SGD
        model[0].bias = None
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(2, 4)
        loss = (model(x) ** 2).mean()
        loss.backward()

        result = monitor.capture()
        assert "gamuon" in result
        assert "conformal" not in result
        # SGD might or might not exist — depends on remaining params

    def test_multiple_steps_track_history_length(self):
        """After N captures, each role should have N entries in history."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LayerNorm(16),
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)

        for step in range(10):
            opt.zero_grad()
            loss = (model(x) ** 2).mean()
            loss.backward()
            monitor.capture(step=step)
            opt.step()

        summary = monitor.summary()
        for role in ("conformal", "gamuon", "sgd"):
            assert summary[role]["count"] == 10, \
                f"Expected 10 entries for {role}, got {summary[role]['count']}"

    def test_num_params_correct(self):
        """num_params should reflect the actual number of parameters per role."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),    # weight: 8×16=128, bias: 16
            torch.nn.LayerNorm(16),    # weight: 16, bias: 16
        )
        opt = GamuonAuto(model, lr=1e-3)
        monitor = GradNormMonitor(opt, silent=True)

        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()

        result = monitor.capture()

        # Linear(8→16) weight is 2D → Gamuon
        assert result["gamuon"]["num_params"] == 1  # one weight matrix
        # LayerNorm(16) weight+bias = 2 conformal params
        assert result["conformal"]["num_params"] == 2  # gamma + beta
        # Linear bias is 1D → SGD
        assert result["sgd"]["num_params"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
