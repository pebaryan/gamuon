"""
Gamuon: A Geometric (Clifford) Algebra-native Optimizer
=======================================================

Gamuon reformulates the Muon optimizer in the language of geometric
(Clifford) algebra.  Instead of approximating the orthogonal projection
via Newton–Schulz iterations, Gamuon:

  1. Decomposes the gradient into geometric grades  (scalar, bivector, strain).
  2. Computes exact rotor updates  via bivector exponentials.
  3. Applies updates through  versor sandwich products  that preserve
     the geometric structure of the weight space.

Theoretical foundation
---------------------
For a weight matrix  W ∈ ℝ^{n×n},  the gradient  G  is a mixed-grade
multivector in the Clifford algebra  Cl(n,0):

    G = ⟨G⟩₀ + ⟨G⟩₂ + ⟨G⟩₊

where:
  •  ⟨G⟩₀ = (tr(G)/n)·I           -- scalar grade (isotropic dilation)
  •  ⟨G⟩₂ = (G - Gᵀ)/2            -- bivector grade (infinitesimal rotation)
  •  ⟨G⟩₊ = (G + Gᵀ)/2 - ⟨G⟩₀     -- symmetric traceless (pure strain)

The bivector  B = ⟨G⟩₂  generates a rotor  R = exp(η·B) ∈ Spin(n)
via the matrix exponential, giving an exact orthogonal update:

    W ← R · W · Rᵀ  -  η₊ · ⟨G⟩₊  -  η₀ · ⟨G⟩₀

This preserves the singular-value spectrum up to the scalar and strain
corrections — a key difference from standard Muon which applies an
additive orthogonalisation.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch



# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CLIFFORD ALGEBRA PRIMITIVES  (Cl(n, 0)  –  Euclidean signature)   ║
# ╚══════════════════════════════════════════════════════════════════════╝


def grade_decompose(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a square matrix into its geometric grades.

    Parameters
    ----------
    M : (n, n) tensor
        A square matrix representing a mixed-grade multivector in Cl(n,0).

    Returns
    -------
    scalar : (n, n) tensor
        ⟨M⟩₀ = tr(M)/n · I    — scalar (grade‑0) part.
    bivector : (n, n) tensor
        ⟨M⟩₂ = (M − Mᵀ)/2     — bivector (grade‑2) part.
    strain : (n, n) tensor
        ⟨M⟩₊ = (M + Mᵀ)/2 − ⟨M⟩₀   — symmetric traceless (mixed even grade).
    """
    n = M.shape[-1]
    # --- scalar grade ---------------------------------------------------
    trace = M.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / n
    scalar = trace.unsqueeze(-1) * torch.eye(n, dtype=M.dtype, device=M.device)
    # --- bivector grade -------------------------------------------------
    bivector = (M - M.transpose(-2, -1)) / 2
    # --- symmetric traceless (strain) -----------------------------------
    symmetric = (M + M.transpose(-2, -1)) / 2
    strain = symmetric - scalar
    return scalar, bivector, strain


def bivector_norm(B: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of a bivector (antisymmetric matrix).

    For a simple bivector  B  this equals  |θ|  where  θ/2  is the
    rotation angle of the generated rotor.
    """
    return torch.sqrt((B ** 2).sum(dim=(-2, -1), keepdim=True).clamp(min=1e-30))


def bivector_exp(B: torch.Tensor) -> torch.Tensor:
    """Exponential of a bivector  →  rotor in Spin(n).

    For a bivector  B  (antisymmetric matrix),  exp(B)  is an orthogonal
    matrix.  This is the standard matrix exponential restricted to the
    Lie algebra  so(n).

    For  n = 2  and  n = 3  we use closed‑form expressions (Rodrigues);
    for larger  n  we fall back to  torch.matrix_exp.
    """
    n = B.shape[-1]
    if n == 2:
        return _bivector_exp_2d(B)
    elif n == 3:
        return _bivector_exp_3d(B)
    else:
        return torch.matrix_exp(B)


def _bivector_exp_2d(B: torch.Tensor) -> torch.Tensor:
    """Closed‑form rotor exponential in Cl(2,0).

    A 2×2 antisymmetric matrix has the form
        B = [[0,  θ], [-θ, 0]] .

    exp(B) = [[cos θ,  sin θ], [-sin θ,  cos θ]]   ∈ SO(2).
    """
    θ = B[..., 0, 1]                           # scalar angle
    cos = θ.cos().unsqueeze(-1).unsqueeze(-1)
    sin = θ.sin().unsqueeze(-1).unsqueeze(-1)
    top = torch.cat([cos, sin], dim=-1)
    bot = torch.cat([-sin, cos], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _bivector_exp_3d(B: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula for SO(3) from a bivector in Cl(3,0).

    For  B ∈ so(3), let  ω = (B[2,1], B[0,2], B[1,0])  be the
    rotation vector,  θ = ‖ω‖.  Then

        exp(B) = I + sin θ·(B/θ) + (1−cos θ)·(B/θ)² .
    """
    ω_x = B[..., 2, 1]
    ω_y = B[..., 0, 2]
    ω_z = B[..., 1, 0]
    θ = torch.sqrt(ω_x ** 2 + ω_y ** 2 + ω_z ** 2).clamp(min=1e-30)

    # Normalised cross‑product matrix  K = B / θ
    K = B / θ.unsqueeze(-1).unsqueeze(-1)

    # K²  (symmetric)
    K2 = K @ K

    sin_θ = θ.sin().unsqueeze(-1).unsqueeze(-1)
    cos_θ = θ.cos().unsqueeze(-1).unsqueeze(-1)

    eye = torch.eye(3, dtype=B.dtype, device=B.device)
    return eye + sin_θ * K + (1 - cos_θ) * K2


def rotor_apply(R: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply a rotor  R ∈ Spin(n)  to a matrix  W  via sandwich product.

        W′ = R · W · Rᵀ

    This is the versor action of the Clifford algebra:  R  acts on the
    multivector  W  by conjugation, preserving the grade structure.
    """
    return R @ W @ R.transpose(-2, -1)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GRADIENT ACCUMULATION  (multivector momentum)                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MultivectorMomentum:
    """Grade‑aware momentum buffer for the Gamuon optimizer.

    Maintains separate exponential moving averages for each geometric
    grade of the gradient, allowing per‑grade learning rates and
    decay factors.
    """

    __slots__ = ("scalar_m", "bivector_m", "strain_m",
                 "scalar_v", "bivector_v", "strain_v")

    def __init__(self, n: int, device: torch.device, dtype: torch.dtype):
        zero = lambda: torch.zeros(n, n, device=device, dtype=dtype)
        self.scalar_m = zero()
        self.bivector_m = zero()
        self.strain_m = zero()
        self.scalar_v = zero()
        self.bivector_v = zero()
        self.strain_v = zero()

    @torch.no_grad()
    def step(
        self,
        scalar: torch.Tensor,
        bivector: torch.Tensor,
        strain: torch.Tensor,
        betas: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update momentum buffers and return bias‑corrected moments.

        Parameters
        ----------
        betas : (β₁, β₂)
            β₁  – momentum decay  (first moment)
            β₂  – velocity decay   (second moment / RMS)
        """
        b1, b2 = betas

        # --- first moments (mean) ---------------------------------------
        self.scalar_m.mul_(b1).add_(scalar, alpha=1 - b1)
        self.bivector_m.mul_(b1).add_(bivector, alpha=1 - b1)
        self.strain_m.mul_(b1).add_(strain, alpha=1 - b1)

        # --- second moments (uncentred variance / RMS) ------------------
        self.scalar_v.mul_(b2).add_(scalar ** 2, alpha=1 - b2)
        self.bivector_v.mul_(b2).add_(bivector ** 2, alpha=1 - b2)
        self.strain_v.mul_(b2).add_(strain ** 2, alpha=1 - b2)

        return (self.scalar_m, self.bivector_m, self.strain_m,
                self.scalar_v, self.bivector_v, self.strain_v)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GAMUON OPTIMISER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class Gamuon(torch.optim.Optimizer):
    """Gamuon — Geometric (Clifford) Algebra-native Optimiser.

    Extends the Muon paradigm by formulating updates in the Clifford
    algebra  Cl(n,0).  Gradient is decomposed by geometric grade;
    the bivector part generates an exact rotor (via matrix exponential)
    that acts on weights through a versor sandwich, while scalar and
    strain parts drive dilation and shape changes respectively.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default 1e-3
        Base learning rate.  Scaled per‑grade by ``lr_scalar``,
        ``lr_bivector``, ``lr_strain``.
    betas : (float, float), default (0.9, 0.999)
        Coefficients for first‑ and second‑moment estimates
        (multivector momentum).
    eps : float, default 1e-8
        Term added to denominator for numerical stability.
    weight_decay : float, default 0.0
        L2 weight decay (applied as isotropic dilation on the scalar grade).
    lr_scalar : float, default 1.0
        Multiplier for the learning rate on the scalar (dilation) grade.
    lr_bivector : float, default 1.0
        Multiplier for the learning rate on the bivector (rotor) grade.
    lr_strain : float, default 1.0
        Multiplier for the learning rate on the strain grade.
    foreach : bool, default True
        Whether to fuse parameter updates for efficiency.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_scalar: float = 1.0,
        lr_bivector: float = 1.0,
        lr_strain: float = 1.0,
        foreach: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            lr_scalar=lr_scalar, lr_bivector=lr_bivector, lr_strain=lr_strain,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimisation step.

        Decomposes the gradient into scalar, bivector, and strain grades,
        updates multivector momentum, generates an exact rotor from the
        bivector via matrix exponential, and applies the versor sandwich
        update  W \u2192 R\u00b7W\u00b7R\u1d40 \u2212 strain\u2212 scalar dilation.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        float or None
            The loss from ``closure``, or ``None`` if no closure was given.

        Example
        -------
        >>> W = nn.Parameter(torch.randn(32, 32))
        >>> opt = Gamuon([W], lr=1e-3)
        >>>
        >>> for step in range(100):
        ...     opt.zero_grad()
        ...     loss = (W ** 2).mean()
        ...     loss.backward()
        ...     opt.step()
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_list = group["params"]

            # -----------------------------------------------------------
            #  Group hyper‑parameters
            # -----------------------------------------------------------
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr_s = group["lr_scalar"]
            lr_b = group["lr_bivector"]
            lr_p = group["lr_strain"]

            # -----------------------------------------------------------
            #  Per‑parameter step
            # -----------------------------------------------------------
            for p in p_list:
                if p.grad is None:
                    continue
                g = p.grad.data
                if g.is_sparse:
                    raise RuntimeError("Gamuon does not support sparse gradients")

                # -- ensure square by padding if necessary --
                orig_shape = g.shape
                orig_p_data = p.data
                needs_unpad = False
                if g.dim() == 2 and g.shape[0] != g.shape[1]:
                    m, n = g.shape
                    max_dim = max(m, n)
                    padded_g = g.new_zeros(max_dim, max_dim)
                    padded_g[:m, :n] = g
                    g = padded_g
                    padded_p = p.data.new_zeros(max_dim, max_dim)
                    padded_p[:m, :n] = p.data
                    p.data = padded_p
                    needs_unpad = True
                    orig_shape_for_unpad = (m, n)
                elif g.dim() != 2:
                    raise NotImplementedError(
                        "Gamuon currently supports only 2‑D parameters. "
                        f"Got shape {orig_shape}"
                    )

                n = g.shape[-1]
                device, dtype = g.device, g.dtype

                # -- initialise state (always keyed by the original parameter) --
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = MultivectorMomentum(n, device, dtype)

                state["step"] += 1
                step_t = torch.tensor(state["step"], dtype=dtype, device=device)

                # -- weight decay (applied as scalar dilation) --
                if wd != 0:
                    g.add_(p.data, alpha=wd)

                # -- grade decomposition --
                scalar, bivector, strain = grade_decompose(g)

                # -- multivector momentum --
                mom = state["momentum"]
                (m_s, m_b, m_p, v_s, v_b, v_p) = mom.step(
                    scalar, bivector, strain, (beta1, beta2)
                )

                # -- bias correction --
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t

                # -- compute updates per grade --

                # • Scalar grade:  isotropic dilation
                #   Use the mean of the bias-corrected scalar momentum diagonal
                m_s_bc = m_s.diagonal().mean() / bc1
                v_s_bc = (v_s.diagonal().mean() / bc2).sqrt()
                scalar_step = (lr * lr_s) * m_s_bc / (v_s_bc + eps)

                # • Bivector grade:  generate rotor  R = exp(η · B̂)
                m_b_bc = m_b / bc1.unsqueeze(-1).unsqueeze(-1)
                v_b_bc = v_b / bc2.unsqueeze(-1).unsqueeze(-1)
                bivector_step = (lr * lr_b) * m_b_bc / (v_b_bc.sqrt() + eps)
                R = bivector_exp(bivector_step)

                # • Strain grade:  symmetric deformation
                m_p_bc = m_p / bc1.unsqueeze(-1).unsqueeze(-1)
                v_p_bc = v_p / bc2.unsqueeze(-1).unsqueeze(-1)
                update_strain = (lr * lr_p) * m_p_bc / (v_p_bc.sqrt() + eps)

                # -- apply update via versor sandwich --
                updated = (rotor_apply(R, p.data)
                           - scalar_step * torch.eye(n, device=device, dtype=dtype)
                           - update_strain)

                # -- unpad if necessary and write back --
                if needs_unpad:
                    m, n = orig_shape_for_unpad
                    orig_p_data.copy_(updated[:m, :n])
                    # Restore original parameter reference
                    p.data = orig_p_data
                else:
                    p.data.copy_(updated)

        return loss


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  NEWTON–SCHULZ FALLBACK  (for comparison / ablation)               ║
# ╚══════════════════════════════════════════════════════════════════════╗


def newton_schulz(
    G: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """Approximate  sign(G) = U Vᵀ  via Newton–Schulz iterations.

    This is the method used by the standard Muon optimizer.  Provided
    here for ablation studies and comparison with the exact bivector
    exponential.

    The iteration  Xₖ₊₁ = (3Xₖ − Xₖ·Xₖᵀ·Xₖ) / 2  converges to the
    nearest orthogonal matrix to  G  in the Frobenius norm.
    """
    X = G.clone()
    for _ in range(num_iters):
        X = (3 * X - X @ X.transpose(-2, -1) @ X) / 2
    return X


class GamuonNS(torch.optim.Optimizer):
    """Hybrid Gamuon using Newton–Schulz (for ablation / comparison).

    Uses the same grade decomposition and momentum as Gamuon, but
    replaces the bivector exponential with Newton–Schulz iterations
    (as in standard Muon).  This makes it a useful baseline for
    isolating the effect of the exact rotor update.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_iters: int = 5,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            ns_iters=ns_iters,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                # Project gradient onto orthogonal group via NS
                sign_g = newton_schulz(g, num_iters=group["ns_iters"])

                # Standard Muon-style update with grade decomposition
                _, bivector, strain = grade_decompose(g)
                scalar_val = g.diagonal().mean().item()
                n = g.shape[-1]
                device, dtype = g.device, g.dtype

                # Apply update:  orthogonal part + strain + scalar
                update = (group["lr"]) * (
                    sign_g
                    - strain
                    - scalar_val * torch.eye(n, device=device, dtype=dtype)
                )
                p.data.add_(-update)

        return loss


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFORMAL MUON  (CGA Cl(4,1)  for normalisation layers)           ║
# ╚══════════════════════════════════════════════════════════════════════╝


def find_conformal_pairs(
    model: torch.nn.Module,
    types: Optional[Tuple[type, ...]] = None,
    detect_weightnorm: bool = True,
) -> list:
    """Find  (γ, β)  parameter pairs from normalisation layers.

    Scans all submodules of *model* and returns a list of
    ``(weight_param, bias_param)`` tuples for every module matching
    one of the specified *types*.  Each tuple is suitable for passing
    directly to :class:`ConformalMuon`.

    Also detects **weight‑normalized** modules (``torch.nn.utils.weight_norm``)
    by scanning for 1‑D ``weight_g`` parameters, which are paired with
    ``None`` (no bias counterpart).

    Parameters
    ----------
    model : nn.Module
        The model to scan.
    types : tuple of types, optional
        Module types to detect.  Defaults to
        ``(nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.GroupNorm, nn.RMSNorm)``.
    detect_weightnorm : bool, default True
        Whether to also scan for 1‑D ``weight_g`` parameters created by
        :func:`torch.nn.utils.weight_norm`.

    Returns
    -------
    list of (Parameter, Parameter | None)
        List of  (weight, bias)  pairs found.  Bias is ``None`` for
        modules without a bias (RMSNorm, weight_norm, LayerNorm(bias=False)).
    """
    if types is None:
        types = (
            torch.nn.LayerNorm,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.GroupNorm,
            torch.nn.RMSNorm,
        )
    pairs = []
    for module in model.modules():
        if isinstance(module, types):
            w = getattr(module, "weight", None)
            b = getattr(module, "bias", None)
            if w is not None:
                pairs.append((w, b) if b is not None else (w, None))
        # Detect weight_norm parametrizations
        if detect_weightnorm:
            weight_g = getattr(module, "weight_g", None)
            if weight_g is not None and isinstance(weight_g, torch.nn.Parameter):
                # weight_g is an "effectively 1D" scale parameter; handles
                # 1-D (out_features,), column (out_features, 1), and
                # Conv shapes like (out_channels, 1, 1, 1)
                is_1d_scale = sum(1 for s in weight_g.shape if s > 1) == 1
                if is_1d_scale:
                    # Avoid double-counting if weight_g happens to be on a norm module
                    if not isinstance(module, types if types is not None else ()):
                        pairs.append((weight_g, None))
    return pairs


def _affine_exp(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the exponential map of the 1D affine group to a  (γ, β)  pair.

    The affine group  Aff(1)  has Lie algebra spanned by:
        G_d = [[1, 0], [0, 0]]   (dilation generator)
        G_t = [[0, 1], [0, 0]]   (translation generator)

    For  (a, b)  in the Lie algebra, the exponential map gives:
        exp(a·G_d + b·G_t) = [[e^a,  b·(e^a - 1)/a], [0, 1]]

    Applied to  (γ, β):
        γ' = γ · e^a
        β' = e^a · β + b · (e^a - 1) / a

    Parameters
    ----------
    a : tensor
        Dilation coefficient  (same shape as gamma).
    b : tensor
        Translation coefficient  (same shape as beta).
    eps : float
        Threshold below which  a  is treated as zero
        (pure translation limit).

    Returns
    -------
    delta_gamma : tensor
        Multiplicative factor for gamma  (exp_a).
    delta_beta : tensor
        Additive update for beta.
    """
    exp_a = torch.exp(a)
    # Where |a| is tiny, use the limit:  (e^a - 1)/a → 1
    mask = a.abs() > eps
    translation_factor = torch.where(
        mask,
        (exp_a - 1) / a,
        torch.ones_like(a),
    )
    delta_gamma = exp_a
    delta_beta = b * translation_factor
    return delta_gamma, delta_beta


class ConformalMuon(torch.optim.Optimizer):
    """Conformal Muon — geometric optimisation for normalisation layers.

    Treats  (γ, β)  parameter pairs of LayerNorm / BatchNorm as elements
    of the affine group  Aff(1)  (dilation + translation), which is the
    1‑D restriction of the conformal group  Spin(4,1)  in CGA Cl(4,1).

    Instead of updating  γ  and  β  independently (as Adam does), the
    update respects the semidirect-product structure
       ℝ  ⋊  ℝ⁺
    of the conformal group: dilations are multiplicative, translations
    are additive, and they do **not** commute.

    Parameters
    ----------
    params : iterable or nn.Module
        - If an **nn.Module**, auto‑detects all  (γ, β)  pairs via
          :func:`find_conformal_pairs`  (covers ``LayerNorm``,
          ``BatchNorm*\b``, ``GroupNorm``, ``RMSNorm``, and
          weight‑normalized ``weight_g`` parameters) and treats remaining parameters
          with SGD fallback.
        - If an **iterable of dicts** (standard PyTorch param groups),
          groups can optionally include ``"is_conformal": True`` to
          apply the affine update.  Groups without this flag get plain SGD.
    lr : float, default 1e-3
        Learning rate.
    betas : (float, float), default (0.9, 0.999)
        Coefficients for first‑ and second‑moment estimates on the
        Lie algebra  (dilation and translation channels).
    eps : float, default 1e-8
        Numerical stability term.
    weight_decay : float, default 0.0
        Weight decay applied to all parameters.

    Example
    -------
    >>> model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
    >>> opt = ConformalMuon(model, lr=1e-3)
    >>> # Or with explicit pairs:
    >>> pairs = find_conformal_pairs(model)
    >>> opt = ConformalMuon(pairs, lr=1e-3)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # ── Handle Module auto‑detection ────────────────────────────
        if isinstance(params, torch.nn.Module):
            model = params
            pairs = find_conformal_pairs(model)
            other = []
            pair_params = set()
            for w, b in pairs:
                pair_params.add(id(w))
                if b is not None:
                    pair_params.add(id(b))
            for p in model.parameters():
                if id(p) not in pair_params:
                    other.append(p)

            param_groups = []
            for w, b in pairs:
                group_params = [w]
                if b is not None:
                    group_params.append(b)
                param_groups.append({"params": group_params, "is_conformal": True})
            if other:
                param_groups.append({"params": other, "is_conformal": False})
            super().__init__(param_groups, defaults)

        elif isinstance(params, list) and all(
            isinstance(g, tuple) and len(g) == 2 for g in params
        ):
            # List of (weight, bias) tuples
            param_groups = []
            for w, b in params:
                group_params = [w]
                if b is not None:
                    group_params.append(b)
                param_groups.append({"params": group_params, "is_conformal": True})
            super().__init__(param_groups, defaults)

        else:
            # Standard param groups (user must add "is_conformal" key)
            super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimisation step.

        For conformal parameter groups (pairs of 1‑D tensors), the
        update is the affine‑group exponential map.  Other groups
        receive plain SGD.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        float or None
            The loss from ``closure``, or ``None`` if no closure was given.

        Example
        -------
        >>> model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        >>> opt = ConformalMuon(model, lr=1e-3)
        >>>
        >>> for x, y in dataloader:
        ...     opt.zero_grad()
        ...     loss = model(x).sum()
        ...     loss.backward()
        ...     opt.step()
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            is_conformal = group.get("is_conformal", False)

            if is_conformal:
                # ── Conformal (affine) group update ──────────────────
                p_list = group["params"]
                gamma = p_list[0]
                beta = p_list[1] if len(p_list) > 1 else None

                self._step_conformal(gamma, beta, group)
            else:
                # ── SGD fallback ─────────────────────────────────────
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    if wd != 0:
                        g.add_(p.data, alpha=wd)
                    p.data.add_(g, alpha=-lr)

        return loss

    def _step_conformal(
        self,
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor],
        group: dict,
    ) -> None:
        """Apply the affine‑group (conformal) update to a  (γ, β)  pair."""
        if gamma.grad is None:
            return

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        g_gamma = gamma.grad.data
        g_beta = beta.grad.data if beta is not None and beta.grad is not None else None

        # Weight decay
        if wd != 0:
            g_gamma.add_(gamma.data, alpha=wd)
            if g_beta is not None:
                g_beta.add_(beta.data, alpha=wd)

        # ── Lie algebra gradients ───────────────────────────────────
        #   g_α = γ · g_γ    (dilation generator coefficient)
        #   g_β = g_β         (translation generator coefficient)
        g_alpha = gamma.data * g_gamma  # shape: (d,) or scalar

        # ── State ────────────────────────────────────────────────────
        state = self.state[gamma]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg_a"] = torch.zeros_like(gamma.data)
            state["exp_avg_sq_a"] = torch.zeros_like(gamma.data)
            if g_beta is not None:
                state["exp_avg_b"] = torch.zeros_like(beta.data)
                state["exp_avg_sq_b"] = torch.zeros_like(beta.data)

        state["step"] += 1
        step_t = torch.tensor(state["step"], dtype=gamma.dtype, device=gamma.device)

        # ── Momentum on the Lie algebra ─────────────────────────────
        state["exp_avg_a"].mul_(beta1).add_(g_alpha, alpha=1 - beta1)
        state["exp_avg_sq_a"].mul_(beta2).add_(g_alpha ** 2, alpha=1 - beta2)

        if g_beta is not None:
            state["exp_avg_b"].mul_(beta1).add_(g_beta, alpha=1 - beta1)
            state["exp_avg_sq_b"].mul_(beta2).add_(g_beta ** 2, alpha=1 - beta2)

        # ── Bias correction ─────────────────────────────────────────
        bc1 = 1 - beta1 ** step_t
        bc2 = 1 - beta2 ** step_t

        m_a = state["exp_avg_a"] / bc1
        v_a = state["exp_avg_sq_a"] / bc2

        # ── Lie algebra step ────────────────────────────────────────
        a = -lr * m_a / (v_a.sqrt() + eps)

        # ── Affine exponential update ───────────────────────────────
        delta_gamma, delta_beta = _affine_exp(a, torch.zeros_like(a), eps)

        gamma.data.mul_(delta_gamma)

        if g_beta is not None:
            m_b = state["exp_avg_b"] / bc1
            v_b = state["exp_avg_sq_b"] / bc2
            b = -lr * m_b / (v_b.sqrt() + eps)

            # Complete affine update:  β' = e^a · β + b · (e^a - 1) / a
            _, trans = _affine_exp(a, b, eps)
            beta.data.mul_(delta_gamma)  # e^a · β
            beta.data.add_(trans)  # + b · (e^a - 1) / a


# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  AUTO META-OPTIMIZER                                         ║
# ║  Gamuon + ConformalMuon + SGD in one shot                     ║
# ╚════════════════════════════════════════════════════════════════════════════════╝


class GamuonAuto:
    """One-stop meta-optimizer: auto-detects norm pairs, 2D weights, and the rest.

    ``GamuonAuto`` is the simplest way to use Gamuon in practice.  Pass it any
    ``nn.Module`` and it automatically partitions parameters:

    * **Norm-layer**  (\u03b3, \u03b2)  pairs  (LayerNorm, BatchNorm, RMSNorm,
      GroupNorm, weight_norm)  \u2192  :class:`ConformalMuon`  (affine-group update)
    * **2\u2011D matrix** weights  \u2192  :class:`Gamuon`  (grade decomposition + rotor)
    * **1\u2011D / other** params  \u2192  plain SGD

    All three sub-optimisers share the same learning rate and momentum
    hyper-parameters, so you can use this class exactly as you would
    ``torch.optim.Adam``.

    Parameters
    ----------
    params : nn.Module or iterable
        - If an **nn.Module**, all parameters are auto-discovered and
          partitioned as described above.
        - If an **iterable** (standard param groups), each group must
          include an ``"role"`` key with one of:
          ``"conformal"``, ``"gamuon"``, ``"sgd"``.
    lr : float, default 1e-3
        Learning rate.
    betas : (float, float), default (0.9, 0.999)
        Coefficients for first- and second-moment estimates.
    eps : float, default 1e-8
        Numerical stability term.
    weight_decay : float, default 0.0
        Weight decay applied to all parameters.
    foreach : bool, default True
        Whether Gamuon should use fused foreach operations.

    Example
    -------
    >>> model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
    >>> opt = GamuonAuto(model, lr=1e-3)
    >>>
    >>> for step in range(1000):
    ...     loss = model(x).sum()
    ...     opt.zero_grad()
    ...     loss.backward()
    ...     opt.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: bool = True,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.foreach = foreach

        self._conformal: Optional[ConformalMuon] = None
        self._gamuon: Optional[Gamuon] = None
        self._sgd: Optional[torch.optim.SGD] = None

        if isinstance(params, torch.nn.Module):
            self._init_from_module(params)
        else:
            self._init_from_groups(params)

        # Collect sub-optimisers that actually have work to do
        self._optimizers: list[torch.optim.Optimizer] = [
            o for o in (self._conformal, self._gamuon, self._sgd)
            if o is not None and any(
                len(g["params"]) > 0 for g in o.param_groups
            )
        ]

    # ── Module-based initialisation ─────────────────────────────────

    def _init_from_module(self, model: torch.nn.Module) -> None:
        pairs = find_conformal_pairs(model)
        conf_ids: set[int] = set()
        for w, b in pairs:
            conf_ids.add(id(w))
            if b is not None:
                conf_ids.add(id(b))

        gamuon_params: list[torch.Tensor] = []
        sgd_params: list[torch.Tensor] = []

        for p in model.parameters():
            if id(p) in conf_ids:
                continue
            if p.ndim == 2:
                gamuon_params.append(p)
            else:
                sgd_params.append(p)

        if pairs:
            self._conformal = ConformalMuon(
                pairs,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        if gamuon_params:
            self._gamuon = Gamuon(
                [{"params": gamuon_params}],
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                foreach=self.foreach,
            )
        if sgd_params:
            self._sgd = torch.optim.SGD(
                [{"params": sgd_params}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    # ── Manual group-based initialisation ───────────────────────────

    def _init_from_groups(self, param_groups: Iterable[dict]) -> None:
        conf_params: list[dict] = []
        gamuon_params: list[dict] = []
        sgd_params: list[dict] = []

        for group in param_groups:
            role = group.get("role", "sgd")
            if role == "conformal":
                conf_params.append(group)
            elif role == "gamuon":
                gamuon_params.append(group)
            else:
                sgd_params.append(group)

        if conf_params:
            self._conformal = ConformalMuon(
                conf_params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        if gamuon_params:
            self._gamuon = Gamuon(
                gamuon_params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                foreach=self.foreach,
            )
        if sgd_params:
            self._sgd = torch.optim.SGD(
                sgd_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    # ── Public API ──────────────────────────────────────────────────

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimisation step.

        Iterates over all active sub-optimizers (ConformalMuon \u2192 Gamuon
        \u2192 SGD) and calls their respective ``step()``.  The closure is
        only passed to the first sub-optimizer; subsequent sub-optimizers
        receive ``closure=None``.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        float or None
            The loss from ``closure``, or ``None`` if no closure was given.

        Example
        -------
        >>> model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        >>> opt = GamuonAuto(model, lr=1e-3)
        >>>
        >>> # Standard usage (no closure)
        >>> for x, y in dataloader:
        ...     opt.zero_grad()
        ...     loss = model(x).sum()
        ...     loss.backward()
        ...     opt.step()
        >>>
        >>> # With closure (e.g. LBFGS-style)
        >>> def closure():
        ...     loss = model(x).sum()
        ...     loss.backward()
        ...     return loss
        >>> opt.step(closure)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for opt in self._optimizers:
            opt.step(closure=None)
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clear the gradients of all parameters.

        Delegates to each active sub-optimizer's ``zero_grad()``.
        The behaviour of ``set_to_none`` matches PyTorch's semantics:
        ``False`` zeroes the tensors in-place, ``True`` sets the
        gradient attributes to ``None`` (lower memory footprint).

        Parameters
        ----------
        set_to_none : bool, default False
            If ``True``, sets gradients to ``None`` instead of zeroing.
            Reduces memory by allowing PyTorch to reclaim gradient
            storage after each step.

        Example
        -------
        >>> opt = GamuonAuto(model, lr=1e-3)
        >>>
        >>> # Default: zero in-place
        >>> opt.zero_grad()
        >>>
        >>> # Memory-efficient: set to None
        >>> opt.zero_grad(set_to_none=True)
        """
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Return the state of all sub-optimisers as a nested dict.

        Compatible with ``torch.save`` / ``torch.load`` for checkpointing.
        Missing sub-optimisers (those with no parameters) are stored as
        empty dicts ``{}`` so the three-key structure is always preserved.

        Returns
        -------
        dict
            Nested dict with keys ``"conformal"``, ``"gamuon"``,
            ``"sgd"``, each mapping to the corresponding sub-optimiser's
            state dict (or ``{}`` if the sub-optimiser is not active).

        Example
        -------
        >>> opt = GamuonAuto(model, lr=1e-3)
        >>>
        >>> # Save checkpoint
        >>> torch.save({
        ...     "model_state": model.state_dict(),
        ...     "optimizer_state": opt.state_dict(),
        ... }, "checkpoint.pt")
        >>>
        >>> # Inspect saved keys
        >>> sd = opt.state_dict()
        >>> list(sd.keys())
        ['conformal', 'gamuon', 'sgd']
        """
        return {
            "conformal": self._conformal.state_dict() if self._conformal else {},
            "gamuon": self._gamuon.state_dict() if self._gamuon else {},
            "sgd": self._sgd.state_dict() if self._sgd else {},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load a previously saved state dict.

        Restores the state of each active sub-optimiser from the
        corresponding key in *state_dict*.  Keys for sub-optimisers
        that are not present in this ``GamuonAuto`` instance are
        silently skipped, allowing partial restoration (e.g. loading
        only the conformal sub-optimiser's state).

        Parameters
        ----------
        state_dict : dict
            A dict with the same three-key structure produced by
            :meth:`state_dict`.  Only the keys matching active
            sub-optimisers are consumed; extra or missing keys
            are ignored without error.

        Example
        -------
        >>> opt = GamuonAuto(model, lr=1e-3)
        >>>
        >>> # Save and later restore
        >>> torch.save({
        ...     "model_state": model.state_dict(),
        ...     "optimizer_state": opt.state_dict(),
        ... }, "checkpoint.pt")
        >>>
        >>> checkpoint = torch.load("checkpoint.pt")
        >>> opt.load_state_dict(checkpoint["optimizer_state"])
        """
        if self._conformal and "conformal" in state_dict:
            self._conformal.load_state_dict(state_dict["conformal"])
        if self._gamuon and "gamuon" in state_dict:
            self._gamuon.load_state_dict(state_dict["gamuon"])
        if self._sgd and "sgd" in state_dict:
            self._sgd.load_state_dict(state_dict["sgd"])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GRADIENT NORM MONITOR                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


class GradNormMonitor:
    """Monitor and log gradient norms across GamuonAuto\u2019s sub-optimizers.

    Captures the L2 norm of gradients for each sub-optimizer role
    (``conformal``, ``gamuon``, ``sgd``) at each call to :meth:`capture`.
    Useful for debugging gradient flow, tuning learning rates, and
    detecting vanishing / exploding gradients.

    Parameters
    ----------
    optimizer : GamuonAuto
        The optimiser whose parameters\u2019 gradients will be monitored.
    log_freq : int, default 1
        How often to print a summary line (1 = every call to ``capture()``).
        Set to 0 to suppress live logging.
    silent : bool, default False
        If True, suppress all console output (still stores history).

    Example
    -------
    >>> model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
    >>> opt = GamuonAuto(model, lr=1e-3)
    >>> monitor = GradNormMonitor(opt, log_freq=10)
    >>> for step in range(100):
    ...     loss = model(x).sum()
    ...     opt.zero_grad()
    ...     loss.backward()
    ...     monitor.capture(step)
    ...     opt.step()
    >>> stats = monitor.summary()
    >>> # monitor.plot() if matplotlib is available
    """

    def __init__(
        self,
        optimizer: GamuonAuto,
        log_freq: int = 1,
        silent: bool = False,
    ):
        """Initialise the monitor and detect active sub-optimizer roles.

        Scans ``optimizer``'s three optional sub-optimizer attributes
        (``_conformal``, ``_gamuon``, ``_sgd``) and builds the internal
        role mapping and history storage for those that are active.

        Parameters
        ----------
        optimizer : GamuonAuto
            The meta-optimizer to monitor. The monitor reads gradient norms
            from each sub-optimizer's ``param_groups``; no gradients are
            modified.
        log_freq : int, default 1
            If > 0, prints a ``grad_norm`` line on every call to
            :meth:`capture`.  Set to ``0`` to suppress live logging
            (history is still accumulated).
        silent : bool, default False
            If ``True``, suppress all console output regardless of
            ``log_freq``.  Useful for automated benchmarks or when
            collecting metrics programmatically.

        Notes
        -----
        The role detection is a snapshot at construction time.  Adding or
        removing sub-optimizers after construction is not supported and
        will not be reflected in captures.

        If all three sub-optimizers are ``None`` (empty model), the
        monitor will have no roles, and :meth:`capture` will return an
        empty dict.
        """
        self._optimizer = optimizer
        self._log_freq = log_freq
        self._silent = silent

        # Map sub-optimizer \u2192 role name
        self._roles: dict[torch.optim.Optimizer, str] = {}
        for role, attr in [("conformal", "_conformal"),
                           ("gamuon", "_gamuon"),
                           ("sgd", "_sgd")]:
            sub = getattr(optimizer, attr, None)
            if sub is not None:
                self._roles[sub] = role

        # History: {role: {metric: [values]}}
        self._history: dict[str, dict[str, list]] = {}
        for role in self._roles.values():
            self._history[role] = {
                "step": [],
                "total_norm": [],
                "mean_norm": [],
                "max_norm": [],
                "zero_frac": [],
            }

    def capture(self, step: Optional[int] = None) -> dict[str, dict]:
        """Record gradient norms for all active sub-optimizers.

        Iterates over the parameters of each sub-optimizer, computes the
        L2 norm of each parameter's gradient (if present), and stores
        per-role aggregate statistics in the internal history.

        Parameters
        ----------
        step : int, optional
            Current training step or iteration number.  Included in the
            log line (if ``log_freq > 0``) and stored in history for
            later use by :meth:`plot` and :meth:`summary`.  Can be any
            integer; callers often use the global training step or an
            ``(epoch, batch)``-derived counter.

        Returns
        -------
        dict[str, dict]
            Nested dictionary indexed by role (``"conformal"``,
            ``"gamuon"``, ``"sgd"``).  Each value is a dict with keys:

            - ``total_norm`` *(float)* — L2 norm of the concatenated
              gradient vector for all parameters in this role,
              i.e. ``sqrt(\u03a3_i \u2016p_i.grad\u2016_2^2)``.
            - ``mean_norm`` *(float)* — Average per-parameter L2 norm
              (``total_norm / num_params`` with non-zero gradients, or
              ``0.0`` if all gradients are ``None``).
            - ``max_norm`` *(float)* — Maximum per-parameter L2 norm
              observed in this role (``0.0`` if all gradients are
              ``None``).
            - ``num_params`` *(int)* — Total number of parameters
              tracked in this role, regardless of whether each
              parameter has a gradient attached.
            - ``zero_frac`` *(float)* — Fraction of parameters whose
              gradient is either ``None`` or exactly zero
              (0.0 \u2264 ``zero_frac`` \u2264 1.0).

        Raises
        ------
        RuntimeError
            If a parameter's gradient tensor is on a different device
            than the parameter itself (the ``.norm(2)`` call would
            fail).  This should not happen in normal training loops.

        Notes
        -----
        - Gradients are **not** modified or cleared by this method.
        - If all parameters in a role have ``None`` gradients
          (e.g. before the first backward pass), ``total_norm``,
          ``mean_norm``, and ``max_norm`` are all ``0.0``, while
          ``zero_frac`` is ``1.0``.
        - After ``zero_grad(set_to_none=True)``, gradients become
          ``None``, so the next ``capture()`` sees ``zero_frac = 1.0``.
          After ``zero_grad(set_to_none=False)``, gradients become
          zero-valued tensors, so ``zero_frac`` reflects the fraction
          of parameters with ``p.grad.norm(2) == 0.0`` — typically 1.0
          as well, since the zeros are counted.
        - The order of keys in the returned dict is not guaranteed.
        """
        now: dict[str, dict] = {}

        for sub_opt, role in self._roles.items():
            total_sq = 0.0
            param_norms: list[float] = []
            zero_count = 0
            total_count = 0

            for group in sub_opt.param_groups:
                for p in group["params"]:
                    total_count += 1
                    if p.grad is not None:
                        n = p.grad.data.norm(2).item()
                        param_norms.append(n)
                        total_sq += n * n
                        if n == 0.0:
                            zero_count += 1
                    else:
                        zero_count += 1

            total_norm = total_sq ** 0.5
            mean_norm = sum(param_norms) / len(param_norms) if param_norms else 0.0
            max_norm = max(param_norms) if param_norms else 0.0
            zero_frac = zero_count / total_count if total_count > 0 else 0.0

            rec = {
                "total_norm": total_norm,
                "mean_norm": mean_norm,
                "max_norm": max_norm,
                "num_params": total_count,
                "zero_frac": zero_frac,
            }
            now[role] = rec

            self._history[role]["step"].append(step)
            self._history[role]["total_norm"].append(total_norm)
            self._history[role]["mean_norm"].append(mean_norm)
            self._history[role]["max_norm"].append(max_norm)
            self._history[role]["zero_frac"].append(zero_frac)

        # \u2014 Live logging \u2014
        if not self._silent and self._log_freq > 0 and now:
            step_str = f" [step {step}]" if step is not None else ""
            parts = []
            for role in ("conformal", "gamuon", "sgd"):
                if role in now:
                    parts.append(f"{role}:{now[role]['total_norm']:.3f}")
            # Use print so user sees the log even in a Jupyter / console setting
            print(f"grad_norm{step_str}  {'  '.join(parts)}")

        return now

    def summary(self) -> dict[str, dict]:
        """Return aggregated statistics per role across all captured steps.

        Computes summary statistics on the ``total_norm`` time series
        for each role, plus the mean zero-gradient fraction.  Uses pure
        Python arithmetic (safe for CPU profiling without GPU sync).

        Returns
        -------
        dict[str, dict]
            Nested dictionary indexed by role.  Each value contains:

            - ``count`` *(int)* — Number of :meth:`capture` calls
              recorded.  ``0`` if :meth:`reset` was called or no
              captures have been made.
            - ``mean`` *(float)* — Arithmetic mean of ``total_norm``
              across all captured steps.
            - ``std`` *(float)* — Population standard deviation of
              ``total_norm`` (``0.0`` if ``count \u2264 1``).
            - ``min`` *(float)* — Minimum ``total_norm`` observed.
            - ``max`` *(float)* — Maximum ``total_norm`` observed.
            - ``last`` *(float)* — Most recent ``total_norm`` value.
              Useful for detecting trends relative to the historical
              ``mean``.
            - ``zero_frac_mean`` *(float)* — Average ``zero_frac``
              across all captured steps (0.0 \u2264 value \u2264 1.0).
              A persistently high value (e.g. ``> 0.5``) may indicate
              dead units, frozen parameters, or incorrect gradient flow.

        Notes
        -----
        - All metrics are computed from ``total_norm`` only.
          ``mean_norm`` and ``max_norm`` history are stored but not
          included in the summary output.  Use direct inspection of
          ``self._history`` if per-role per-parameter detail is needed.
        - Returns an all-zeros entry for roles with no recorded steps
          (``count=0``) rather than omitting them, so callers can
          safely index into the result without key-checking.
        """
        result: dict[str, dict] = {}
        for role, hist in self._history.items():
            norms = hist["total_norm"]
            n = len(norms)
            if n == 0:
                result[role] = {
                    "count": 0, "mean": 0.0, "std": 0.0,
                    "min": 0.0, "max": 0.0, "last": 0.0, "zero_frac_mean": 0.0,
                }
            else:
                mean = sum(norms) / n
                variance = sum((x - mean) ** 2 for x in norms) / n if n > 1 else 0.0
                result[role] = {
                    "count": n,
                    "mean": mean,
                    "std": variance ** 0.5,
                    "min": min(norms),
                    "max": max(norms),
                    "last": norms[-1],
                    "zero_frac_mean": sum(hist["zero_frac"]) / n,
                }
        return result

    def plot(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Visualise gradient norm trajectories per sub-optimizer role.

        Generates a two-panel figure with shared x-axis:

        - **Top panel:** ``total_norm`` vs. step for each active role.
        - **Bottom panel:** ``mean_norm`` vs. step for each active role.

        Requires ``matplotlib``.  If not installed, prints a warning
        message and returns silently — the history is still accessible
        via :meth:`summary` or direct attribute inspection.

        Parameters
        ----------
        show : bool, default True
            Whether to display the plot interactively via
            ``plt.show()``.  Set to ``False`` when running in
            non-interactive environments (headless servers, CI, etc.).
        save_path : str, optional
            If provided, saves the figure to this file path at 150 DPI.
            Common values: ``"grad_norms.png"``, ``"plots/step_42.svg"``.
            The file format is inferred from the extension.

        Notes
        -----
        - Steps that were recorded with ``step=None`` are plotted at
          their index in the capture sequence (0, 1, 2, ...) rather
          than omitted.
        - The figure is always closed after display/save to prevent
          memory leaks in long-running training loops.
        - If no history has been captured, the plot axes will be empty
          (each sub-optimizer's line is simply not drawn).

        Examples
        --------
        >>> monitor = GradNormMonitor(optimizer, silent=True)
        >>> for step in range(100):
        ...     # ... forward / backward ...
        ...     monitor.capture(step)
        >>> monitor.plot(show=False, save_path="grad_norm_trajectory.png")
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        except ImportError:
            print("matplotlib not installed \u2014 skipping plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        for role, hist in self._history.items():
            steps = hist["step"]
            if not steps:
                continue
            x = [s if s is not None else i for i, s in enumerate(steps)]
            axes[0].plot(x, hist["total_norm"], label=role,
                         marker=".", markersize=3, linewidth=1)
            axes[1].plot(x, hist["mean_norm"], label=role,
                         marker=".", markersize=3, linewidth=1)

        axes[0].set_ylabel("Total gradient norm")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean gradient norm")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def reset(self) -> None:
        """Clear all accumulated history for all roles.

        Removes every recorded entry from the internal history buffers.
        After calling :meth:`reset`:

        - :meth:`summary` returns ``count=0`` for all roles until new
          data is recorded by subsequent :meth:`capture` calls.
        - :meth:`plot` draws empty axes (no lines).
        - The sub-optimizer references and role mapping are preserved;
          only the time-series data is discarded.

        This is useful for resetting the monitor at the start of a new
        validation run or after a learning-rate change without creating
        a new monitor instance.

        Notes
        -----
        - Existing captures in ``self._history`` are cleared in-place
          using ``list.clear()``, so any references held by the caller
          to individual history lists are also cleared.
        - The monitor's ``log_freq`` and ``silent`` settings are
          unchanged.
        """
        for role, hist in self._history.items():
            hist["step"].clear()
            hist["total_norm"].clear()
            hist["mean_norm"].clear()
            hist["max_norm"].clear()
            hist["zero_frac"].clear()
