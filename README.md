# Gamuon

**Geometric (Clifford) Algebra-native Optimizer** — a reformulation of the Muon optimizer in the language of geometric algebra.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Table of Contents

- [What is Gamuon?](#what-is-gamuon)
- [Theoretical Overview](#theoretical-overview)
  - [Grade Decomposition](#grade-decomposition)
  - [Bivector Exponential & Rotors](#bivector-exponential--rotors)
  - [Versor Sandwich Update](#versor-sandwich-update)
  - [Multivector Momentum](#multivector-momentum)
  - [Comparison with Muon](#comparison-with-muon)
  - [Clifford Algebra Signatures for Architectures](#clifford-algebra-signatures-for-architectures)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [Per-Grade Learning Rates](#per-grade-learning-rates)
  - [Non-Square Matrices](#non-square-matrices)
  - [Weight Decay](#weight-decay)
  - [Comparison with Newton-Schulz Baseline](#comparison-with-newton-schulz-baseline)
- [API Reference](#api-reference)
  - [`grade_decompose`](#grade_decompose)
  - [`bivector_exp`](#bivector_exp)
  - [`rotor_apply`](#rotor_apply)
  - [`MultivectorMomentum`](#multivectormomentum)
  - [`Gamuon`](#gamuon)
  - [`newton_schulz`](#newton_schulz)
  - [`GamuonNS`](#gamuonns)
- [Running Tests](#running-tests)
- [Citation](#citation)
- [References](#references)

---

## What is Gamuon?

Gamuon is a drop-in optimizer for PyTorch that replaces the Newton–Schulz iterations used by [Muon](https://kellerjordan.github.io/posts/muon/) with **exact Clifford algebra operations**. Instead of numerically approximating the orthogonal projection of a gradient matrix, Gamuon:

1. **Decomposes** the gradient into distinct geometric grades: scalar (dilation), bivector (rotation), and strain (symmetric deformation).
2. **Generates exact rotors** from the bivector grade via the matrix exponential, using closed-form Rodrigues formulas for 2×2 and 3×3 and `torch.matrix_exp` for larger matrices.
3. **Applies updates** through the versor sandwich product `R · W · Rᵀ`, which preserves the grade structure of the weight space — the natural action of the Clifford group.

The result is a geometrically principled optimizer that respects the **Lie group structure** of neural network weight spaces.

> **A note on exactness:** For 2×2 and 3×3 matrices, the rotor exponential uses closed-form formulas. For n≥4, `torch.matrix_exp` is used internally, which itself uses iterative Padé approximation — so the rotor is numerically accurate but not algebraically exact for larger matrices.

---

## Theoretical Overview

### Grade Decomposition

In the Clifford algebra $\mathrm{Cl}(n,0)$ (Euclidean signature), any square matrix $G$ decomposes into three geometrically distinct parts:

$$G = \langle G \rangle_0 + \langle G \rangle_2 + \langle G \rangle_+$$

where:

| Grade | Expression | Geometric Meaning |
|---|---|---|
| **Scalar** $\langle G \rangle_0$ | $\frac{\mathrm{tr}(G)}{n} \cdot I$ | Isotropic dilation (uniform scaling) |
| **Bivector** $\langle G \rangle_2$ | $\frac{G - G^\mathsf{T}}{2}$ | Infinitesimal rotation (antisymmetric) |
| **Strain** $\langle G \rangle_+$ | $\frac{G + G^\mathsf{T}}{2} - \langle G \rangle_0$ | Symmetric traceless deformation |

These three grades are **orthogonal** under the Frobenius inner product $\langle A, B \rangle = \mathrm{tr}(A^\mathsf{T} B)$.

### Bivector Exponential & Rotors

A bivector $B = \langle G \rangle_2$ generates a **rotor** $R \in \mathrm{Spin}(n)$ via the matrix exponential:

$$R = \exp(\eta \cdot B)$$

where $\eta$ is the learning rate. This rotor is an **orthogonal transformation** with $\det(R) = 1$:

- **2×2** — Closed-form SO(2) rotation:
  $$B = \begin{pmatrix}0 & \theta \\ -\theta & 0\end{pmatrix}, \quad R = \begin{pmatrix}\cos\theta & \sin\theta \\ -\sin\theta & \cos\theta\end{pmatrix}$$

- **3×3** — Rodrigues formula for SO(3): given the rotation vector $\omega = (B_{2,1}, B_{0,2}, B_{1,0})$ with $\theta = \|\omega\|$,
  $$R = I + \sin\theta \cdot \frac{B}{\theta} + (1 - \cos\theta) \cdot \left(\frac{B}{\theta}\right)^2$$

- **n ≥ 4** — Falls back to `torch.matrix_exp`.

### Versor Sandwich Update

A rotor $R$ acts on a weight matrix $W$ by **conjugation** (the versor sandwich):

$$W' = R \cdot W \cdot R^\mathsf{T}$$

This is the **unique grade-preserving action** of the Clifford group on multivectors. It satisfies:

| Property | Why it matters |
|---|---|
| **Norm-preserving** | $\|W'\|_F = \|W\|_F$ (rotor part only) |
| **Spectrum-preserving** | Eigenvalues of $W$ are rotated, not rescaled |
| **Group action** | $(R_2 R_1) \cdot W \cdot (R_2 R_1)^\mathsf{T} = R_2 \cdot (R_1 \cdot W \cdot R_1^\mathsf{T}) \cdot R_2^\mathsf{T}$ |

The full Gamuon update combines all three grades:

$$W \leftarrow R \cdot W \cdot R^\mathsf{T} - \eta_+ \cdot \langle G \rangle_+ - \eta_0 \cdot \langle G \rangle_0$$

where only the scalar and strain parts drive changes to the singular-value spectrum; the bivector part rotates the weight matrix as a rigid geometric object.

### Multivector Momentum

Gamuon maintains separate exponential moving averages for each grade, analogous to Adam but extended to the Clifford algebra:

```
Grade          First moment (mean)    Second moment (RMS)
─────────────────────────────────────────────────────────
Scalar ⟨·⟩₀    m_s ← β₁·m_s + (1-β₁)·s   v_s ← β₂·v_s + (1-β₂)·s²
Bivector ⟨·⟩₂  m_b ← β₁·m_b + (1-β₁)·b   v_b ← β₂·v_b + (1-β₂)·b²
Strain   ⟨·⟩₊  m_p ← β₁·m_p + (1-β₁)·p   v_p ← β₂·v_p + (1-β₂)·p²
```

Each grade also gets an independent learning rate multiplier (`lr_scalar`, `lr_bivector`, `lr_strain`), allowing fine-grained control over how each geometric mode contributes to training.

### Comparison with Muon

| Aspect | Muon (standard) | Gamuon |
|---|---|---|
| **Orthogonal projection** | Newton–Schulz iterations (approximate, 5–10 iters) | Bivector exponential `exp(B)` (exact) |
| **Grade structure** | Implicit (treats gradient as flat matrix) | Explicit (scalar + bivector + strain) |
| **Weight update** | Additive: $W \leftarrow W - \eta \cdot \mathrm{sign}(G)$ | Versor sandwich: $W \leftarrow R W R^\mathsf{T} - \text{(strain + scalar)}$ |
| **Momentum** | Standard (scalar per parameter) | Grade-aware (separate EMA per grade) |
| **Convergence** | Quadratic (NS fixed-point) | Exact (one-step exponential) |
| **Per-grade LR** | Not applicable | Independent $lr_s, lr_b, lr_p$ |

The Newton–Schulz iteration $X \leftarrow (3X - XX^\mathsf{T}X)/2$ can be understood as a **geometric renormalization group flow** on the singular-value spectrum $\sigma$:

$$\sigma_{k+1} = \frac{3\sigma_k - \sigma_k^3}{2}$$

with superstable fixed points at $\sigma = \pm 1$ and an unstable fixed point at $\sigma = 0$. Gamuon replaces this iterative flow with the **integrated flow** — the exponential map — computing the rotor in a single step.

### Clifford Algebra Signatures for Architectures

Different neural network layers naturally inhabit different Clifford algebras:

| Architecture | Relevant Algebra | Notes |
|---|---|---|
| Dense/MLP | $\mathrm{Cl}(m, n)$ | Weight matrix $W \in \mathbb{R}^{m \times n}$ as bivector |
| Convolutional | $\mathrm{Cl}(c_{\text{out}}, c_{\text{in}}) \otimes \mathrm{Cl}(k_h, k_w)$ | Channel × spatial split; current Gamuon pads to square |
| Attention (Q, K) | $\mathrm{Cl}(d, d)$ (split signature) | Query/key duality; per-matrix Gamuon applicable |
| **LayerNorm** ($\gamma, \beta$) | $\mathrm{Cl}(4,1)$ (Conformal GA) | $\gamma$ = dilation versor, $\beta$ = translation versor in CGA |
| BatchNorm | $\mathrm{Cl}(4,1)$ (Conformal GA) | Same conformal structure as LayerNorm |

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+

### Install from PyPI (coming soon)

```bash
pip install gamuon
```

### Install from source

```bash
git clone https://github.com/pebaryan/gamuon.git
cd gamuon
pip install -e .
```

### Installed with PyTorch

Since PyTorch is platform-specific (CPU/CUDA/ROCm/MPS), it is **not** listed as a hard dependency. Install PyTorch 2.0+ following the [official guide](https://pytorch.org/get-started/locally/), then:

```bash
pip install gamuon[torch]
```

Or install everything at once:

```bash
pip install -e .[torch,dev]  # includes test dependencies
```

---

## Quick Start

```python
import torch
from gamuon import Gamuon

# A simple linear layer
model = torch.nn.Linear(64, 64)
optimizer = Gamuon(model.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
```

---

## Usage Examples

### Basic Usage

```python
import torch
from gamuon import Gamuon

# Any 2D parameters will be optimized via geometric algebra
W = torch.nn.Parameter(torch.randn(32, 32))
opt = Gamuon([W], lr=0.01)

for step in range(100):
    opt.zero_grad()
    loss = (W ** 2).mean()
    loss.backward()
    opt.step()
```

### Per-Grade Learning Rates

Each geometric grade can be controlled independently:

```python
opt = Gamuon(
    model.parameters(),
    lr=1e-3,
    lr_scalar=0.5,     # Half learning rate for dilation updates
    lr_bivector=2.0,   # Double learning rate for rotation updates
    lr_strain=1.0,     # Default learning rate for strain updates
)
```

This is useful when you want to encourage or discourage specific geometric modes — for example, prioritizing rotational updates over scalar dilation in normalization-heavy architectures.

### Non-Square Matrices

Gamuon automatically pads non-square matrices to square, applies the update, and slices back:

```python
# (6, 4) matrix → padded to (6, 6), updated, sliced back to (6, 4)
W = torch.nn.Parameter(torch.randn(6, 4))
opt = Gamuon([W], lr=0.01)  # Works automatically
```

### Weight Decay

Weight decay is applied as an isotropic dilation on the scalar grade:

```python
opt = Gamuon(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

This is geometrically natural: $\mathtt{wd} \cdot W$ is a pure scalar-grade contribution, so it feeds into the dilation component of the update.

### Comparison with Newton-Schulz Baseline

For ablation studies, `GamuonNS` replaces the bivector exponential with standard Newton–Schulz iterations:

```python
from gamuon import GamuonNS

opt_ns = GamuonNS(model.parameters(), lr=1e-3, ns_iters=10)
opt_ga = Gamuon(model.parameters(), lr=1e-3)

# Compare convergence
for step in range(200):
    # ... training loop with each optimizer ...
```

---

## API Reference

### `grade_decompose`

```python
def grade_decompose(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Decompose a square matrix into its geometric grades in $\mathrm{Cl}(n,0)$.

| Parameter | Shape | Description |
|---|---|---|
| `M` | `(n, n)` | Mixed-grade multivector (the gradient matrix) |

| Returns | Description |
|---|---|
| `scalar` | `⟨M⟩₀ = tr(M)/n · I` — isotropic dilation |
| `bivector` | `⟨M⟩₂ = (M − Mᵀ)/2` — infinitesimal rotation |
| `strain` | `⟨M⟩₊ = (M + Mᵀ)/2 − ⟨M⟩₀` — symmetric traceless deformation |

The three grades are orthogonal under the Frobenius inner product and sum exactly to `M`.

---

### `bivector_exp`

```python
def bivector_exp(B: torch.Tensor) -> torch.Tensor
```

Exponential of a bivector (antisymmetric matrix), producing a rotor $R \in \mathrm{Spin}(n)$.

| Parameter | Shape | Description |
|---|---|---|
| `B` | `(n, n)` | Bivector (antisymmetric) matrix |

| Returns | Description |
|---|---|
| `R` | `(n, n)` | Orthogonal matrix with $\det(R) = 1$ |

Uses closed-form expressions:
- **n=2:** SO(2) rotation matrix via $\cos\theta, \sin\theta$.
- **n=3:** Rodrigues formula for SO(3).
- **n≥4:** `torch.matrix_exp(B)`.

---

### `rotor_apply`

```python
def rotor_apply(R: torch.Tensor, W: torch.Tensor) -> torch.Tensor
```

Apply a rotor $R \in \mathrm{Spin}(n)$ to a matrix $W$ via the versor sandwich product.

$$W' = R \cdot W \cdot R^\mathsf{T}$$

| Parameter | Shape | Description |
|---|---|---|
| `R` | `(n, n)` | Rotor (orthogonal matrix, $\det = 1$) |
| `W` | `(n, n)` | Weight matrix to transform |

| Returns | Description |
|---|---|
| `W'` | `(n, n)` | Transformed weight matrix |

This is the **grade-preserving action** of the Clifford group: the geometric grade of $W$ is unchanged by the conjugation.

---

### `MultivectorMomentum`

```python
class MultivectorMomentum(n, device, dtype)
```

Grade-aware momentum buffer maintaining separate EMA estimates for scalar, bivector, and strain grades.

**Buffers:** `scalar_m`, `bivector_m`, `strain_m` (first moments) and `scalar_v`, `bivector_v`, `strain_v` (second moments), each of shape `(n, n)`.

```python
def step(self, scalar, bivector, strain, betas) -> Tuple[6 tensors]
```

| Parameter | Description |
|---|---|
| `scalar`, `bivector`, `strain` | Grade-decomposed gradient components |
| `betas` | `(β₁, β₂)` — first and second moment decay rates |

Returns `(m_s, m_b, m_p, v_s, v_b, v_p)` — the six momentum buffers after the update.

---

### `Gamuon`

```python
class Gamuon(torch.optim.Optimizer)
```

The geometric algebra-native optimizer. Full per-grade momentum with rotor-based updates.

| Parameter | Default | Description |
|---|---|---|
| `params` | — | Iterable of parameters or parameter groups |
| `lr` | `1e-3` | Base learning rate |
| `betas` | `(0.9, 0.999)` | Momentum decay rates $(β₁, β₂)$ |
| `eps` | `1e-8` | Numerical stability term |
| `weight_decay` | `0.0` | L2 weight decay (scalar-grade dilation) |
| `lr_scalar` | `1.0` | Learning rate multiplier for scalar grade |
| `lr_bivector` | `1.0` | Learning rate multiplier for bivector grade |
| `lr_strain` | `1.0` | Learning rate multiplier for strain grade |
| `foreach` | `True` | Fuse parameter updates for efficiency |

**Per-step update:**

1. Pad non-square gradients to square matrices.
2. Apply weight decay as `g ← g + wd · W`.
3. Decompose gradient into `(scalar, bivector, strain)`.
4. Update multivector momentum (grade-aware EMA).
5. Bias-correct moments.
6. Compute scalar step (isotropic dilation).
7. Compute bivector step and generate rotor `R = exp(η · B̂)`.
8. Compute strain step (symmetric deformation).
9. Apply: `W ← R · W · Rᵀ - strain_update - scalar_update`.
10. Unpad and write back.

---

### `newton_schulz`

```python
def newton_schulz(G: torch.Tensor, num_iters: int = 5) -> torch.Tensor
```

Approximate $\mathrm{sign}(G) = UV^\mathsf{T}$ via Newton–Schulz iterations.

The iteration $X_{k+1} = (3X_k - X_k X_k^\mathsf{T} X_k)/2$ converges quadratically to the nearest orthogonal matrix to $G$ in Frobenius norm.

**Caveat:** The iteration is only convergent when all singular values of $G$ lie in $(0, \sqrt{3})$. For arbitrary inputs, spectral normalization is recommended.

---

### `GamuonNS`

```python
class GamuonNS(torch.optim.Optimizer)
```

Hybrid optimizer using Newton–Schulz instead of the exact bivector exponential. Provided as an ablation baseline for isolating the effect of the exact rotor update.

| Parameter | Default | Description |
|---|---|---|
| `params` | — | Iterable of parameters |
| `lr` | `1e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | Momentum decay rates |
| `eps` | `1e-8` | Numerical stability |
| `weight_decay` | `0.0` | L2 weight decay |
| `ns_iters` | `5` | Number of Newton–Schulz iterations |

---

## Running Tests

```bash
pytest test_gamuon.py -v --tb=short
```

The test suite covers:

| Category | Tests | What's Verified |
|---|---|---|
| Grade decomposition | 8 tests | Isotropy, antisymmetry, tracelessness, reconstruction, orthogonality, pure cases, batching |
| Bivector exponential | 7 tests | Orthogonality (2D, 3D, 4D), closed-form accuracy, zero case, composition |
| Rotor sandwich | 3 tests | Norm preservation, composition, identity |
| Multivector momentum | 2 tests | Convergence, return shape |
| Newton-Schulz | 2 tests | Orthogonality, SPD convergence |
| Gamuon optimizer | 7 tests | Linear regression, square matrix, non-square padding, per-grade LR, weight decay, momentum persistence, closure |
| GamuonNS | 1 test | Runtime |
| Edge cases | 5 tests | 2×2, 3×3, 16×16, zero gradient, multiple params |

---

## Citation

If you use Gamuon in your research, please cite:

```bibtex
@software{gamuon2026,
  author = {Peb Ruswono Aryan},
  title = {Gamuon: Geometric (Clifford) Algebra-native Optimizer},
  year = {2026},
  url = {https://github.com/pebaryan/gamuon}
}
```

---

## References

1. **Jordan, K. et al.** — [Muon: An optimizer for matrix-structured parameters](https://kellerjordan.github.io/posts/muon/) (2024).
2. **Hestenes, D.** — *New Foundations for Classical Mechanics* (Kluwer, 1999). The canonical introduction to geometric algebra.
3. **Doran, C. & Lasenby, A.** — *Geometric Algebra for Physicists* (Cambridge, 2003). Comprehensive treatment of Clifford algebras and the conformal model.
4. **Dorst, L., Fontijne, D. & Mann, S.** — *Geometric Algebra for Computer Science* (Morgan Kaufmann, 2007). Algorithms and implementation patterns.
5. **Gavrilov, A. et al.** — "Optimization on the Orthogonal Group via Matrix Exponentials" (2022). Connections between Riemannian optimization and Lie-group methods.
6. **Higham, N. J.** — "Computing the Polar Decomposition — with Applications" (SIAM, 1986). The Newton–Schulz iteration and its convergence analysis.
