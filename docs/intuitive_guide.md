# Gamuon: An Intuitive Guide

> *No geometric algebra required.*

If terms like "Clifford algebra" or "bivector exponential" make your eyes glaze
over — don't worry. This guide explains what Gamuon does, why you might want to
use it, and how to get started — all in plain language.

---

## What is Gamuon?

Gamuon is a **drop-in optimizer for PyTorch** that can sometimes train models
faster or better than Adam, especially for large transformers and other
architectures with many matrix-shaped parameters.

Think of it as a smarter alternative to:

- **Adam** — the default optimizer for most deep learning
- **Muon** — a newer optimizer gaining popularity for large language models

The technical difference involves geometric algebra, but the practical
difference is simple: **Gamuon respects the natural structure of weight
matrices**, treating them as geometric objects rather than flat collections of
numbers.

---

## The Big Idea in Three Analogies

### 1. The Rubber Sheet

Imagine a weight matrix as a **rubber sheet** stretched over a frame.

| Operation | What it does to the rubber sheet | What it does to the weights |
|---|---|---|
| **Rotation** | Turns the sheet in place without stretching it | Changes which directions the layer pays attention to |
| **Scaling** | Inflates or shrinks the sheet uniformly | Changes the overall magnitude of the output |
| **Stretching** | Deforms the sheet along specific directions | Changes the shape of the layer's response |

Most optimizers (Adam, SGD) treat every pixel of the rubber sheet independently
— each pixel gets its own adjustment with no regard for what its neighbors are
doing. Gamuon instead says: "Let me rotate the sheet as a whole, inflate it as
a whole, and stretch it as a whole." This is more physically natural and can
lead to more stable training.

### 2. The Crowded Room

Imagine you're giving directions in a crowded room.

- **Adam** talks to each person individually: "You go left. You go right. You
  stay put." It works, but it doesn't leverage the fact that people are in
  groups.
- **Gamuon** talks to groups: "Everyone on this side, rotate 30 degrees
  clockwise. Everyone, take one step back." It coordinates.

Neural network weight matrices are naturally "grouped" — the weights form a
matrix where rows and columns are connected. Gamuon exploits these connections
instead of ignoring them.

### 3. The Sailing Boat

A sailboat can't just move in any direction arbitrarily — the sail and the hull
impose a **geometry** on how the boat can move (you can't sail directly
upwind). Good sailors work *with* this geometry rather than fighting it.

Neural network weights have their own geometry. Standard optimizers sometimes
push weights in directions they "can't" naturally go, wasting updates. Gamuon
works *with* the geometry, so every update is meaningful.

---

## Quick Start

The simplest way to use Gamuon is via `GamuonAuto` — pass it your model and it
automatically figures out the right optimizer for every parameter:

```python
import torch
from gamuon import GamuonAuto

model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.LayerNorm(512),
)

# One-shot auto optimizer — detects norm layers, 2D matrices, and other params
optimizer = GamuonAuto(model, lr=1e-3)

# Standard training loop — nothing else changes
for x, y in dataloader:
    optimizer.zero_grad()
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
```

Under the hood, `GamuonAuto` partitions your model's parameters into three
groups and dispatches each to the best optimizer:

| Parameter type | Optimizer | What it does |
|---|---|---|
| **Norm-layer (\u03b3, \u03b2) pairs** | `ConformalMuon` | Affine-group exponential (keeping \u03b3 positive) |
| **2-D weight matrices** | `Gamuon` | Grade decomposition + rotor update |
| **1-D / other params** | Plain SGD | Standard gradient descent |

Same API as `torch.optim.Adam`. Same hyperparameters (`lr`, `betas`, `eps`,
`weight_decay`). It just works.

### Advanced: Using Gamuon directly

If you prefer to manage the partitioning yourself (for custom parameter groups
or per-grade learning rates), you can still use `Gamuon` directly:

```python
from gamuon import Gamuon

optimizer = Gamuon(model.parameters(), lr=1e-3, lr_bivector=2.0)
```

> **Note:** `Gamuon` only applies the grade-decomposition update to 2-D weight
> matrices. Norm-layer (\u03b3, \u03b2) parameters are silently skipped since they
> are 1-D. If your model has norm layers (most do), use `GamuonAuto` instead to
> get the full `ConformalMuon` treatment for them.

But for most users and most models, **`GamuonAuto` is the recommended starting
point**.

---

## When to Use Gamuon (vs Adam vs Muon)

| Use this... | When... |
|---|---|
| **Adam** | You want the safe, universal default. Works well everywhere. |
| **GamuonAuto** | You want the simplest all-in-one setup. Auto-detects norm layers, 2D matrices, and other params — one call, done. **Recommended for most users.** |
| **Gamuon** | You have many **square or nearly-square weight matrices** and want per-grade control (e.g., tuning `lr_bivector`). |
| **Muon (standard)** | You want the latest research optimizer for LLMs. Uses Newton-Schulz iterations instead of exact math. |
| **ConformalMuon** | You only want to optimise norm-layer (\u03b3, \u03b2) parameters. |

> **Tip:** See the table under [Quick Start](#quick-start) for how `GamuonAuto`
> maps parameter types to optimisers.

### Practical guidance

- **Gamuon shines when your model has many large square weight matrices.**
  Transformers (QKV projections, output projections, FFN layers) are the
  sweet spot.
- **Gamuon is probably overkill for:** Small models, convolutional networks,
  models dominated by non-matrix parameters (embeddings, biases).
- **Gamuon doesn't support:** 1-D parameters (biases, norm params), 3-D+
  parameters (conv kernels) — these are left untouched or handled by SGD
  fallback.
- **Gamuon + Adam is a valid strategy:** Use Gamuon for weight matrices and
  Adam for everything else. The benchmark script does exactly this.

### How much faster is it?

On a small transformer (d_model=128, 3 layers), all optimizers converge to the
same loss in roughly the same number of steps. The per-step cost is:

> **Sizing note:** These numbers are from a small transformer (d_model=128).
> The overhead scales with matrix size (O(n³) for Gamuon vs O(n²) for Adam),
> so the relative cost is model-dependent. For larger models, the forward/
> backward passes dominate more, making the optimizer overhead a smaller
> fraction of total time.

| Optimizer | Relative speed (vs Adam) |
|---|---|
| Adam | 1× (fastest) |
| Gamuon | ~4–5× slower per step |
| Gamuon+Conf | ~5× slower per step |
| Muon (NS) | ~2× slower per step |

The tradeoff: Gamuon may converge in **fewer steps** because each step is more
geometrically meaningful. Whether this pays off depends on your model size and
training regime.

---

## Going Deeper: The Three Components of an Update

Every weight matrix update in Gamuon has three parts:

### 1. Rotation (the "Rotor" part)

**What it does:** Rotates the weight matrix as a rigid object, like turning a
book on a table. The singular values (the "shape" of the matrix) stay exactly
the same. Gamuon computes this rotation directly from the *antisymmetric* part
of the gradient (the part that flips sign when transposed).

**Why it matters:** This is what makes Gamuon different from Adam. Adam can
only move weights "straight" (additively). Gamuon can also *rotate* them,
which is a more natural operation for a linear transformation.

### 2. Scaling (the "Scalar" part)

**What it does:** Uniformly inflates or shrinks all weights — like turning the
volume knob on a speaker.

**Why it matters:** Controls the overall magnitude of the layer's output.
Weight decay hooks into this component naturally.

### 3. Stretching (the "Strain" part)

**What it does:** Deforms the weight matrix along specific directions without
rotating it — like squeezing a stress ball.

**Why it matters:** Changes the relative importance of different input features
without rotating the whole matrix.

### Putting it together

```python
# Gamuon's internal update for one weight matrix:
#   W_new = Rotate(W) - Scaling - Stretching
#
# Adam's update:
#   W_new = W - (some elementwise adjustment)
#
# The rotation part is what makes Gamuon special.
```

---

## Per-Grade Learning Rates

Because Gamuon separates rotation, scaling, and stretching, you can control
each one independently:

```python
optimizer = Gamuon(
    model.parameters(),
    lr=1e-3,
    lr_bivector=2.0,   # Encourage more rotation (×2 learning rate)
    lr_scalar=0.5,     # Reduce uniform scaling (×0.5)
    lr_strain=1.0,     # Keep stretching at base rate
)
```

**When to adjust these:**

- **Increase `lr_bivector`** during warmup — encourages the model to explore
  different "directions" (rotations) in weight space.
- **Decrease `lr_scalar`** if you notice weights shrinking too much
  (rank collapse).
- **Set `lr_scalar=0`** to make the rotor part *spectrum-preserving* — the
  singular values of each weight matrix won't change from the rotation alone
  (only the strain and scalar parts modify them).

---

## ConformalMuon: For Normalization Layers

Gamuon handles weight matrices. But what about the **normalization layers**
(LayerNorm, BatchNorm, RMSNorm) with their (γ, β) parameters?

Standard Adam treats γ and β independently — they each get their own learning
rate and update. But γ (scale) and β (shift) are **geometrically coupled**:
applying γ then β is different from applying β then γ.

ConformalMuon respects this coupling. The easiest way to use it (along with
Gamuon for weight matrices) is `GamuonAuto`:

```python
from gamuon import GamuonAuto

# Single call — auto-detects everything
optimizer = GamuonAuto(model, lr=1e-3)
```

Or use `ConformalMuon` directly if you only want to optimise norm layers:

```python
from gamuon import ConformalMuon

# Auto-detect all norm layers in your model
optimizer = ConformalMuon(model, lr=1e-3)
```

For fine-grained control, you can also pass explicit (\u03b3, \u03b2) pairs:

```python
from gamuon import ConformalMuon, find_conformal_pairs

pairs = find_conformal_pairs(model)
optimizer = ConformalMuon(pairs, lr=1e-3)
```

**What ConformalMuon does differently:**

| Aspect | Adam | ConformalMuon |
|---|---|---|
| γ (scale) update | γ ← γ - lr × gradient | γ ← γ × exp(...) — **multiplicative** |
| β (shift) update | β ← β - lr × gradient | β ← β × exp(...) + ... — **coupled** |
| γ stuck negative? | Yes (common in training) | **No** — exp(...) is always positive |
| Geometry respected? | No | Yes |

The practical upshot: **ConformalMuon keeps γ positive by construction** (since
`γ' = γ × exp(...) > 0`), which is the mathematically correct behavior for a
scale parameter. Adam can push γ negative, which is physically nonsensical for
a standard deviation.

---

## Combining Everything: The "Gamuon+Conf" Strategy

The recommended setup for transformer training:

```python
from gamuon import GamuonAuto

# Single call — everything is auto-detected and optimised
optimizer = GamuonAuto(model, lr=1e-3)

# Standard training loop
for x, y in dataloader:
    optimizer.zero_grad()
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
```

`GamuonAuto` is what the "Gamuon+Conf" entry in the benchmark uses. It
automatically calls `find_conformal_pairs` under the hood, applies
`ConformalMuon` to norm-layer params, `Gamuon` to 2-D weight matrices, and
plain SGD to everything else.

---

## FAQ

### Is Gamuon a drop-in replacement for Adam?

Almost. Same API, same hyperparameters. The catches:

- Doesn't support sparse gradients
- Only works with 2-D parameters (weight matrices)
- Non-square matrices are padded to square (works automatically)

### How does Gamuon handle non-square matrices?

Most real weight matrices aren't square (e.g., `Linear(768, 3072)` is 768×3072).
Gamuon **automatically pads** them to square, runs the geometric update, then
slices back to the original shape — all inside each step with no permanent
copy:

```python
# (6, 4) matrix → padded to (6, 6), updated, sliced back to (6, 4)
W = torch.nn.Parameter(torch.randn(6, 4))
opt = Gamuon([W], lr=0.01)  # Works automatically
```

### Do I need to understand geometric algebra?

**No.** That's the point of this guide. You can use Gamuon like any other
PyTorch optimizer. The math is an implementation detail.

### Why does Gamuon cost more per step?

Gamuon does three things Adam doesn't:

1. **Grade decomposition** — splitting the gradient into rotation/scale/stretch
2. **Matrix exponential** — computing the rotation from the gradient
3. **Sandwich product** — applying the rotation to the weights

Each of these is a matrix operation (`O(n³)`), adding overhead. The hope is
that you need fewer total steps.

### When is the extra cost worth it?

When your model has many large square weight matrices and training is
bottlenecked by convergence rather than per-step compute. Typically:
transformers with hidden sizes of 1024+.

### Can I use Gamuon with mixed precision (AMP)?

Yes. Gamuon works with `torch.cuda.amp` just like any other optimizer.

### Does Gamuon work with gradient accumulation?

Yes. Same as any optimizer — zero_grad every N steps.

### How does Gamuon compare to Shampoo or SOAP?

These are also "geometric" optimizers, but with different philosophies:

| Optimizer | Core idea | What it costs |
|---|---|---|
| **Gamuon** | Rotate weights using the gradient's antisymmetric part | Matrix exponential (`O(n³)`) |
| **Shampoo** | Precondition with Kronecker-factored gradient statistics | Matrix roots (`O(n³)`) |
| **SOAP** | Eigen-decompose gradient statistics, then Adam in eigenbasis | Eigen decomposition (`O(n³)`) |
| **Muon** | Approximate orthogonal projection via Newton-Schulz | 5–10 matmuls (`O(n³)`) |

All are `O(n³)` per step for each weight matrix. Gamuon's advantage is that
the matrix exponential produces an **exact** rotation, while Newton-Schulz
gives an approximation.

### I see the word "versor" a lot. What's that?

A **versor** is just a fancy name for a geometric operation that can be
applied via the "sandwich" formula `R · W · Rᵀ`. It's the geometric algebra
equivalent of a rotation matrix. Don't overthink it.

### What about weight decay?

Weight decay in Gamuon is applied as part of the scalar (scale) component —
it uniformly shrinks the weights, which is exactly what L2 regularization
should do geometrically.

```python
# Standard weight decay — works as expected
optimizer = Gamuon(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

## Glossary (for when you encounter the math docs)

| Term | Plain English | Why it matters |
|---|---|---|
| **Grade decomposition** | Splitting a matrix into rotation + scale + stretch parts | The secret sauce — allows each part to be updated differently |
| **Bivector** | The antisymmetric part of a matrix (skew-symmetric: `B = -Bᵀ`) | Generates the rotation update from the gradient's antisymmetric part |
| **Scalar grade** | The uniform scale part of a matrix | Controls overall magnitude |
| **Strain** | The stretching/compression part of a matrix | Changes shape without rotating |
| **Rotor** | A pure rotation (like a rotation matrix) | Applied to weights via the sandwich formula |
| **Versor sandwich** | `R · W · Rᵀ` — rotating a matrix by applying R from both sides | How rotations are applied to weights |
| **Clifford algebra** | A mathematical framework for handling rotations, scales, and stretches in a unified way | The underlying math (you don't need to understand it) |
| **CGA / Conformal GA** | Geometric algebra extended to handle scaling and translation together | Used by ConformalMuon for norm layers |
| **Affine group Aff(1)** | The group of "scale then shift" operations (`x → γx + β`) | The mathematical structure of normalization layers |

---

## Further Reading

If you *do* want to understand the math:

- **[docs/theory.md](theory.md)** — The full theoretical treatment (heavy math
  warning)
  - **[§6.6 GamuonAuto](theory.md#66-gamuonauto-a-unified-meta-optimizer)** —
    The mathematical motivation for the auto-dispatch rule, equivalence proof,
    and why each parameter type gets its assigned optimizer
- **[docs/gamuon_auto.md](gamuon_auto.md)** — Dedicated API reference with
  parameter tables, usage examples, and checkpointing guide
- **README.md** — Quick overview with some equations
- The `gamuon.py` source — Well-commented with docstrings

### Performance tip: `foreach`

Gamuon has a `foreach=True` parameter (on by default) that fuses operations
across all parameters for better GPU utilization. You usually don't need to
think about this, but if you're debugging, you can set `foreach=False` to
process parameters one at a time (slower but easier to debug).

---

If you want to see benchmarks:

- **[benchmarks/benchmark_transformer.py](../benchmarks/benchmark_transformer.py)**
  — Run your own comparison

---

> *This guide is intentionally non-mathematical. If something is unclear or you
> have questions, please open a GitHub issue.*
