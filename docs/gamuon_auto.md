# GamuonAuto API Reference

> **One-stop meta-optimizer** — auto-detects norm pairs, 2-D weight matrices, and other parameters, dispatching each to the optimal geometric or standard optimizer.

---

## Quick Summary

```python
from gamuon import GamuonAuto

optimizer = GamuonAuto(model, lr=1e-3, betas=(0.9, 0.999))
```

That's it. `GamuonAuto` inspects your model's architecture, classifies every
parameter by its algebraic type, and applies the best optimizer to each:

| Parameter type | Optimizer | Why |
|---|---|---|
| Norm-layer (γ, β) pairs | `ConformalMuon` | Affine-group exponential respects the γ–β coupling |
| 2-D weight matrices | `Gamuon` | Grade decomposition + exact rotor via matrix exponential |
| 1-D / other params | Plain SGD | Euclidean gradient descent |

The public API matches `torch.optim.Optimizer` — `step()`, `zero_grad()`,
`state_dict()`, `load_state_dict()` — so it works as a drop-in replacement
for Adam or any other optimizer.

---

## Constructor

```python
GamuonAuto(
    params,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    foreach: bool = True,
)
```

### Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `params` | `nn.Module` **or** iterable of dicts | — | **Module mode**: pass an `nn.Module` and all parameters are auto-discovered. **Manual mode**: pass standard PyTorch param groups each with a `"role"` key. |
| `lr` | `float` | `1e-3` | Base learning rate — shared by all three sub-optimizers. |
| `betas` | `(float, float)` | `(0.9, 0.999)` | Coefficients for first- and second-moment estimates (Adam-style). |
| `eps` | `float` | `1e-8` | Numerical stability term added to denominator. |
| `weight_decay` | `float` | `0.0` | L2 weight decay. Applied as isotropic scalar dilation in `Gamuon`, Lie-algebra decay in `ConformalMuon`, standard L2 in SGD. |
| `foreach` | `bool` | `True` | Whether `Gamuon` should fuse parameter updates for efficiency (passed through to the `Gamuon` sub-optimizer; ignored by `ConformalMuon` and SGD). |

---

## Input Modes

### 1. Module mode (recommended)

Pass any `nn.Module`. Detection is fully automatic:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(768, 768),
    torch.nn.LayerNorm(768),
    torch.nn.Linear(768, 3072),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768),
    torch.nn.LayerNorm(768),
)

optimizer = GamuonAuto(model, lr=3e-4)
```

`GamuonAuto` scans the model with `find_conformal_pairs()` to detect norm
layers, then partitions remaining parameters by dimensionality.

### 2. Manual param-group mode

Pass standard PyTorch param groups with a `"role"` key:

```python
optimizer = GamuonAuto([
    {"params": model.layer1.parameters(), "role": "gamuon"},
    {"params": model.norm.parameters(), "role": "conformal"},
    {"params": model.bias_parameters(), "role": "sgd"},
], lr=1e-3)
```

Valid roles: `"conformal"`, `"gamuon"`, `"sgd"` (default).

---

## Dispatch Logic

When constructed with an `nn.Module`, `GamuonAuto._init_from_module` executes
the following partition:

```
          ┌─────────────────────────────────────┐
          │        model.parameters()           │
          └──────────┬──────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │ find_conformal_pairs │─── (γ, β) pairs → ConformalMuon
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  p.ndim == 2  ?     │─── Yes → Gamuon (grade decompositon + rotor)
          └──────────┬──────────┘
                     │ No
          ┌──────────▼──────────┐
          │    Remainder         │─── → Plain SGD
          └─────────────────────┘
```

### What is detected as conformal?

The `find_conformal_pairs` scanner recognizes:

| Module / Pattern | Detected as | Bias? |
|---|---|---|
| `nn.LayerNorm` | (weight, bias) | ✓ (if `bias=True`) |
| `nn.BatchNorm1d/2d/3d` | (weight, bias) | ✓ |
| `nn.GroupNorm` | (weight, bias) | ✓ |
| `nn.RMSNorm` | (weight, None) | ✗ (no bias) |
| Any module with `weight_norm` applied | `(weight_g, None)` | ✗ |
| `nn.InstanceNorm1d/2d/3d` | (weight, bias) | Requires passing ``types=(nn.InstanceNorm1d, ...)`` to `find_conformal_pairs` explicitly |

Weight-normalized `weight_g` parameters are detected even when they are not
strictly 1-D (e.g. shape `(out_features, 1)` for Linear or
`(out_channels, 1, 1, 1)` for Conv2d).

### What goes to Gamuon?

Any 2-D parameter (`p.ndim == 2`) that was **not** already assigned to a
conformal pair. This includes:

- `nn.Linear` weight matrices
- `nn.Embedding` weight matrices
- Any other 2-D parameter created via `nn.Parameter(torch.randn(m, n))`

### What goes to SGD?

Everything else: 1-D biases, scalar parameters, 3-D+ convolutional kernels,
embedding params that are 1-D, etc.

---

## Public API

All methods match the `torch.optim.Optimizer` interface.

### `step(closure=None)`

```python
def step(self, closure: Optional[Callable] = None) -> Optional[float]
```

Perform a single optimization step. Internally iterates through all three
sub-optimizers in order: ConformalMuon → Gamuon → SGD.

| Argument | Description |
|---|---|
| `closure` | Optional callable that re-evaluates the model and returns the loss. Only passed to the first sub-optimizer (subsequent calls pass `closure=None`). |

**Returns:** The loss from `closure`, or `None`.

### `zero_grad(set_to_none=False)`

```python
def zero_grad(self, set_to_none: bool = False) -> None
```

Clear the gradients of all parameters across all sub-optimizers.

| Argument | Default | Description |
|---|---|---|
| `set_to_none` | `False` | If `True`, sets grads to `None` instead of zeroing (saves memory). |

### `state_dict()`

```python
def state_dict(self) -> dict
```

Return the state of all sub-optimizers as a nested dictionary with three keys:

```python
{
    "conformal": { ... },   # ConformalMuon state dict (or {} if None)
    "gamuon":    { ... },   # Gamuon state dict (or {} if None)
    "sgd":       { ... },   # SGD state dict (or {} if None)
}
```

Compatible with `torch.save()` for checkpointing.

### `load_state_dict(state_dict)`

```python
def load_state_dict(self, state_dict: dict) -> None
```

Load a previously saved state dict. Expects the same three-key format produced
by `state_dict()`. Only loads state for sub-optimizers that exist (missing keys
are silently skipped).

---

## Usage Examples

### Minimal training loop

```python
import torch
from gamuon import GamuonAuto

model = torch.nn.Sequential(
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
)
optimizer = GamuonAuto(model, lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
```

### Custom learning rate schedules

Because `GamuonAuto` does **not** inherit from `torch.optim.Optimizer`, you
cannot pass it directly to `torch.optim.lr_scheduler`. Instead, adjust the
sub-optimizer learning rates directly or use a manual schedule:

```python
optimizer = GamuonAuto(model, lr=1e-3)

for epoch in range(num_epochs):
    # Manual cosine decay
    frac = epoch / num_epochs
    new_lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * frac))

    for opt in optimizer._optimizers:
        for group in opt.param_groups:
            group["lr"] = new_lr

    for x, y in dataloader:
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
```

> **Note:** `_optimizers` is a list of the three sub-optimizer instances
> (only those that have parameters). This is a private attribute — the
> recommended approach is to use a wrapper scheduler class (see below).

### Scheduler wrapper

```python
class GamuonAutoScheduler:
    """Wrapper to apply a PyTorch scheduler to all sub-optimizers."""

    def __init__(self, optimizer: GamuonAuto, scheduler_factory):
        self.optimizer = optimizer
        self.schedulers = []
        for opt in optimizer._optimizers:
            self.schedulers.append(scheduler_factory(opt))

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def get_last_lr(self):
        return [sch.get_last_lr() for sch in self.schedulers]
```

Usage:

```python
optimizer = GamuonAuto(model, lr=1e-3)
scheduler = GamuonAutoScheduler(
    optimizer,
    lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100),
)

for epoch in range(100):
    scheduler.step()
    # ... training loop ...
```

### Mixed precision (AMP)

Works with `torch.cuda.amp` just like any optimizer:

```python
scaler = torch.cuda.amp.GradScaler()
optimizer = GamuonAuto(model, lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = model(x).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient accumulation

```python
optimizer = GamuonAuto(model, lr=1e-3)
accum_steps = 4

for i, (x, y) in enumerate(dataloader):
    loss = model(x).sum() / accum_steps
    loss.backward()

    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Checkpointing

```python
# Save
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}, "checkpoint.pt")

# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
```

---

## Accessing Sub-Optimizers

The three sub-optimizers are stored as public attributes:

| Attribute | Type | Description |
|---|---|---|
| `._conformal` | `Optional[ConformalMuon]` | ConformalMuon instance for norm-layer (γ, β) pairs. `None` if no norm layers detected. |
| `._gamuon` | `Optional[Gamuon]` | Gamuon instance for 2-D weight matrices. `None` if no 2-D params. |
| `._sgd` | `Optional[torch.optim.SGD]` | SGD instance for remaining parameters. `None` if no 1-D/other params. |

You can access these for inspection or custom logic:

```python
optimizer = GamuonAuto(model, lr=1e-3)

# Inspect which sub-optimizers are active
if optimizer._gamuon is not None:
    print(f"Gamuon has {len(optimizer._gamuon.param_groups[0]['params'])} params")

# Get the total number of parameters managed
total = 0
for opt in optimizer._optimizers:
    for group in opt.param_groups:
        total += len(group["params"])
print(f"Total parameter groups: {total}")
```

---

## Equivalence with Manual Composition

`GamuonAuto(model, lr=1e-3)` produces **identical** results to the following
manual setup:

```python
from gamuon import Gamuon, ConformalMuon, find_conformal_pairs
from torch.optim import SGD

pairs = find_conformal_pairs(model)
conf_ids = {id(w) for w, _ in pairs} | {id(b) for _, b in pairs if b is not None}

matrix_params = []
other_params = []
for p in model.parameters():
    if id(p) in conf_ids:
        continue
    (matrix_params if p.ndim == 2 else other_params).append(p)

opt_manual = CombinedOptimizer(
    ConformalMuon(pairs, lr=1e-3),
    Gamuon([{"params": matrix_params}], lr=1e-3) if matrix_params else None,
    SGD([{"params": other_params}], lr=1e-3) if other_params else None,
)
```

The `GamuonAuto` version is equivalent but ~20 lines shorter. See
[benchmarks/benchmark_transformer.py](../benchmarks/benchmark_transformer.py)
for an empirical equivalence check.

---

## Test Coverage

`GamuonAuto` has dedicated unit tests in `test_gamuon.py` (`TestGamuonAuto` class):

| Test | What it verifies |
|---|---|
| `test_auto_partitions_basic_model` | Linear + LayerNorm → 3 sub-optimizers, 4 conformal params, 2 SGD params |
| `test_auto_no_norm_layers` | No ConformalMuon when model has no norm layers |
| `test_auto_only_norm_layers` | No Gamuon when model has no 2-D params |
| `test_auto_weight_norm` | `weight_g` detected as conformal param |
| `test_auto_rmsnorm_groupnorm` | Both RMSNorm and GroupNorm detected correctly |
| `test_auto_empty_model` | No sub-optimizers for empty `nn.Sequential()` |
| `test_auto_training_step` | Forward / backward / step runs without error |
| `test_auto_state_dict_structure` | State dict has valid sub-keys with param IDs |
| `test_auto_state_dict_same_instance_roundtrip` | `load_state_dict` restores state correctly |
| `test_auto_scale_invariant_ln_converges` | Loss decreases over 50 steps with LayerNorm |
| `test_auto_output_shape` | Output changes after optimizer step |

Run them with:

```bash
pytest test_gamuon.py -v -k "TestGamuonAuto"
```

---

## Common Questions

### Is `GamuonAuto` an `Optimizer` subclass?

No — `GamuonAuto` is a plain class that wraps three `Optimizer` instances.
It implements `step()`, `zero_grad()`, `state_dict()`, and `load_state_dict()`
with the same signatures, so it works as a drop-in replacement in most training
loops. The main difference is that some PyTorch utilities (like
`torch.optim.lr_scheduler` or `torch.cuda.amp.GradScaler`) may expect a single
`Optimizer` instance — see the scheduler wrapper example above.

### Can I mix `GamuonAuto` with different learning rates?

Yes — use **manual param-group mode**:

```python
optimizer = GamuonAuto([
    {"params": model.encoder.parameters(), "lr": 1e-3, "role": "gamuon"},
    {"params": model.decoder.parameters(), "lr": 3e-4, "role": "gamuon"},
    {"params": model.norms.parameters(), "lr": 1e-3, "role": "conformal"},
])
```

### Does `GamuonAuto` support `torch.compile`?

Yes — `GamuonAuto` works with compiled models since all three sub-optimizers
operate on raw tensors outside the compiled graph. Enable compilation normally:

```python
model = torch.compile(model)
optimizer = GamuonAuto(model, lr=1e-3)
```

### How do I inspect which parameters went where?

Access the sub-optimizers' `param_groups`:

```python
optimizer = GamuonAuto(model, lr=1e-3)

for name, opt in [
    ("ConformalMuon", optimizer._conformal),
    ("Gamuon", optimizer._gamuon),
    ("SGD", optimizer._sgd),
]:
    if opt is not None:
        for group in opt.param_groups:
            print(f"{name}: {len(group['params'])} params, lr={group['lr']}")
```

---

> *For theoretical motivation and the mathematical details behind each dispatch
> decision, see [§6.6 GamuonAuto: A Unified Meta-Optimizer](theory.md#66-gamuonauto-a-unified-meta-optimizer).*
>
> *For monitoring gradient norms across the three sub-optimizers (live logging,
> summary statistics, plotting), see [`docs/grad_norm.md`](grad_norm.md).*
