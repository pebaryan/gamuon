# GradNormMonitor

> **Gradient norm monitoring utility** — captures per-role gradient statistics across `GamuonAuto`'s sub-optimizers (conformal, gamuon, sgd) for debugging gradient flow, tuning learning rates, and detecting vanishing / exploding gradients.

---

## Quick Summary

```python
from gamuon import GamuonAuto, GradNormMonitor

model = torch.nn.Sequential(
    torch.nn.Linear(64, 64),
    torch.nn.LayerNorm(64),
)
optimizer = GamuonAuto(model, lr=1e-3)
monitor = GradNormMonitor(optimizer, log_freq=10)

for step in range(100):
    loss = model(x).sum()
    optimizer.zero_grad()
    loss.backward()
    monitor.capture(step)      # record gradient norms
    optimizer.step()

stats = monitor.summary()      # aggregated statistics
# monitor.plot()               # visualisation (if matplotlib is installed)
```

At each `capture()` call, `GradNormMonitor` records the total, mean, and max L2
norm of gradients for each active sub-optimizer role, along with the fraction
of parameters that have zero (or missing) gradients.

---

## Constructor

```python
GradNormMonitor(
    optimizer: GamuonAuto,
    log_freq: int = 1,
    silent: bool = False,
)
```

### Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `GamuonAuto` | — | The meta-optimizer whose sub-optimizers will be monitored. The monitor inspects `._conformal`, `._gamuon`, and `._sgd` to determine which roles are active. |
| `log_freq` | `int` | `1` | Print a live `grad_norm` summary line on every call to `capture()` when > 0. Set to `0` to suppress all live logging (history is still recorded). |
| `silent` | `bool` | `False` | If `True`, suppress all console output regardless of `log_freq`. Useful in automated scripts or when collecting metrics programmatically. |

### Role Detection

The monitor detects which roles are active by inspecting the three optional
sub-optimizer attributes of `GamuonAuto`:

| `GamuonAuto` attribute | Monitor role key | Present when… |
|---|---|---|
| `._conformal` | `"conformal"` | Model has norm-layer (γ, β) pairs |
| `._gamuon` | `"gamuon"` | Model has 2-D weight matrices |
| `._sgd` | `"sgd"` | Model has 1-D / other parameters |

Only roles with a non-`None` sub-optimizer appear in capture results and history.

---

## Public API

### `capture(step=None)`

```python
def capture(self, step: Optional[int] = None) -> dict[str, dict]
```

Record gradient norms for all active sub-optimizers. This is the primary
method — call it after `loss.backward()` (and before or after `optimizer.step()`,
depending on what you want to measure).

| Argument | Description |
|---|---|
| `step` | Optional integer step number, included in logs and history for later reference. |

**Returns** — A nested dict keyed by role (`"conformal"`, `"gamuon"`, `"sgd"`),
each containing:

| Key | Type | Description |
|---|---|---|
| `total_norm` | `float` | L2 norm of the concatenated gradient vector for this role: <br> `sqrt(∑ ‖p.grad‖₂²)` |
| `mean_norm` | `float` | Average per-parameter L2 norm for this role. |
| `max_norm` | `float` | Maximum per-parameter L2 norm for this role. |
| `num_params` | `int` | Number of parameters tracked in this role. |
| `zero_frac` | `float` | Fraction of parameters in this role that have a zero gradient or `None` gradient (0.0 – 1.0). |

**Live logging.** When `log_freq > 0` and `silent=False`, `capture()` prints a
line on **every** call:

```
grad_norm [step 42]  conformal:1.234  gamuon:5.678  sgd:0.098
```

The output is binary: if `log_freq > 0`, every call prints. Set `log_freq=0`
or `silent=True` to suppress.

### `summary()`

```python
def summary(self) -> dict[str, dict]
```

Return aggregated statistics across all captured steps, per role.

**Returns** — A nested dict keyed by role, each containing:

| Key | Type | Description |
|---|---|---|
| `count` | `int` | Number of captured steps. |
| `mean` | `float` | Mean `total_norm` across all steps. |
| `std` | `float` | Population standard deviation of `total_norm`. |
| `min` | `float` | Minimum `total_norm` observed. |
| `max` | `float` | Maximum `total_norm` observed. |
| `last` | `float` | Most recent `total_norm`. |
| `zero_frac_mean` | `float` | Average `zero_frac` across all steps (0.0 – 1.0). |

### `plot(show=True, save_path=None)`

```python
def plot(self, show: bool = True, save_path: Optional[str] = None) -> None
```

Visualise gradient norm trajectories for each active role across captured
steps. Requires **matplotlib** — if not installed, prints a warning and
returns silently.

The plot contains two stacked subplots:
- **Top**: Total gradient norm vs. step
- **Bottom**: Mean gradient norm vs. step

Each role is plotted as a separate line with markers.

| Argument | Default | Description |
|---|---|---|
| `show` | `True` | Whether to display the plot interactively via `plt.show()`. |
| `save_path` | `None` | If provided, saves the figure to this path (e.g. `"grad_norms.png"`). |

> **Saving without displaying:** Set `show=False, save_path="plot.png"` to save
> silently — useful in automated benchmark scripts.

### `reset()`

```python
def reset(self) -> None
```

Clear all accumulated history. Subsequent `capture()` calls start from a
clean slate, and `summary()` returns `count=0` for all roles until new data
is recorded.

---

## Usage Examples

### Basic training loop with live monitoring

```python
import torch
from gamuon import GamuonAuto, GradNormMonitor

model = torch.nn.Sequential(
    torch.nn.Linear(64, 64),
    torch.nn.LayerNorm(64),
    torch.nn.Linear(64, 10),
)
optimizer = GamuonAuto(model, lr=1e-3)
monitor = GradNormMonitor(optimizer, log_freq=10)

for epoch in range(10):
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()

        # Record gradient norms (before step — captures raw gradient sizes)
        monitor.capture(step=epoch * len(dataloader) + batch_idx)

        optimizer.step()

# Print final aggregated statistics
stats = monitor.summary()
for role, s in stats.items():
    print(f"{role}: mean={s['mean']:.4f} ± {s['std']:.4f}, "
          f"last={s['last']:.4f}, zero_frac={s['zero_frac_mean']:.3f}")
```

### Detecting vanishing gradients

Use `monitor.summary()` after the first few steps to flag roles with
unusually small norms:

```python
monitor = GradNormMonitor(optimizer, silent=True)

for step, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    monitor.capture(step)
    optimizer.step()

    if step == 10:   # Check after 10 steps
        stats = monitor.summary()
        for role, s in stats.items():
            if s["mean"] < 1e-6:
                print(f"WARNING: vanishing gradients in {role} "
                      f"(mean={s['mean']:.2e})")
            if s["zero_frac_mean"] > 0.5:
                print(f"WARNING: >50% zero gradients in {role} "
                      f"(zero_frac={s['zero_frac_mean']:.2f})")
```

### Comparing norms before and after optimizer step

Call `capture()` both before and after `step()` to measure how much
the sub-optimizers change the gradient distribution:

```python
monitor = GradNormMonitor(optimizer, silent=True)

for step in range(100):
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()

    norms_before = monitor.capture(step=step * 2)
    optimizer.step()

    # Re-capture after step — most gradients are still intact unless
    # zero_grad was called; this is useful when step() modifies gradients
    # in-place (e.g. weight decay addition)
    norms_after = monitor.capture(step=step * 2 + 1)
```

> **Note:** `optimizer.step()` does **not** clear gradients — call
> `optimizer.zero_grad()` before the next backward pass. The
> `capture()` → `step()` pattern above shows that gradients persist
> after `step()` unless they are modified in-place.

### Silently collecting metrics for experiments

For automated benchmark scripts, suppress all console output:

```python
monitor = GradNormMonitor(optimizer, silent=True)

for step in range(2000):
    # ... training ...
    monitor.capture(step)

# Save summary to a file
stats = monitor.summary()
import json
with open("grad_norm_summary.json", "w") as f:
    json.dump(stats, f, indent=2)

# Save plot if matplotlib is available
monitor.plot(show=False, save_path="grad_norm_trajectory.png")
```

---

## Interpreting the Metrics

| Metric | What it tells you |
|---|---|
| **`total_norm`** | Overall magnitude of the gradient signal for this parameter group. A sudden drop may indicate vanishing gradients; a sudden spike may indicate an exploding gradient. |
| **`mean_norm`** | Average per-parameter magnitude. Useful for comparing gradient sizes across roles with different numbers of parameters. |
| **`max_norm`** | Largest single-parameter gradient. Useful for detecting outlier parameters with disproportionately large updates. |
| **`zero_frac`** | Fraction of parameters with zero or `None` gradients. A high value (e.g. > 0.5) may indicate dead units, incorrect gradient flow, or frozen parameters. |
| **`zero_frac_mean`** | Average zero fraction across all captured steps. Consistently high values suggest a structural issue (e.g., wrong parameter partition). |

### Typical ranges

| Context | `total_norm` | `zero_frac` |
|---|---|---|
| Healthy training (conformal) | 1e-2 – 1e1 | 0.0 – 0.1 |
| Healthy training (gamuon) | 1e-1 – 1e2 | 0.0 – 0.05 |
| Healthy training (sgd) | 1e-3 – 1e0 | 0.0 – 0.1 |
| Vanishing gradients | < 1e-6 | > 0.5 |
| Exploding gradients | > 1e4 | < 0.01 |
| After `zero_grad()` | 0.0 | 1.0 (if `set_to_none=True`) or ~1.0 (if `False`, since zero-valued grads are counted as zero) |

---

## Test Coverage

`GradNormMonitor` has dedicated unit tests in `test_gamuon.py`
(`TestGradNormMonitor` class):

| Test | What it verifies |
|---|---|
| `test_capture_returns_all_active_roles` | Returns dict with all three roles (conformal, gamuon, sgd) when all are present |
| `test_capture_norms_are_positive` | After a backward pass, norms are strictly positive |
| `test_capture_without_backward_zeros` | Without a backward pass, `zero_frac = 1.0` and `total_norm = 0.0` |
| `test_capture_after_zero_grad` | After `zero_grad()`, norms are zero but grads are not `None` |
| `test_capture_after_step_zero_grad` | Full forward → backward → step → zero_grad cycle produces expected norms |
| `test_summary_returns_aggregated_stats` | `summary()` returns correct count, mean, std, min, max, last |
| `test_reset_clears_history` | `reset()` clears all accumulated history |
| `test_log_freq_zero_no_output` | `log_freq=0` suppresses console output |
| `test_log_freq_one_prints` | `log_freq=1` prints `grad_norm` lines with role stats |
| `test_only_gamuon_active` | Model with only 2-D params (no norms) → only `"gamuon"` role |
| `test_multiple_steps_track_history_length` | After N captures, each role has N entries |
| `test_num_params_correct` | `num_params` reflects actual parameter count per role |

Run them with:

```bash
pytest test_gamuon.py -v -k "TestGradNormMonitor"
```

---

## Common Questions

### Can I use `GradNormMonitor` without `GamuonAuto`?

No — `GradNormMonitor` is specifically designed to inspect the three
sub-optimizers of a `GamuonAuto` instance. For a plain optimizer (e.g.
`torch.optim.Adam`), use PyTorch's built-in gradient utilities or a
custom hook.

### Does `capture()` modify gradients?

No — it only reads `p.grad.data.norm(2)` for each parameter and copies the
scalar value into the history. Gradients are not modified.

### What happens if I call `capture()` multiple times without a new backward pass?

The same gradient values will be recorded again. If you call `capture()`
before and after `optimizer.step()`, the norms will be identical unless
the optimizer modifies gradients in-place (e.g., weight decay addition
in `Gamuon` and `ConformalMuon`). Call `optimizer.zero_grad()` between
captures to see zero gradients.

### Why does `zero_frac` include both zero-valued and `None` gradients?

Because both cases mean the parameter is not receiving a meaningful update.
A `None` gradient (after `zero_grad(set_to_none=True)`) and a zero-valued
gradient both contribute zero to `total_norm`, so they are counted together.

### How much memory does `history` use?

Very little — only scalar values are stored (five floats plus an optional
integer per role per capture). After 10,000 steps with 3 active roles,
the history holds ~150,000 float values (~1.2 MB).

---

> *For the implementation, see the [`GradNormMonitor` class](../gamuon.py).*
