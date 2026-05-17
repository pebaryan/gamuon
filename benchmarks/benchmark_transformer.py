"""
Benchmark: Gamuon vs Muon (Newton-Schulz) vs Adam
==================================================

Trains a small transformer language model with each optimizer and compares
convergence speed (loss vs steps), wall-clock time, and final loss.

Usage:
    python benchmarks/benchmark_transformer.py           # quick run (200 steps)
    python benchmarks/benchmark_transformer.py --steps 1000   # longer run
    python benchmarks/benchmark_transformer.py --no-plot      # no PDF output
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Gamuon ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gamuon import (Gamuon, GamuonNS, grade_decompose, newton_schulz,
                       ConformalMuon, find_conformal_pairs)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SMALL TRANSFORMER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention (simplified for benchmarking)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return self.out_proj(attn @ v)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    """Minimal decoder-only transformer for benchmarking."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        d_ff: int = 512,
        n_layers: int = 3,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Causal mask
        self.register_buffer("causal_mask", None)

    def _make_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self._make_mask(T, x.device)
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MUON WRAPPER  (Newton-Schulz via GamuonNS, with proper 2D-only)   ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MuonOptimizer(torch.optim.Optimizer):
    """Pure Muon-style optimizer using Newton-Schulz iterations.

    Applies the NS projection to all 2D parameters (weight matrices),
    and standard SGD to everything else (biases, 1D params).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ns_iters: int = 5,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, ns_iters=ns_iters)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            ns_iters = group["ns_iters"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                # Weight decay
                if wd != 0:
                    g.add_(p.data, alpha=wd)

                if g.dim() == 2 and g.shape[0] > 1 and g.shape[1] > 1:
                    # Pad to square if needed, run NS, then slice back
                    m, n = g.shape
                    needs_pad = m != n
                    if needs_pad:
                        max_dim = max(m, n)
                        padded = g.new_zeros(max_dim, max_dim)
                        padded[:m, :n] = g
                        orig_g = g
                        g = padded

                    # Normalize to guarantee NS convergence
                    scale = g.norm() + 1e-10
                    g_norm = g / scale
                    for _ in range(ns_iters):
                        g_norm = (3 * g_norm - g_norm @ g_norm.T @ g_norm) / 2

                    if needs_pad:
                        update = g_norm[:m, :n]
                    else:
                        update = g_norm

                    p.data.add_(update, alpha=-lr)
                else:
                    # Non-matrix params: plain SGD
                    p.data.add_(g, alpha=-lr)

        return loss


class CombinedOptimizer:
    """Wraps two optimizers into one for combined step().

    Used to pair Gamuon (2D params) with SGD (biases/norms).
    """

    def __init__(self, *optimizers):
        self.optimizers = [o for o in optimizers if o.param_groups and any(
            len(g["params"]) > 0 for g in o.param_groups
        )]

    def zero_grad(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step(*args, **kwargs)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TRAINING LOOP                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


def generate_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic next-token prediction data."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Shift right: predict token i+1 from tokens 0..i
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = 0  # pad last position
    return x, y


def get_param_groups(model: nn.Module, lr: float) -> list[dict]:
    """Return parameter groups: 2D params + biases/1D params."""
    matrix_params = []
    other_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2:
            matrix_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if matrix_params:
        groups.append({"params": matrix_params, "lr": lr})
    if other_params:
        groups.append({"params": other_params, "lr": lr})
    return groups


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    label: str,
) -> dict:
    """Run training and return loss history + timing data."""
    model.train()
    losses = []
    step_times = []
    total_start = time.perf_counter()

    for step in range(steps):
        x, y = generate_batch(vocab_size, batch_size, seq_len, device)

        step_start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            ignore_index=0,
        )
        loss.backward()
        optimizer.step()
        step_times.append(time.perf_counter() - step_start)

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"  [{label}] step {step + 1:5d}/{steps}  loss {loss.item():.4f}  "
                  f"time/step {step_times[-1]:.4f}s")

    total_time = time.perf_counter() - total_start

    return {
        "label": label,
        "steps": steps,
        "losses": losses,
        "loss_final": losses[-1],
        "loss_min": min(losses),
        "total_time_s": total_time,
        "mean_step_time_s": sum(step_times) / len(step_times),
        "median_step_time_s": sorted(step_times)[len(step_times) // 2],
        "step_times": step_times,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


def plot_results(
    results: list[dict],
    save_path: Optional[Path] = None,
):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[benchmark] matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "Gamuon": "#4C72B0",
        "Gamuon+Conf": "#8E44AD",
        "Muon (NS)": "#DD8452",
        "Adam": "#55A868",
    }
    markers = {
        "Gamuon": "o",
        "Gamuon+Conf": "D",
        "Muon (NS)": "s",
        "Adam": "^",
    }

    # ── Loss vs steps ──────────────────────────────────────────────
    ax = axes[0, 0]
    for r in results:
        label = r["label"]
        steps_arr = range(1, r["steps"] + 1)
        ax.plot(steps_arr, r["losses"], label=label, color=colors.get(label, "gray"),
                alpha=0.9, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Loss (smoothed) ────────────────────────────────────────────
    ax = axes[0, 1]
    window = max(1, min(r["steps"] for r in results) // 20)
    for r in results:
        label = r["label"]
        losses = torch.tensor(r["losses"])
        kernel = torch.ones(window) / window
        # conv1d with padding produces len(input) + padding*2 - kernel + 1 elements
        pad = window // 2
        smoothed = F.conv1d(
            losses.view(1, 1, -1), kernel.view(1, 1, -1), padding=pad
        ).squeeze().numpy()
        # Trim to match original length (center crop)
        excess = len(smoothed) - r["steps"]
        if excess > 0:
            smoothed = smoothed[excess // 2:excess // 2 + r["steps"]]
        steps_arr = range(1, r["steps"] + 1)
        ax.plot(steps_arr, smoothed, label=label,
                color=colors.get(label, "gray"), linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title(f"Smoothed Loss  (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Final loss bar chart ───────────────────────────────────────
    ax = axes[1, 0]
    labels = [r["label"] for r in results]
    final_losses = [r["loss_final"] for r in results]
    min_losses = [r["loss_min"] for r in results]
    x_pos = range(len(labels))
    width = 0.35
    bars1 = ax.bar([p - width / 2 for p in x_pos], final_losses, width,
                   label="Final loss", color=[colors.get(l, "gray") for l in labels],
                   alpha=0.8)
    bars2 = ax.bar([p + width / 2 for p in x_pos], min_losses, width,
                   label="Min loss", color=[colors.get(l, "gray") for l in labels],
                   alpha=0.4, hatch="//")
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Loss")
    ax.set_title("Final & Minimum Loss")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── Step time box plot ─────────────────────────────────────────
    ax = axes[1, 1]
    time_data = [r["step_times"] for r in results]
    bp = ax.boxplot(time_data, labels=[r["label"] for r in results],
                    patch_artist=True)
    for patch, label in zip(bp["boxes"], [r["label"] for r in results]):
        patch.set_facecolor(colors.get(label, "gray"))
        patch.set_alpha(0.6)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Step time (s)")
    ax.set_title("Per-Step Wall-Clock Time")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Optimizer Comparison  —  Small Transformer",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[benchmark] Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Gamuon vs Muon vs Adam on a small transformer"
    )
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of training steps (default: 200)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip PDF plot generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[benchmark] Device: {device}")
    print(f"[benchmark] Transformer: {args.n_layers} layers, "
          f"d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"[benchmark] Data: vocab={args.vocab_size}, "
          f"batch={args.batch_size}, seq_len={args.seq_len}")
    print(f"[benchmark] Steps: {args.steps}, LR: {args.lr}, Seed: {args.seed}")
    print()

    torch.manual_seed(args.seed)

    # ── Build model (shared across all optimizers) ─────────────────
    # We create a fresh copy for each optimizer to ensure fair comparison
    def make_model():
        return SmallTransformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            max_len=args.seq_len + 1,
        ).to(device)

    # ── Define optimizers ──────────────────────────────────────────
    # Gamuon w/ SGD fallback:   Garnuon for 2D matrix params,
    #                            SGD for biases & norm params.
    # Gamuon+Conf:               Gamuon for matrix params,
    #                            ConformalMuon for norm (γ, β),
    #                            SGD for remaining non-2D params.
    # Muon:                      NS projection for 2D + SGD for rest.
    # Adam:                      Standard baseline (all params).

    def make_gamuon(model):
        groups = get_param_groups(model, args.lr)
        matrix_group = None
        other_group = None
        for g in groups:
            if g["params"][0].ndim == 2:
                matrix_group = {"params": g["params"], "lr": g["lr"]}
            else:
                other_group = {"params": g["params"], "lr": g["lr"]}

        gamuon_opt = Gamuon(
            [matrix_group] if matrix_group else [],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        sgd_opt = torch.optim.SGD(
            [other_group] if other_group else [],
            lr=args.lr,
        )
        return CombinedOptimizer(gamuon_opt, sgd_opt)

    def make_gamuon_conformal(model):
        """Gamuon for matrix weights + ConformalMuon for norm (γ, β)."""
        # Detect norm-layer parameter pairs
        pairs = find_conformal_pairs(model)
        conformal_ids = set()
        for w, b in pairs:
            conformal_ids.add(id(w))
            if b is not None:
                conformal_ids.add(id(b))

        # Partition remaining params: 2D → Gamuon, rest → SGD
        matrix_params = []
        other_params = []
        for _, p in model.named_parameters():
            if id(p) in conformal_ids:
                continue
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                other_params.append(p)

        gamuon_opt = Gamuon(
            [{"params": matrix_params, "lr": args.lr}] if matrix_params else [],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        conformal_opt = ConformalMuon(
            pairs,
            lr=args.lr,
            betas=(0.9, 0.999),
        )
        sgd_opt = torch.optim.SGD(
            [{"params": other_params, "lr": args.lr}] if other_params else [],
            lr=args.lr,
        )
        return CombinedOptimizer(gamuon_opt, conformal_opt, sgd_opt)

    def make_muon(model):
        return MuonOptimizer(
            get_param_groups(model, args.lr),
            lr=args.lr,
            weight_decay=0.0,
            ns_iters=5,
        )

    def make_adam(model):
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

    optimizers = [
        ("Gamuon", make_gamuon),
        ("Gamuon+Conf", make_gamuon_conformal),
        ("Muon (NS)", make_muon),
        ("Adam", make_adam),
    ]

    # ── Run benchmarks ─────────────────────────────────────────────
    results = []
    for label, make_opt in optimizers:
        print(f"{'=' * 60}")
        print(f"  Optimizer: {label}")
        print(f"{'=' * 60}")
        model = make_model()
        opt = make_opt(model)
        r = run_training(
            model, opt,
            steps=args.steps,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            label=label,
        )
        results.append(r)
        print()

    # ── Summary ────────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Optimizer':<15} {'Final loss':<12} {'Min loss':<12} "
          f"{'Total time':<12} {'Mean step':<12}")
    print("-" * 63)
    for r in results:
        print(f"{r['label']:<15} {r['loss_final']:<12.4f} {r['loss_min']:<12.4f} "
              f"{r['total_time_s']:<12.3f}s {r['mean_step_time_s']:<12.5f}s")
    print()

    # ── Save results ──────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent
    results_json = results_dir / "benchmark_results.json"
    # Strip step_times from JSON (too verbose)
    json_data = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "step_times"}
        entry["step_times_summary"] = {
            "mean": r["mean_step_time_s"],
            "median": r["median_step_time_s"],
            "min": min(r["step_times"]),
            "max": max(r["step_times"]),
        }
        json_data.append(entry)
    with open(results_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[benchmark] Results saved to {results_json}")

    # ── Plot ───────────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = results_dir / "benchmark_plot.pdf"
        plot_results(results, save_path=plot_path)
    else:
        print("[benchmark] Plotting disabled (--no-plot)")

    print("[benchmark] Done!")


if __name__ == "__main__":
    main()
