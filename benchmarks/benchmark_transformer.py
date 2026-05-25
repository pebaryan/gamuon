"""
Benchmark: Gamuon vs Muon (Newton-Schulz) vs Adam
==================================================

Trains a small transformer language model with each optimizer and compares
convergence speed (loss vs steps), wall-clock time, and final loss.

Usage:
    python benchmarks/benchmark_transformer.py           # synthetic, 200 steps
    python benchmarks/benchmark_transformer.py --steps 1000     # longer run
    python benchmarks/benchmark_transformer.py --no-plot        # no PDF output
    python benchmarks/benchmark_transformer.py --ablations      # add no-rotor / pad-square
    python benchmarks/benchmark_transformer.py --task wikitext  # real text task
                                                                # (wikitext-2-raw-v1
                                                                #  + gpt2 BPE)
"""

from __future__ import annotations

import argparse
import json
import math
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
from gamuon import (Gamuon, GamuonNS, GamuonAuto, grade_decompose,
                       newton_schulz, bivector_exp, rotor_apply,
                       MultivectorMomentum,
                       ConformalMuon, find_conformal_pairs)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SMALL TRANSFORMER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class GroupNormWrapper(nn.GroupNorm):
    """Wrapper that permutes (B, T, D) → (B, D, T) for GroupNorm, then back.

    nn.GroupNorm expects channels-first input (N, C, *spatial), but
    the transformer uses (batch, seq_len, d_model). This wrapper
    inherits from nn.GroupNorm so isinstance checks (used by
    _norm_layers, find_conformal_pairs, etc.) work naturally.
    """

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D, T) for GroupNorm → (B, T, D)
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        return x.permute(0, 2, 1)


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

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 norm_type: str = "layernorm", groupnorm_groups: int = 8):
        super().__init__()
        if norm_type == "groupnorm":
            self.ln1 = GroupNormWrapper(groupnorm_groups, d_model)
            self.ln2 = GroupNormWrapper(groupnorm_groups, d_model)
        elif norm_type == "rmsnorm":
            self.ln1 = nn.RMSNorm(d_model)
            self.ln2 = nn.RMSNorm(d_model)
        else:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, dropout)
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
        norm_type: str = "layernorm",
        groupnorm_groups: int = 8,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, dropout, norm_type, groupnorm_groups)
            for _ in range(n_layers)
        ])
        if norm_type == "groupnorm":
            self.ln_f = GroupNormWrapper(groupnorm_groups, d_model)
        elif norm_type == "rmsnorm":
            self.ln_f = nn.RMSNorm(d_model)
        else:
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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ABLATIONS                                                          ║
# ║  Variants that isolate specific design choices in Gamuon.           ║
# ╚══════════════════════════════════════════════════════════════════════╝


class GamuonPadSquare(Gamuon):
    """Ablation: pre-fix non-square handling that pads to square.

    Subclasses :class:`Gamuon` and overrides only the rectangular path
    with the original pad-to-square algorithm (gradient and weight are
    padded with zeros to ``max(m, n) × max(m, n)``, the square sandwich
    update is applied, then the result is sliced back).  This is
    mathematically incorrect (the rotor mixes the padded zero rows /
    columns into the original block, so the spectrum-preservation
    invariant doesn't hold for the (m, n) slice) — it exists here only
    to compare against the Stiefel-style two-sided update that
    replaced it.

    The square path is inherited unchanged from ``Gamuon``.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr_s = group["lr_scalar"]
            lr_b = group["lr_bivector"]
            lr_p = group["lr_strain"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if g.dim() != 2:
                    raise NotImplementedError(
                        f"GamuonPadSquare: only 2-D params (got {tuple(g.shape)})"
                    )
                if wd != 0:
                    g.add_(p.data, alpha=wd)

                m, n = g.shape
                if m == n:
                    # Square: delegate to inherited square path.
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum"] = MultivectorMomentum(n, g.device, g.dtype)
                    state["step"] += 1
                    t = state["step"]
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    Gamuon._step_square(
                        p, g, n, lr, lr_s, lr_b, lr_p,
                        beta1, beta2, bc1, bc2, eps, state,
                    )
                    continue

                # ── Rectangular: pad to square, run the square update,
                #    slice back.  This is the pre-fix behaviour.
                max_dim = max(m, n)
                padded_g = g.new_zeros(max_dim, max_dim)
                padded_g[:m, :n] = g
                padded_p = p.data.new_zeros(max_dim, max_dim)
                padded_p[:m, :n] = p.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = MultivectorMomentum(
                        max_dim, g.device, g.dtype
                    )
                state["step"] += 1
                t = state["step"]
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t

                # Stand in a shadow Parameter so the square path can mutate
                # `.data` without touching the real parameter.
                shadow = torch.nn.Parameter(padded_p, requires_grad=False)
                Gamuon._step_square(
                    shadow, padded_g, max_dim, lr, lr_s, lr_b, lr_p,
                    beta1, beta2, bc1, bc2, eps, state,
                )
                p.data.copy_(shadow.data[:m, :n])

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

    @property
    def param_groups(self) -> list[dict]:
        """Flat list of every sub-optimizer's param groups, in order.

        Lets a schedule iterate and mutate ``g["lr"]`` uniformly across
        all sub-optimizers (Gamuon / Adam-for-embed / SGD-for-biases /
        ConformalMuon).  Each entry is a live dict that the underlying
        optimizer reads from on each ``step()`` call.
        """
        return [g for opt in self.optimizers for g in opt.param_groups]


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


def load_wikitext_tokens(split: str = "train") -> tuple[torch.Tensor, int]:
    """Tokenize a wikitext-2-raw-v1 split with the gpt2 BPE tokenizer.

    Returns the concatenated token stream as a 1-D long tensor and the
    tokenizer's vocab size.  Both the dataset and the tokenizer are
    expected to already live in the local HF cache.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # datasets 4.x returns a Column object for ds["text"]; the fast
    # tokenizer requires a plain list[str].
    texts = list(ds["text"])
    encs = tok(texts, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    for row in encs:
        if row:
            ids.extend(row)
    return torch.tensor(ids, dtype=torch.long), tok.vocab_size


def make_wikitext_batch_fn(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
):
    """Return a closure that draws random  (x, y)  windows from ``tokens``.

    ``x``  is a  ``(batch_size, seq_len)``  block of token IDs and  ``y``
    is the next-token target (shifted by one), both on ``device``.
    """
    n = tokens.numel() - seq_len - 1
    if n <= 0:
        raise ValueError(
            f"Wikitext stream too short ({tokens.numel()} tokens) for "
            f"seq_len={seq_len}"
        )
    tokens_dev = tokens.to(device)

    def batch_fn() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, n, (batch_size,), device=device)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1, device=device)
        block = tokens_dev[idx]                # (B, seq_len+1)
        x = block[:, :-1].contiguous()
        y = block[:, 1:].contiguous()
        return x, y

    return batch_fn


def load_wikitext_val_batches(
    seq_len: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-build a fixed set of  (x, y)  validation batches from
    wikitext-2-raw-v1 's validation split.

    Windows are placed at evenly-spaced offsets across the val stream so
    every call with the same arguments produces the same batches —
    important for ``val_history`` to be comparable across captures.
    """
    tokens, _ = load_wikitext_tokens(split="validation")
    n = tokens.numel() - seq_len - 1
    if n <= 0:
        raise ValueError(
            f"Wikitext validation stream too short ({tokens.numel()} tokens) "
            f"for seq_len={seq_len}"
        )
    tokens_dev = tokens.to(device)
    total_windows = num_batches * batch_size
    # Evenly spaced deterministic starts
    starts = torch.linspace(0, n - 1, total_windows, device=device).long()
    base = torch.arange(seq_len + 1, device=device)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(num_batches):
        b_starts = starts[b * batch_size : (b + 1) * batch_size]
        idx = b_starts.unsqueeze(1) + base
        block = tokens_dev[idx]                # (B, seq_len+1)
        x = block[:, :-1].contiguous()
        y = block[:, 1:].contiguous()
        batches.append((x, y))
    return batches


def cosine_lr_factor(
    step: int,
    total: int,
    warmup_frac: float = 0.0,
    min_ratio: float = 0.1,
) -> float:
    """Return a multiplier on the peak LR for cosine decay (with optional
    linear warmup).  The multiplier is in [min_ratio, 1.0].

    ``step``      0-indexed current step.
    ``total``     total number of training steps.
    ``warmup_frac`` fraction of ``total`` spent in linear warmup
                  (multiplier rises 1/warmup_steps → 1.0).
    ``min_ratio``  multiplier at the final step.
    """
    if total <= 0:
        return 1.0
    warmup_steps = max(0, int(round(total * warmup_frac)))
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total - warmup_steps)
    decay_step = step - warmup_steps
    cos = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
    return min_ratio + (1.0 - min_ratio) * cos


@torch.no_grad()
def eval_val_loss(
    model: nn.Module,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    ignore_index: int = -100,
) -> float:
    """Mean per-token cross-entropy over a fixed set of val batches.

    Switches the model to ``eval()`` for the pass and restores its
    previous mode afterwards.  Uses ``reduction='sum'`` then divides by
    the count of non-ignored target tokens, so partial batches at the
    tail (if any) don't bias the average.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_batches:
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        if ignore_index == -100:
            valid = y.numel()
        else:
            valid = int((y != ignore_index).sum().item())
        total_loss += loss.item()
        total_tokens += valid
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)


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


def split_params_for_muon(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Split params into  (embed_head, internal_2d, other_1d).

    Muon-style optimizers (Muon, Gamuon, GamuonNS) do dense matrix
    operations whose memory cost is quadratic in the larger weight
    dimension.  For the gpt2 BPE (vocab=50257) the token embedding
    and final classifier exceed the practical limit of
    ``torch.matrix_exp`` and Newton–Schulz alike, so standard practice
    routes those layers to Adam instead.  See Keller Jordan's Muon
    write-up: embeddings and the output head are explicitly excluded.

    The split is structural (by ``nn.Embedding`` and the ``head``
    attribute) rather than by a magic size threshold, so it produces
    the same routing on the synthetic-vocab task as on wikitext.
    """
    embed_head_ids: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            embed_head_ids.update(id(p) for p in mod.parameters())
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        embed_head_ids.update(id(p) for p in model.head.parameters())

    embed_head: list[torch.Tensor] = []
    internal: list[torch.Tensor] = []
    other: list[torch.Tensor] = []
    for p in model.parameters():
        if id(p) in embed_head_ids:
            embed_head.append(p)
        elif p.ndim == 2:
            internal.append(p)
        else:
            other.append(p)
    return embed_head, internal, other


def _norm_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return (name, module) pairs for all norm layers (LayerNorm, RMSNorm, or GroupNorm)."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm)):
            out.append((name, mod))
    return out


def extract_norm_params(model: nn.Module) -> dict[str, tuple[float, float]]:
    """Extract norm layer (γ, β) mean values from a model.

    For RMSNorm, beta is reported as 0.0 (no bias parameter).
    GroupNorm has both gamma and beta (like LayerNorm).
    Returns a dict mapping layer name → (gamma_mean, beta_mean).
    """
    params = {}
    for name, mod in _norm_layers(model):
        gamma = mod.weight.data.mean().item() if mod.weight is not None else 0.0
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)) and mod.bias is not None:
            beta = mod.bias.data.mean().item()
        else:
            beta = 0.0
        params[name] = (gamma, beta)
    return params


def extract_norm_tensors(model: nn.Module) -> dict[str, dict]:
    """Extract full (γ, β) tensors from norm layers, plus their means.

    For RMSNorm, beta is None and beta_mean is 0.0.
    GroupNorm has both gamma and beta (like LayerNorm).
    Returns a dict mapping layer name → {"gamma": tensor, "beta": tensor | None,
    "gamma_mean": float, "beta_mean": float, "norm_type": str}.
    """
    data = {}
    for name, mod in _norm_layers(model):
        g = mod.weight.data.detach().clone() if mod.weight is not None else None
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)) and hasattr(mod, "bias") and mod.bias is not None:
            b = mod.bias.data.detach().clone()
        else:
            b = None
        if isinstance(mod, nn.RMSNorm):
            norm_type = "RMSNorm"
        elif isinstance(mod, nn.GroupNorm):
            norm_type = "GroupNorm"
        else:
            norm_type = "LayerNorm"
        data[name] = {
            "gamma": g,
            "beta": b,
            "gamma_mean": g.mean().item() if g is not None else 0.0,
            "beta_mean": b.mean().item() if b is not None else 0.0,
            "gamma_norm": g.norm().item() if g is not None else 0.0,
            "beta_norm": b.norm().item() if b is not None else 0.0,
            "norm_type": norm_type,
        }
    return data


def run_training(
    model: nn.Module,
    optimizer,
    steps: int,
    vocab_size: int,
    batch_fn,
    device: torch.device,
    label: str,
    *,
    track_norms: bool = False,
    norm_capture_interval: int = 50,
    ignore_index: int = -100,
    val_batches: Optional[list] = None,
    val_capture_interval: Optional[int] = None,
    schedule: Optional[str] = None,
    warmup_frac: float = 0.0,
    min_lr_ratio: float = 0.1,
) -> dict:
    """Run training and return loss history + timing data.

    ``batch_fn``  is a zero-argument callable returning  ``(x, y)``  on
    the target device.  The synthetic and wikitext data sources both
    expose this shape — the training loop itself is task-agnostic.

    If ``val_batches`` is provided, evaluates mean per-token val loss
    over those batches every ``val_capture_interval`` steps (plus once
    at step 0 and once at the end).  The val pass switches the model
    to ``eval()`` and back; gradients are disabled inside ``eval_val_loss``.

    If ``schedule == "cosine"``, applies cosine decay (with optional
    linear warmup of ``warmup_frac × steps``) to every param group's
    ``lr`` field, scaling from the peak LR down to ``min_lr_ratio × peak``.
    Works on any optimizer (or ``CombinedOptimizer``) that exposes
    ``param_groups``.

    If track_norms is True, snapshots of norm-layer (γ, β) mean values
    are recorded every norm_capture_interval steps.
    """
    model.train()
    losses = []
    step_times = []
    norm_trajectories: list[dict] = []
    val_history: list[tuple[int, float]] = []
    total_start = time.perf_counter()

    do_val = val_batches is not None and val_capture_interval is not None

    # Capture peak LR per param group for the scheduler
    peak_lrs: Optional[list[float]] = None
    if schedule == "cosine":
        peak_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]

    # Capture initial norm params if tracking is enabled
    if track_norms:
        norm_trajectories.append({"step": 0, "layers": extract_norm_tensors(model)})
    if do_val:
        val_history.append((0, eval_val_loss(model, val_batches, vocab_size,
                                             ignore_index)))

    for step in range(steps):
        # Apply LR schedule (cosine + optional warmup)
        if peak_lrs is not None:
            factor = cosine_lr_factor(step, steps, warmup_frac, min_lr_ratio)
            for g, peak in zip(optimizer.param_groups, peak_lrs):
                g["lr"] = peak * factor

        x, y = batch_fn()

        step_start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            ignore_index=ignore_index,
        )
        loss.backward()
        optimizer.step()
        step_times.append(time.perf_counter() - step_start)

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            extra = ""
            if do_val and val_history:
                extra = f"  val {val_history[-1][1]:.4f}"
            print(f"  [{label}] step {step + 1:5d}/{steps}  loss {loss.item():.4f}  "
                  f"time/step {step_times[-1]:.4f}s{extra}")

        # Capture norm params at intervals
        if track_norms and (step + 1) % norm_capture_interval == 0:
            norm_trajectories.append({
                "step": step + 1,
                "layers": extract_norm_tensors(model),
            })

        # Capture val loss at intervals
        if do_val and (step + 1) % val_capture_interval == 0:
            val_history.append((step + 1, eval_val_loss(
                model, val_batches, vocab_size, ignore_index,
            )))

    # Final norm capture
    if track_norms and steps % norm_capture_interval != 0:
        norm_trajectories.append({"step": steps, "layers": extract_norm_tensors(model)})
    if do_val and (not val_history or val_history[-1][0] != steps):
        val_history.append((steps, eval_val_loss(
            model, val_batches, vocab_size, ignore_index,
        )))

    total_time = time.perf_counter() - total_start

    val_min = min(v for _, v in val_history) if val_history else None
    val_final = val_history[-1][1] if val_history else None

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
        "norm_trajectories": norm_trajectories if track_norms else [],
        "val_history": val_history,        # list[(step, val_loss)]
        "val_min": val_min,                # None if val tracking off
        "val_final": val_final,            # None if val tracking off
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


def print_norm_comparison(results: list[dict]):
    """Print a comparison table of norm-layer γ/β trajectories.

    Compares the two norm-update strategies used in this benchmark:
    - Gamuon (uses SGD for norm-layer params)
    - Gamuon+Conf (uses ConformalMuon for norm-layer params)

    Prints initial → final values for each norm layer.
    Supports both LayerNorm (γ + β) and RMSNorm (γ only).
    """
    # Find Gamuon and Gamuon+Conf results with norm trajectories
    gamuon_r = next((r for r in results if r["label"] == "Gamuon" and r.get("norm_trajectories")), None)
    conformal_r = next((r for r in results if r["label"] == "Gamuon+Conf" and r.get("norm_trajectories")), None)

    if not gamuon_r or not conformal_r:
        return

    # Get initial (shared) and final states
    init_layers = gamuon_r["norm_trajectories"][0]["layers"]
    sgd_final = gamuon_r["norm_trajectories"][-1]["layers"]
    conf_final = conformal_r["norm_trajectories"][-1]["layers"]

    # Detect norm type from first layer
    first_key = list(init_layers.keys())[0]
    norm_type = init_layers[first_key].get("norm_type", "LayerNorm")
    has_beta = norm_type in ("LayerNorm", "GroupNorm") and init_layers[first_key].get("beta") is not None

    if has_beta:
        header_lbl = f"{norm_type} \u03b3/\u03b2"
        header = (
            f"{'Layer':<26} {'\u03b3 init':<10} {'\u03b3 SGD':<10} {'\u03b3 Conf':<10}"
            f" {'\u03b2 init':<10} {'\u03b2 SGD':<10} {'\u03b2 Conf':<10}"
        )
        sep = "-" * 89
    else:
        header_lbl = f"{norm_type} \u03b3 (no bias)"
        header = (
            f"{'Layer':<26} {'\u03b3 init':<10} {'\u03b3 SGD':<10} {'\u03b3 Conf':<10}"
        )
        sep = "-" * 59

    print(f"{'=' * max(len(header), 60)}")
    print(f"  {header_lbl}  —  ConformalMuon vs SGD")
    print(f"{'=' * max(len(header), 60)}")
    print(header)
    print(sep)

    for layer_name in init_layers:
        init = init_layers[layer_name]
        sgd = sgd_final.get(layer_name, {})
        conf = conf_final.get(layer_name, {})
        if has_beta:
            print(
                f"{layer_name:<26} "
                f"{init['gamma_mean']:<10.4f} {sgd.get('gamma_mean', 0):<10.4f} {conf.get('gamma_mean', 0):<10.4f} "
                f"{init['beta_mean']:<10.4f} {sgd.get('beta_mean', 0):<10.4f} {conf.get('beta_mean', 0):<10.4f}"
            )
        else:
            print(
                f"{layer_name:<26} "
                f"{init['gamma_mean']:<10.4f} {sgd.get('gamma_mean', 0):<10.4f} {conf.get('gamma_mean', 0):<10.4f}"
            )

    # Summary statistics
    print(sep)
    sgd_gamma_norm = sum(sgd_final[ln]["gamma_norm"] for ln in sgd_final)
    conf_gamma_norm = sum(conf_final[ln]["gamma_norm"] for ln in conf_final)
    print(f"{'Total \u03b3 \u2113\u2082':<26} {'':<10}"
          f"{sgd_gamma_norm:<10.4f} {conf_gamma_norm:<10.4f}")
    if has_beta:
        sgd_beta_norm = sum(abs(sgd_final[ln]["beta_mean"]) for ln in sgd_final)
        conf_beta_norm = sum(abs(conf_final[ln]["beta_mean"]) for ln in conf_final)
        print(f"{'Total |\u03b2| mean':<26} {'':<10}"
              f"{sgd_beta_norm:<10.4f} {conf_beta_norm:<10.4f}")
    print()


def plot_norm_trajectories(results: list[dict], save_path: Optional[Path] = None):
    """Generate a separate figure showing norm-layer \u03b3/\u03b2 trajectories.

    Supports LayerNorm (\u03b3 + \u03b2), RMSNorm (\u03b3 only), and
    GroupNorm (\u03b3 + \u03b2 like LayerNorm).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gamuon_r = next((r for r in results if r["label"] == "Gamuon" and r.get("norm_trajectories")), None)
    conformal_r = next((r for r in results if r["label"] == "Gamuon+Conf" and r.get("norm_trajectories")), None)

    if not gamuon_r or not conformal_r:
        return

    traj_gamuon = gamuon_r["norm_trajectories"]
    traj_conf = conformal_r["norm_trajectories"]
    layer_names = list(traj_gamuon[0]["layers"].keys())

    # Detect norm type from first layer
    first_layer = traj_gamuon[0]["layers"][layer_names[0]]
    norm_type = first_layer.get("norm_type", "LayerNorm")
    has_beta = norm_type in ("LayerNorm", "GroupNorm") and first_layer.get("beta") is not None

    n_layers = len(layer_names)
    n_cols = 2 if has_beta else 1
    fig, axes = plt.subplots(n_layers, n_cols, figsize=(14 if has_beta else 8, 2.5 * n_layers))
    # Ensure axes is always 2D for consistent indexing
    if n_layers == 1 and n_cols == 1:
        axes = axes.reshape(1, 1)
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(-1, n_cols)

    colors = {"Gamuon (SGD)": "#4C72B0", "Gamuon+Conf": "#8E44AD"}

    for i, layer_name in enumerate(layer_names):
        gamuon_steps = [t["step"] for t in traj_gamuon]
        conf_steps = [t["step"] for t in traj_conf]

        # Gamma trajectory
        ax = axes[i, 0]
        ax.plot(gamuon_steps,
                [t["layers"][layer_name]["gamma_mean"] for t in traj_gamuon],
                "o-", color=colors["Gamuon (SGD)"], label="SGD", alpha=0.8, markersize=3)
        ax.plot(conf_steps,
                [t["layers"][layer_name]["gamma_mean"] for t in traj_conf],
                "s-", color=colors["Gamuon+Conf"], label="ConformalMuon", alpha=0.8, markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel("\u03b3 mean")
        ax.set_title(f"{layer_name} — \u03b3 trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Beta trajectory (only for LayerNorm)
        if has_beta:
            ax = axes[i, 1]
            ax.plot(gamuon_steps,
                    [t["layers"][layer_name]["beta_mean"] for t in traj_gamuon],
                    "o-", color=colors["Gamuon (SGD)"], label="SGD", alpha=0.8, markersize=3)
            ax.plot(conf_steps,
                    [t["layers"][layer_name]["beta_mean"] for t in traj_conf],
                    "s-", color=colors["Gamuon+Conf"], label="ConformalMuon", alpha=0.8, markersize=3)
            ax.set_xlabel("Step")
            ax.set_ylabel("\u03b2 mean")
            ax.set_title(f"{layer_name} — \u03b2 trajectory")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{norm_type} \u03b3" + ("\u03b2" if has_beta else "") +
        " Trajectories  —  SGD vs ConformalMuon",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        if norm_type == "GroupNorm":
            suffix = "_groupnorm"
        elif norm_type == "RMSNorm":
            suffix = "_rmsnorm"
        else:
            suffix = ""
        norms_plot_path = save_path.with_name(f"benchmark_norm_trajectories{suffix}.pdf")
        fig.savefig(norms_plot_path, dpi=150, bbox_inches="tight")
        print(f"[benchmark] Norm trajectory plot saved to {norms_plot_path}")
    else:
        plt.show()

    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  LR SWEEP                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


def run_lr_sweep(
    optimizers: list,
    lrs: list[float],
    make_model,
    *,
    steps: int,
    vocab_size: int,
    batch_fn,
    device: torch.device,
    ignore_index: int,
    val_batches: Optional[list] = None,
    val_capture_interval: Optional[int] = None,
    schedule: Optional[str] = None,
    warmup_frac: float = 0.0,
    min_lr_ratio: float = 0.1,
) -> dict:
    """Run every optimizer at every LR.  Returns a nested dict
    ``{label: {lr: run_dict}}``.

    Norm-trajectory tracking is disabled (only meaningful at one LR).
    Val tracking is propagated to each sub-run when provided.
    """
    out: dict[str, dict[float, dict]] = {label: {} for label, _ in optimizers}
    total = len(optimizers) * len(lrs)
    i = 0
    for label, make_opt in optimizers:
        for lr in lrs:
            i += 1
            print(f"{'=' * 60}")
            print(f"  [{i}/{total}] {label}  lr={lr:g}")
            print(f"{'=' * 60}")
            model = make_model()
            opt = make_opt(model, lr)
            r = run_training(
                model, opt,
                steps=steps,
                vocab_size=vocab_size,
                batch_fn=batch_fn,
                device=device,
                label=f"{label}@lr={lr:g}",
                track_norms=False,
                ignore_index=ignore_index,
                val_batches=val_batches,
                val_capture_interval=val_capture_interval,
                schedule=schedule,
                warmup_frac=warmup_frac,
                min_lr_ratio=min_lr_ratio,
            )
            r["base_label"] = label
            r["lr"] = lr
            out[label][lr] = r
            print()
    return out


def _has_val(sweep: dict) -> bool:
    for by_lr in sweep.values():
        for r in by_lr.values():
            if r.get("val_min") is not None:
                return True
    return False


def _matrix_table(sweep: dict, lrs: list[float], metric: str, label_w: int) -> None:
    """Print one  optimizer × LR  table for the given metric key.

    Cells are formatted to width 10.  The cell at each row's best LR
    (lowest value) is marked with a trailing '*'.
    """
    header_lrs = "  ".join(f"{lr:>10g}" for lr in lrs)
    print(f"{'Optimizer':<{label_w}} {header_lrs}")
    print("-" * (label_w + 1 + 12 * len(lrs)))
    for label, by_lr in sweep.items():
        # Best LR for this row
        present = {lr: by_lr[lr][metric] for lr in lrs
                   if lr in by_lr and by_lr[lr].get(metric) is not None}
        best_lr = min(present, key=present.get) if present else None
        cells = []
        for lr in lrs:
            r = by_lr.get(lr)
            v = None if r is None else r.get(metric)
            if v is None:
                cells.append(f"{'—':>10}")
            elif lr == best_lr:
                cells.append(f"{v:>9.4f}*")
            else:
                cells.append(f"{v:>10.4f}")
        print(f"{label:<{label_w}} " + "  ".join(cells))


def print_lr_sweep_summary(sweep: dict, lrs: list[float]) -> None:
    """Print  optimizer × LR  table(s) of min loss + best-of summary.

    If val tracking is enabled, prints both train-loss and val-loss
    matrices and ranks the best-of summary by val loss (otherwise by
    train loss).
    """
    label_w = max(15, max(len(label) for label in sweep) + 1)
    has_val = _has_val(sweep)

    print(f"{'=' * 60}")
    print("  LR SWEEP — train min loss per (optimizer, lr)")
    print(f"{'=' * 60}")
    _matrix_table(sweep, lrs, "loss_min", label_w)
    print()
    print("  * = best LR for that optimizer (lowest train min)")
    print()

    if has_val:
        print(f"{'=' * 60}")
        print("  LR SWEEP — val min loss per (optimizer, lr)")
        print(f"{'=' * 60}")
        _matrix_table(sweep, lrs, "val_min", label_w)
        print()
        print("  * = best LR for that optimizer (lowest val min)")
        print()

    print(f"{'=' * 60}")
    rank_metric = "val_min" if has_val else "loss_min"
    print(f"  BEST-OF SUMMARY (ranked by {rank_metric})")
    print(f"{'=' * 60}")
    cols = ["Optimizer", "Best LR", "Val min", "Val final",
            "Train min", "Train final", "Step (ms)"]
    if not has_val:
        cols = ["Optimizer", "Best LR", "Train min",
                "Train final", "Total", "Step (ms)"]
    widths = [label_w, 10] + [11] * (len(cols) - 2)
    print(" ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + len(widths)))

    def _row_metric(r):
        v = r.get(rank_metric)
        return v if v is not None else float("inf")

    ranked = sorted(
        sweep.items(),
        key=lambda kv: min(_row_metric(r) for r in kv[1].values()),
    )
    for label, by_lr in ranked:
        best_lr = min(by_lr, key=lambda l: _row_metric(by_lr[l]))
        r = by_lr[best_lr]
        step_ms = r["mean_step_time_s"] * 1000
        if has_val:
            vmin = r.get("val_min")
            vfin = r.get("val_final")
            cells = [
                f"{label:<{label_w}}",
                f"{best_lr:<10g}",
                f"{vmin:<11.4f}" if vmin is not None else f"{'—':<11}",
                f"{vfin:<11.4f}" if vfin is not None else f"{'—':<11}",
                f"{r['loss_min']:<11.4f}",
                f"{r['loss_final']:<11.4f}",
                f"{step_ms:<11.2f}",
            ]
        else:
            cells = [
                f"{label:<{label_w}}",
                f"{best_lr:<10g}",
                f"{r['loss_min']:<11.4f}",
                f"{r['loss_final']:<11.4f}",
                f"{r['total_time_s']:<10.1f}s",
                f"{step_ms:<11.2f}",
            ]
        print(" ".join(cells))
    print()


def save_lr_sweep_json(sweep: dict, lrs: list[float], path: Path) -> None:
    """Write a compact JSON for the sweep (strip per-step time arrays;
    keep val_history since it's tiny and useful for plotting offline)."""
    json_data: dict = {"lrs": lrs, "runs": {}}
    for label, by_lr in sweep.items():
        json_data["runs"][label] = {}
        for lr, r in by_lr.items():
            entry = {k: v for k, v in r.items()
                     if k not in ("step_times", "norm_trajectories")}
            entry["step_times_summary"] = {
                "mean": r["mean_step_time_s"],
                "median": r["median_step_time_s"],
                "min": min(r["step_times"]),
                "max": max(r["step_times"]),
            }
            json_data["runs"][label][str(lr)] = entry
    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[benchmark] LR sweep results saved to {path}")


def plot_lr_sweep(sweep: dict, lrs: list[float], save_path: Path) -> None:
    """Min loss vs LR for each optimizer (log-x).  If val tracking
    was on, draws train and val on a 1×2 panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[benchmark] matplotlib not installed — skipping plot")
        return

    colors = {
        "Gamuon": "#4C72B0",
        "Gamuon+Conf": "#8E44AD",
        "Muon (NS)": "#DD8452",
        "Adam": "#55A868",
        "Gamuon (no rotor)": "#7FB6E0",
        "Gamuon (pad-square)": "#2A4E7A",
    }
    has_val = _has_val(sweep)
    if has_val:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
        ax_train, ax_val = axes
    else:
        fig, ax_train = plt.subplots(figsize=(8, 5))
        ax_val = None

    def _draw(ax, metric):
        for label, by_lr in sweep.items():
            xs, ys = [], []
            for lr in lrs:
                r = by_lr.get(lr)
                if r is None or r.get(metric) is None:
                    continue
                xs.append(lr)
                ys.append(r[metric])
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2,
                        color=colors.get(label, "gray"), label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    _draw(ax_train, "loss_min")
    ax_train.set_ylabel("Min train loss")
    ax_train.set_title("LR sensitivity — train")

    if ax_val is not None:
        _draw(ax_val, "val_min")
        ax_val.set_ylabel("Min val loss")
        ax_val.set_title("LR sensitivity — val")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] LR sweep plot saved to {save_path}")


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
        # Ablations
        "Gamuon (no rotor)": "#7FB6E0",     # light blue (related to Gamuon)
        "Gamuon (pad-square)": "#2A4E7A",   # dark blue (related to Gamuon)
    }
    markers = {
        "Gamuon": "o",
        "Gamuon+Conf": "D",
        "Muon (NS)": "s",
        "Adam": "^",
        "Gamuon (no rotor)": "o",
        "Gamuon (pad-square)": "o",
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
    parser.add_argument("--rmsnorm", action="store_true",
                        help="Use RMSNorm instead of LayerNorm for all norm layers")
    parser.add_argument("--groupnorm", action="store_true",
                        help="Use GroupNorm instead of LayerNorm for all norm layers")
    parser.add_argument("--groupnorm-groups", type=int, default=8,
                        help="Number of groups for GroupNorm (default: 8)")
    parser.add_argument(
        "--ablations", action="store_true",
        help="Also run ablation variants: 'Gamuon (no rotor)' "
             "(lr_bivector=0) and 'Gamuon (pad-square)' (pre-fix "
             "non-square padding).  Slows the run roughly proportionally.",
    )
    parser.add_argument(
        "--task", choices=("synthetic", "wikitext"), default="synthetic",
        help="Training task.  'synthetic' (default) generates random "
             "next-token sequences from a small vocabulary.  'wikitext' "
             "uses Salesforce/wikitext (wikitext-2-raw-v1) tokenised "
             "with the gpt2 BPE; vocab_size is overridden to 50257 and "
             "the cached dataset/tokeniser are loaded from $HF_HOME.",
    )
    parser.add_argument(
        "--lr-sweep", default=None, type=str,
        help="Comma-separated list of learning rates to sweep "
             "(e.g. '3e-4,1e-3,3e-3,1e-2').  When set, runs every "
             "optimizer at every LR and reports the best-of per "
             "optimizer plus an LR-sensitivity plot.  Disables norm-"
             "trajectory tracking (which is only meaningful at one LR).",
    )
    parser.add_argument(
        "--val-batches", type=int, default=16,
        help="Number of fixed validation batches per capture (wikitext "
             "only).  Each is batch_size × seq_len tokens drawn at "
             "evenly-spaced offsets through the validation stream.  "
             "Set to 0 to disable val tracking.",
    )
    parser.add_argument(
        "--val-every", type=int, default=0,
        help="Steps between validation captures (wikitext only).  "
             "0 → max(1, steps // 20).  Ignored if --val-batches=0.",
    )
    parser.add_argument(
        "--schedule", choices=("none", "cosine"), default="none",
        help="LR schedule.  'none' = constant LR.  'cosine' = optional "
             "linear warmup → cosine decay to min_lr_ratio × peak LR.",
    )
    parser.add_argument(
        "--warmup-frac", type=float, default=0.05,
        help="Fraction of total steps spent in linear warmup when "
             "--schedule=cosine.  Default 0.05 (5%%).",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.1,
        help="Final-LR / peak-LR ratio when --schedule=cosine.  "
             "Default 0.1.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[benchmark] Device: {device}")
    print(f"[benchmark] Transformer: {args.n_layers} layers, "
          f"d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"[benchmark] Task: {args.task}")

    # ── Pick the data source (synthetic vs. wikitext-2) ────────────
    # vocab_size and ignore_index depend on the task.  Synthetic uses
    # token 0 as a pad sentinel; wikitext doesn't, so we disable the
    # ignore behaviour by using -100 (PyTorch's no-ignore default).
    val_batches = None
    val_capture_interval = None
    if args.task == "wikitext":
        print("[benchmark] Loading wikitext-2-raw-v1 (gpt2 BPE)...")
        tokens, vocab_size = load_wikitext_tokens(split="train")
        print(f"[benchmark]   train tokens={tokens.numel():,}, vocab={vocab_size:,}")
        batch_fn = make_wikitext_batch_fn(
            tokens, args.batch_size, args.seq_len, device
        )
        ignore_index = -100
        if args.val_batches > 0:
            val_batches = load_wikitext_val_batches(
                args.seq_len, args.batch_size, args.val_batches, device,
            )
            val_capture_interval = (
                args.val_every if args.val_every > 0
                else max(1, args.steps // 20)
            )
            print(f"[benchmark]   val batches={len(val_batches)}  "
                  f"({len(val_batches) * args.batch_size * args.seq_len:,} tokens "
                  f"per capture, every {val_capture_interval} steps)")
    else:
        vocab_size = args.vocab_size

        def batch_fn():
            return generate_batch(vocab_size, args.batch_size, args.seq_len, device)
        ignore_index = 0  # preserve historical behaviour

    print(f"[benchmark] Data: vocab={vocab_size}, "
          f"batch={args.batch_size}, seq_len={args.seq_len}")
    print(f"[benchmark] Steps: {args.steps}, LR: {args.lr}, Seed: {args.seed}")
    if args.groupnorm:
        norm_type_str = f"GroupNorm({args.groupnorm_groups}, {args.d_model})"
    elif args.rmsnorm:
        norm_type_str = "RMSNorm"
    else:
        norm_type_str = "LayerNorm"
    print(f"[benchmark] Norm type: {norm_type_str}")
    print()

    torch.manual_seed(args.seed)

    # ── Build model (shared across all optimizers) ─────────────────
    # We create a fresh copy for each optimizer to ensure fair comparison
    def make_model():
        if args.groupnorm:
            norm_type = "groupnorm"
        elif args.rmsnorm:
            norm_type = "rmsnorm"
        else:
            norm_type = "layernorm"
        return SmallTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            max_len=args.seq_len + 1,
            norm_type=norm_type,
            groupnorm_groups=args.groupnorm_groups,
        ).to(device)

    # ── Define optimizers ──────────────────────────────────────────
    # Gamuon w/ SGD fallback:   Garnuon for 2D matrix params,
    #                            SGD for biases & norm params.
    # Gamuon+Conf:               Gamuon for matrix params,
    #                            ConformalMuon for norm (γ, β),
    #                            SGD for remaining non-2D params.
    # Muon:                      NS projection for 2D + SGD for rest.
    # Adam:                      Standard baseline (all params).

    def _matrix_flow_combo(model, lr, matrix_opt_ctor):
        """Build  Adam(embed+head) ⊕ <matrix_opt_ctor>(internal 2D) ⊕ SGD(1D).

        ``matrix_opt_ctor``  is a callable taking the list of internal
        2-D params and returning a configured optimizer (Gamuon,
        GamuonPadSquare, MuonOptimizer, …).
        """
        embed_head, internal, other = split_params_for_muon(model)
        subs = []
        if internal:
            subs.append(matrix_opt_ctor(internal))
        if embed_head:
            subs.append(torch.optim.Adam(
                [{"params": embed_head, "lr": lr}],
                lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
            ))
        if other:
            subs.append(torch.optim.SGD(
                [{"params": other, "lr": lr}], lr=lr,
            ))
        return CombinedOptimizer(*subs)

    def make_gamuon(model, lr):
        return _matrix_flow_combo(model, lr, lambda ps: Gamuon(
            [{"params": ps, "lr": lr}],
            lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
        ))

    def make_gamuon_conformal(model, lr):
        """Gamuon for internal 2D matrices + ConformalMuon for norm (γ, β)
        pairs + Adam for the embedding/head + SGD for any remaining 1D
        param.  Constructed manually rather than via GamuonAuto so the
        large vocab-sized layers can be diverted to Adam (Muon-standard
        practice — the dense rotor on (vocab, d_model) is intractable).
        """
        embed_head, internal, other = split_params_for_muon(model)
        pairs = find_conformal_pairs(model)
        pair_ids: set[int] = set()
        for w, b in pairs:
            pair_ids.add(id(w))
            if b is not None:
                pair_ids.add(id(b))
        other_sgd = [p for p in other if id(p) not in pair_ids]

        subs = []
        if pairs:
            subs.append(ConformalMuon(
                pairs, lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
            ))
        if internal:
            subs.append(Gamuon(
                [{"params": internal, "lr": lr}],
                lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
            ))
        if embed_head:
            subs.append(torch.optim.Adam(
                [{"params": embed_head, "lr": lr}],
                lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
            ))
        if other_sgd:
            subs.append(torch.optim.SGD(
                [{"params": other_sgd, "lr": lr}], lr=lr,
            ))
        return CombinedOptimizer(*subs)

    def make_muon(model, lr):
        return _matrix_flow_combo(model, lr, lambda ps: MuonOptimizer(
            [{"params": ps, "lr": lr}],
            lr=lr, weight_decay=0.0, ns_iters=5,
        ))

    def make_adam(model, lr):
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

    def make_gamuon_no_rotor(model, lr):
        """Ablation: Gamuon with lr_bivector=0 (rotor disabled).

        Tests whether the bivector / rotor path actually contributes
        beyond what the scalar + strain (Adam-like) updates do.
        """
        return _matrix_flow_combo(model, lr, lambda ps: Gamuon(
            [{"params": ps, "lr": lr}],
            lr=lr, betas=(0.9, 0.999),
            weight_decay=0.0, lr_bivector=0.0,
        ))

    def make_gamuon_pad_square(model, lr):
        """Ablation: pre-fix non-square handling (pad to square)."""
        return _matrix_flow_combo(model, lr, lambda ps: GamuonPadSquare(
            [{"params": ps, "lr": lr}],
            lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
        ))

    optimizers = [
        ("Gamuon", make_gamuon),
        ("Gamuon+Conf", make_gamuon_conformal),
        ("Muon (NS)", make_muon),
        ("Adam", make_adam),
    ]
    if args.ablations:
        optimizers.extend([
            ("Gamuon (no rotor)", make_gamuon_no_rotor),
            ("Gamuon (pad-square)", make_gamuon_pad_square),
        ])

    # ── Run benchmarks (single LR or sweep) ────────────────────────
    if args.lr_sweep:
        sweep_lrs = [float(s.strip()) for s in args.lr_sweep.split(",") if s.strip()]
        if not sweep_lrs:
            raise ValueError(f"--lr-sweep parsed to empty list: {args.lr_sweep!r}")
        print(f"[benchmark] LR sweep: {sweep_lrs}")
        print()
        sweep_results = run_lr_sweep(
            optimizers, sweep_lrs, make_model,
            steps=args.steps, vocab_size=vocab_size,
            batch_fn=batch_fn, device=device,
            ignore_index=ignore_index,
            val_batches=val_batches,
            val_capture_interval=val_capture_interval,
            schedule=args.schedule if args.schedule != "none" else None,
            warmup_frac=args.warmup_frac,
            min_lr_ratio=args.lr_min_ratio,
        )
        print_lr_sweep_summary(sweep_results, sweep_lrs)
        results_dir = Path(__file__).resolve().parent
        save_lr_sweep_json(sweep_results, sweep_lrs,
                           results_dir / "benchmark_lr_sweep.json")
        if not args.no_plot:
            plot_lr_sweep(sweep_results, sweep_lrs,
                          results_dir / "benchmark_lr_sweep.pdf")
        print("[benchmark] Done!")
        return

    # Single-LR mode (existing behaviour: norm-trajectory tracking etc.)
    results = []
    for label, make_opt in optimizers:
        print(f"{'=' * 60}")
        print(f"  Optimizer: {label}")
        print(f"{'=' * 60}")
        model = make_model()
        opt = make_opt(model, args.lr)

        # Track norm trajectories for the two optimizers that treat
        # norm layers differently (SGD vs ConformalMuon)
        track_norms = label in ("Gamuon", "Gamuon+Conf")

        r = run_training(
            model, opt,
            steps=args.steps,
            vocab_size=vocab_size,
            batch_fn=batch_fn,
            device=device,
            label=label,
            track_norms=track_norms,
            norm_capture_interval=max(1, args.steps // 20),
            ignore_index=ignore_index,
            val_batches=val_batches,
            val_capture_interval=val_capture_interval,
            schedule=args.schedule if args.schedule != "none" else None,
            warmup_frac=args.warmup_frac,
            min_lr_ratio=args.lr_min_ratio,
        )
        results.append(r)
        print()

    # ── Norm comparison table ──────────────────────────────────────
    print_norm_comparison(results)

    # ── Summary ────────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    label_w = max(15, max(len(r["label"]) for r in results) + 1)
    has_val = any(r.get("val_min") is not None for r in results)
    if has_val:
        print(f"{'Optimizer':<{label_w}} {'Train min':<11} {'Train final':<12} "
              f"{'Val min':<10} {'Val final':<10} {'Total':<10} {'Step (ms)':<10}")
        print("-" * (label_w + 11 + 12 + 10 + 10 + 10 + 10 + 6))
        for r in results:
            vmin = r.get("val_min")
            vfin = r.get("val_final")
            vmin_s = f"{vmin:<10.4f}" if vmin is not None else f"{'—':<10}"
            vfin_s = f"{vfin:<10.4f}" if vfin is not None else f"{'—':<10}"
            print(f"{r['label']:<{label_w}} {r['loss_min']:<11.4f} "
                  f"{r['loss_final']:<12.4f} {vmin_s} {vfin_s} "
                  f"{r['total_time_s']:<9.1f}s {r['mean_step_time_s']*1000:<10.2f}")
    else:
        print(f"{'Optimizer':<{label_w}} {'Final loss':<12} {'Min loss':<12} "
              f"{'Total time':<13} {'Mean step':<12}")
        print("-" * (label_w + 12 + 12 + 13 + 12 + 4))
        for r in results:
            print(f"{r['label']:<{label_w}} {r['loss_final']:<12.4f} "
                  f"{r['loss_min']:<12.4f} "
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
        # Strip norm_trajectories from JSON (verbose, contains tensors)
        # and replace with a lightweight summary
        if r.get("norm_trajectories"):
            traj = r["norm_trajectories"]
            del entry["norm_trajectories"]
            if len(traj) >= 2:
                init_data = traj[0]["layers"]
                final_data = traj[-1]["layers"]
                entry["norm_trajectory_summary"] = {
                    "n_captures": len(traj),
                    "initial": {k: {"gamma_mean": v["gamma_mean"], "beta_mean": v["beta_mean"]}
                                for k, v in init_data.items()},
                    "final": {k: {"gamma_mean": v["gamma_mean"], "beta_mean": v["beta_mean"]}
                              for k, v in final_data.items()},
                }
        json_data.append(entry)
    with open(results_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[benchmark] Results saved to {results_json}")

    # ── Plot ───────────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = results_dir / "benchmark_plot.pdf"
        plot_results(results, save_path=plot_path)
        plot_norm_trajectories(results, save_path=plot_path)
    else:
        print("[benchmark] Plotting disabled (--no-plot)")

    print("[benchmark] Done!")


if __name__ == "__main__":
    main()
