"""experiments/induction/behavior.py
Behavioral induction benchmark for nanochat models.

This module contains only the scientific core of the repeated-random induction
benchmark. It has no CLI, checkpoint, Hugging Face, Modal, or JSON-writing
responsibilities.

For a random token sequence ``s`` of half-length ``L``, the model receives the
autoregressive input induced by::

    [s ; s]

The benchmark follows nanochat's existing induction-score convention:

* ``random_half_loss`` scores target positions before the repeated copy,
* ``induction_loss`` scores the repeated-copy positions where induction can help,
* ``induction_acc`` is next-token accuracy on those induction positions.

Logical batches are sampled in one CPU ``torch.randint`` call and are
microbatched only *after* sampling. Therefore changing the physical microbatch
size leaves the synthetic examples, RNG ordering, and metric weighting
unchanged (apart from harmless kernel-level numerical roundoff).

The default forward-token budget mirrors the memory-safe setting used in the
earlier probe: ``micro_batch_size * (2*L - 1) <= 8192`` when possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_MAX_FORWARD_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class InductionBehaviorResult:
    """Metrics and resolved evaluation settings for one half-length ``L``."""

    random_half_loss: float
    induction_loss: float
    induction_acc: float

    seq_len: int
    input_length: int
    logical_batch_size: int
    micro_batch_size: int
    num_batches: int
    seed: int

    num_examples: int
    random_token_count: int
    induction_token_count: int

    def metrics_dict(self) -> dict[str, float]:
        """Return only the three scientific headline metrics."""
        return {
            "random_half_loss": self.random_half_loss,
            "induction_loss": self.induction_loss,
            "induction_acc": self.induction_acc,
        }

    def as_dict(self) -> dict[str, int | float]:
        """Return a JSON-friendly result including resolved evaluation settings."""
        return {
            **self.metrics_dict(),
            "seq_len": self.seq_len,
            "input_length": self.input_length,
            "logical_batch_size": self.logical_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "num_batches": self.num_batches,
            "seed": self.seed,
            "num_examples": self.num_examples,
            "random_token_count": self.random_token_count,
            "induction_token_count": self.induction_token_count,
        }


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _valid_seq_len(value: int) -> int:
    L = _positive_int("seq_len", value)
    if L < 2:
        raise ValueError(
            "seq_len must be at least 2 so both score regions are non-empty"
        )
    return L


def auto_micro_batch_size(
    logical_batch_size: int,
    seq_len: int,
    *,
    max_forward_tokens: int = DEFAULT_MAX_FORWARD_TOKENS,
) -> int:
    """Choose a physical batch size from a simple ``B*T`` forward budget.

    The model input length for repeated-random induction is ``T = 2*L - 1``.
    At least one example is always returned, even when a single sequence exceeds
    ``max_forward_tokens``; the budget is a heuristic rather than a hard cap.
    """
    B = _positive_int("logical_batch_size", logical_batch_size)
    L = _valid_seq_len(seq_len)
    budget = _positive_int("max_forward_tokens", max_forward_tokens)

    input_length = 2 * L - 1
    return max(1, min(B, budget // input_length))


def _resolve_device(model: Any, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    get_device = getattr(model, "get_device", None)
    if callable(get_device):
        try:
            return torch.device(get_device())
        except (RuntimeError, TypeError, ValueError):
            pass

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError(
            "could not infer model device; pass device= explicitly"
        ) from exc


def _validate_context_length(model: Any, input_length: int) -> None:
    config = getattr(model, "config", None)
    context_len = getattr(config, "sequence_len", None)
    if context_len is None:
        return
    if input_length > int(context_len):
        raise ValueError(
            f"induction input length {input_length} exceeds model context "
            f"sequence_len={context_len}"
        )


def _validate_logits(
    logits: Any,
    *,
    batch_size: int,
    input_length: int,
    vocab_size: int,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"model(x) must return a logits Tensor, got {type(logits).__name__}"
        )
    if logits.ndim != 3:
        raise ValueError(
            f"model logits must have shape (B,T,V), got {tuple(logits.shape)}"
        )
    expected_prefix = (batch_size, input_length)
    if tuple(logits.shape[:2]) != expected_prefix:
        raise ValueError(
            "model logits have wrong batch/sequence shape: "
            f"got {tuple(logits.shape[:2])}, expected {expected_prefix}"
        )
    if logits.shape[-1] < vocab_size:
        raise ValueError(
            f"model logits vocab dimension {logits.shape[-1]} is smaller than "
            f"requested vocab_size={vocab_size}"
        )
    return logits


@torch.inference_mode()
def evaluate_induction_behavior(
    model: Any,
    vocab_size: int,
    *,
    seq_len: int = 256,
    logical_batch_size: int = 16,
    num_batches: int = 4,
    seed: int = 1234,
    micro_batch_size: int | None = None,
    max_forward_tokens: int = DEFAULT_MAX_FORWARD_TOKENS,
    device: str | torch.device | None = None,
) -> InductionBehaviorResult:
    """Evaluate repeated-random induction behavior at one half-length.

    Parameters
    ----------
    model:
        Autoregressive model returning logits with shape ``(B,T,V)``.
    vocab_size:
        Integer range used to sample synthetic tokens: ``[0, vocab_size)``.
    seq_len:
        Half-length ``L`` of the repeated sequence ``[s ; s]``.
    logical_batch_size:
        Number of synthetic examples generated per RNG draw. This controls the
        benchmark's logical sample set and should remain fixed across models.
    num_batches:
        Number of logical batches.
    seed:
        Seed for the CPU ``torch.Generator``.
    micro_batch_size:
        Physical forward batch size. ``None`` selects it automatically from
        ``max_forward_tokens``. Explicit values are clamped to the logical batch
        size but otherwise respected.
    max_forward_tokens:
        Heuristic budget used only when ``micro_batch_size is None``.
    device:
        Forward device. Inferred from ``model`` when omitted.

    Notes
    -----
    The full logical ``(B,L)`` random tensor is sampled before microbatching.
    This is essential: sampling separately per microbatch would change the RNG
    stream and therefore change the benchmark itself.
    """
    V = _positive_int("vocab_size", vocab_size)
    L = _valid_seq_len(seq_len)
    B = _positive_int("logical_batch_size", logical_batch_size)
    batches = _positive_int("num_batches", num_batches)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")

    if micro_batch_size is None:
        mb = auto_micro_batch_size(
            B,
            L,
            max_forward_tokens=max_forward_tokens,
        )
    else:
        requested_mb = _positive_int("micro_batch_size", micro_batch_size)
        mb = min(B, requested_mb)

    input_length = 2 * L - 1
    _validate_context_length(model, input_length)
    target_device = _resolve_device(model, device)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    random_loss_sum = 0.0
    random_count = 0
    induction_loss_sum = 0.0
    induction_count = 0
    induction_correct = 0

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    try:
        for _ in range(batches):
            # One logical RNG draw, matching the original nanochat benchmark.
            # Physical microbatching happens only after this tensor exists.
            s_cpu = torch.randint(
                0,
                V,
                (B, L),
                generator=generator,
                device="cpu",
            )

            for b0 in range(0, B, mb):
                s = s_cpu[b0 : b0 + mb].to(target_device)
                seq = torch.cat((s, s), dim=1)
                x, y = seq[:, :-1], seq[:, 1:]

                logits = _validate_logits(
                    model(x),
                    batch_size=s.shape[0],
                    input_length=input_length,
                    vocab_size=V,
                )
                losses = F.cross_entropy(
                    logits.transpose(1, 2).float(),
                    y,
                    reduction="none",
                )
                preds = logits.argmax(dim=-1)

                # Exact nanochat induction-score slices.
                random_region = losses[:, : L - 1]
                induction_region = losses[:, L:]

                random_loss_sum += random_region.double().sum().item()
                random_count += random_region.numel()

                induction_loss_sum += induction_region.double().sum().item()
                induction_count += induction_region.numel()
                induction_correct += (
                    preds[:, L:] == y[:, L:]
                ).sum().item()
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    if random_count == 0 or induction_count == 0:
        raise RuntimeError(
            "induction benchmark produced an empty score region; this indicates "
            "an internal validation bug"
        )

    return InductionBehaviorResult(
        random_half_loss=random_loss_sum / random_count,
        induction_loss=induction_loss_sum / induction_count,
        induction_acc=induction_correct / induction_count,
        seq_len=L,
        input_length=input_length,
        logical_batch_size=B,
        micro_batch_size=mb,
        num_batches=batches,
        seed=seed,
        num_examples=B * batches,
        random_token_count=random_count,
        induction_token_count=induction_count,
    )
