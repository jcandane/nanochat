"""experiments/induction/dac.py
Mechanistic DAC induction-current probe for nanochat models.

This module measures the antisymmetric Q/K score sector on the task-specific
induction edges used by the repeated-random ``[s ; s]`` benchmark.

For query row ``i`` and key column ``j``:

    A_ij = 0.5 * (q_i · k_j - k_i · q_j) / sqrt(head_dim)

where q and k are reconstructed exactly at the point seen by the attention
operator: after RoPE, QK normalization, and nanochat's 1.2x scaling.

The collector is intentionally independent of checkpoint loading, CLI, Modal,
Hugging Face, and result-file writing. It is designed to be active while
``experiments.induction.behavior.evaluate_induction_behavior`` performs the
model forwards, so behavioral and mechanistic measurements share the same
synthetic examples and require only one inference pass.

Under the current operator-square convention, the active flux coefficient is:

    DMAP      (beta, alpha) = (0, 0) -> 0
    HMAP      (beta, alpha) = (1, 0) -> 0
    AMAP      (beta, alpha) = (0, 1) -> 1
    attention (beta, alpha) = (1, 1) -> 1

Do not reconstruct that mapping here. Pass the canonical coefficient from
``experiments.common.arms.ArmSpec.active_flux_coefficient``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from nanochat.gpt import apply_rotary_emb, norm


DEFAULT_OFFSETS = (-2, -1, 0, 1, 2)


def _validate_offsets(offsets: Sequence[int]) -> tuple[int, ...]:
    values = tuple(offsets)
    if not values:
        raise ValueError("offsets must contain at least one integer")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
        raise ValueError(f"offsets must be integers, got {values!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"offsets must not contain duplicates, got {values!r}")
    return values


def _json_tensor(x: torch.Tensor) -> Any:
    """Convert a CPU tensor to nested JSON-safe Python values.

    Non-finite floating values become ``None`` so the canonical results writer
    can keep strict JSON (``allow_nan=False``).
    """
    if x.ndim == 0:
        value = float(x.item())
        return value if torch.isfinite(x).item() else None

    if x.ndim == 1:
        return [_json_tensor(v) for v in x]
    return [_json_tensor(v) for v in x]


class DACInductionCollector:
    """Collect layer/head-resolved DAC current on induction edges.

    Parameters
    ----------
    model:
        nanochat ``GPT``-like model. Each transformer block must expose
        ``block.attn.c_q`` and ``block.attn.c_k`` and the model must expose
        rotary buffers ``cos`` and ``sin``.
    offsets:
        Local source offsets around the true induction source. ``0`` measures
        the task edge itself; nearby offsets provide a positional control.
    active_flux_coefficient:
        Coefficient multiplying the antisymmetric flux sector in the *evaluation
        operator*. Obtain this from the canonical ``ArmSpec`` rather than
        recomputing it from alpha/beta here.

    Notes
    -----
    The operator family requires paired Q/K heads for this measurement:
    ``n_kv_head == n_head``.
    """

    def __init__(
        self,
        model: Any,
        *,
        offsets: Sequence[int] = DEFAULT_OFFSETS,
        active_flux_coefficient: float,
    ) -> None:
        self.model = model
        self.offsets = _validate_offsets(offsets)
        self.active_flux_coefficient = float(active_flux_coefficient)

        config = getattr(model, "config", None)
        if config is None:
            raise TypeError("model must expose a .config attribute")

        self.n_layer = int(config.n_layer)
        self.n_head = int(config.n_head)
        self.n_kv_head = int(config.n_kv_head)
        self.n_embd = int(config.n_embd)

        if self.n_kv_head != self.n_head:
            raise RuntimeError(
                "DAC extraction requires n_kv_head == n_head so Q/K heads pair "
                f"one-to-one; got n_head={self.n_head}, "
                f"n_kv_head={self.n_kv_head}"
            )
        if self.n_embd % self.n_head != 0:
            raise RuntimeError(
                f"n_embd={self.n_embd} is not divisible by n_head={self.n_head}"
            )

        self.head_dim = self.n_embd // self.n_head
        self.current_seq_len: int | None = None
        self._closed = False

        self._handles: list[Any] = []
        self._pending_q: dict[int, torch.Tensor] = {}
        self._reset_stats()

        blocks = getattr(getattr(model, "transformer", None), "h", None)
        if blocks is None or len(blocks) != self.n_layer:
            raise TypeError(
                "model must expose transformer.h with one block per config.n_layer"
            )

        for layer_idx, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "c_q") or not hasattr(attn, "c_k"):
                self.close()
                raise TypeError(
                    f"layer {layer_idx} attention must expose c_q and c_k modules"
                )
            self._handles.append(
                attn.c_q.register_forward_hook(self._make_q_hook(layer_idx))
            )
            self._handles.append(
                attn.c_k.register_forward_hook(self._make_k_hook(layer_idx))
            )

    def __enter__(self) -> "DACInductionCollector":
        if self._closed:
            raise RuntimeError("cannot re-enter a closed DACInductionCollector")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _reset_stats(self) -> None:
        nl, nh, no = self.n_layer, self.n_head, len(self.offsets)

        self.sum = torch.zeros(nl, nh, dtype=torch.float64)
        self.abs_sum = torch.zeros(nl, nh, dtype=torch.float64)
        self.sq_sum = torch.zeros(nl, nh, dtype=torch.float64)
        self.count = torch.zeros(nl, dtype=torch.float64)

        self.offset_sum = torch.zeros(nl, nh, no, dtype=torch.float64)
        self.offset_abs_sum = torch.zeros(nl, nh, no, dtype=torch.float64)
        self.offset_count = torch.zeros(nl, no, dtype=torch.float64)

        self._pending_q.clear()

    def begin(self, seq_len: int) -> None:
        """Reset statistics and begin collecting one half-length ``L``."""
        if self._closed:
            raise RuntimeError("collector is closed")
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 2:
            raise ValueError(f"seq_len must be an integer >= 2, got {seq_len!r}")

        self.current_seq_len = seq_len
        self._reset_stats()

    def _make_q_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.current_seq_len is None:
                return
            if layer_idx in self._pending_q:
                raise RuntimeError(
                    f"layer {layer_idx}: c_q hook fired twice before c_k hook"
                )
            self._pending_q[layer_idx] = output

        return hook

    @staticmethod
    def _pair_A(
        qh: torch.Tensor,
        kh: torch.Tensor,
        t_idx: torch.Tensor,
        j_idx: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Compute A_ij on selected oriented edges.

        ``qh`` and ``kh`` have shape ``(B,H,T,D)``. Multiplication remains in
        the model activation dtype while reductions accumulate in fp32.
        """
        qt = qh[:, :, t_idx, :]
        kj = kh[:, :, j_idx, :]
        kt = kh[:, :, t_idx, :]
        qj = qh[:, :, j_idx, :]

        qk = torch.sum(qt * kj, dim=-1, dtype=torch.float32)
        kq = torch.sum(kt * qj, dim=-1, dtype=torch.float32)
        return 0.5 * (qk - kq) * scale

    def _make_k_hook(self, layer_idx: int):
        def hook(_module, _inputs, k_raw):
            if self.current_seq_len is None:
                return

            q_raw = self._pending_q.pop(layer_idx, None)
            if q_raw is None:
                raise RuntimeError(
                    f"layer {layer_idx}: c_k hook fired without matching c_q output"
                )

            if not isinstance(q_raw, torch.Tensor) or not isinstance(k_raw, torch.Tensor):
                raise TypeError("c_q and c_k forward hooks must receive Tensor outputs")
            if q_raw.shape != k_raw.shape:
                raise RuntimeError(
                    f"layer {layer_idx}: q/k projection shapes differ: "
                    f"{tuple(q_raw.shape)} vs {tuple(k_raw.shape)}"
                )
            if q_raw.ndim != 3:
                raise RuntimeError(
                    f"layer {layer_idx}: expected q/k shape (B,T,C), "
                    f"got {tuple(q_raw.shape)}"
                )

            B, T, C = q_raw.shape
            L = self.current_seq_len
            expected_T = 2 * L - 1

            if T != expected_T:
                raise RuntimeError(
                    f"layer {layer_idx}: expected induction input T={expected_T} "
                    f"for L={L}, got T={T}"
                )
            if C != self.n_embd:
                raise RuntimeError(
                    f"layer {layer_idx}: q/k width {C} != n_embd={self.n_embd}"
                )

            H, D = self.n_head, self.head_dim
            q = q_raw.view(B, T, H, D)
            k = k_raw.view(B, T, H, D)

            cos = self.model.cos[:, :T]
            sin = self.model.sin[:, :T]

            # Reconstruct exactly the q/k vectors used by nanochat attention.
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = norm(q) * 1.2
            k = norm(k) * 1.2

            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            scale = D ** -0.5

            # nanochat induction benchmark:
            # query rows t=L,...,2L-2
            # true source j_ind=t-L+1=1,...,L-1
            t = torch.arange(L, 2 * L - 1, device=q.device)
            j = t - L + 1

            a = self._pair_A(qh, kh, t, j, scale)

            self.sum[layer_idx] += a.sum(dim=(0, 2)).double().cpu()
            self.abs_sum[layer_idx] += a.abs().sum(dim=(0, 2)).double().cpu()
            self.sq_sum[layer_idx] += a.square().sum(dim=(0, 2)).double().cpu()
            self.count[layer_idx] += B * a.shape[-1]

            # Positional controls around the true source, restricted to the
            # first copy s[0:L].
            for offset_idx, delta in enumerate(self.offsets):
                jj = j + delta
                valid = (jj >= 0) & (jj < L)
                if not bool(valid.any()):
                    continue

                aa = self._pair_A(qh, kh, t[valid], jj[valid], scale)
                self.offset_sum[layer_idx, :, offset_idx] += (
                    aa.sum(dim=(0, 2)).double().cpu()
                )
                self.offset_abs_sum[layer_idx, :, offset_idx] += (
                    aa.abs().sum(dim=(0, 2)).double().cpu()
                )
                self.offset_count[layer_idx, offset_idx] += B * aa.shape[-1]

        return hook

    def finish(self) -> dict[str, Any]:
        """Finalize the current half-length and return JSON-safe statistics."""
        if self._closed:
            raise RuntimeError("collector is closed")
        if self.current_seq_len is None:
            raise RuntimeError("call begin(seq_len) before finish()")
        if self._pending_q:
            raise RuntimeError(
                "unmatched q hooks remain for layers "
                f"{sorted(self._pending_q)}"
            )
        if bool((self.count == 0).any()):
            missing = torch.nonzero(self.count == 0).flatten().tolist()
            raise RuntimeError(
                f"no DAC edges were observed for layer(s) {missing}; "
                "did the model perform a forward pass after begin()?"
            )

        count = self.count.unsqueeze(1)
        raw_mean = self.sum / count
        raw_abs_mean = self.abs_sum / count
        raw_rms = torch.sqrt(self.sq_sum / count)

        # Offset counts can be zero for extreme offsets. Keep those cells
        # undefined rather than dividing by zero and leaking NaN into JSON.
        offset_denom = self.offset_count.unsqueeze(1)
        valid_offsets = offset_denom > 0

        offset_mean = torch.full_like(self.offset_sum, float("nan"))
        offset_abs_mean = torch.full_like(self.offset_abs_sum, float("nan"))

        expanded_valid = valid_offsets.expand_as(self.offset_sum)
        expanded_denom = offset_denom.expand_as(self.offset_sum)
        offset_mean[expanded_valid] = (
            self.offset_sum[expanded_valid] / expanded_denom[expanded_valid]
        )
        offset_abs_mean[expanded_valid] = (
            self.offset_abs_sum[expanded_valid] / expanded_denom[expanded_valid]
        )

        # Local contrast = true induction edge minus immediate neighbors.
        # It is only defined when offsets include 0 and at least one of -1/+1.
        local_contrast = torch.full_like(raw_mean, float("nan"))
        if 0 in self.offsets:
            i0 = self.offsets.index(0)
            neighbors = [
                offset_mean[:, :, self.offsets.index(d)]
                for d in (-1, 1)
                if d in self.offsets
            ]
            if neighbors:
                neighbor_mean = torch.stack(neighbors, dim=0).mean(dim=0)
                local_contrast = offset_mean[:, :, i0] - neighbor_mean

        c = self.active_flux_coefficient
        result = {
            "definition": (
                "A_ij = 0.5*(q_i.k_j - k_i.q_j)/sqrt(head_dim)"
            ),
            "orientation": (
                "query row i -> key column j; measured pre-causal-mask as "
                "the underlying score kernel"
            ),
            "seq_len": self.current_seq_len,
            "active_flux_coefficient": c,
            "raw_mean": _json_tensor(raw_mean),
            "raw_abs_mean": _json_tensor(raw_abs_mean),
            "raw_rms": _json_tensor(raw_rms),
            "active_mean": _json_tensor(c * raw_mean),
            "active_abs_mean": _json_tensor(abs(c) * raw_abs_mean),
            "local_contrast_raw": _json_tensor(local_contrast),
            "local_contrast_active": _json_tensor(c * local_contrast),
            "offsets": list(self.offsets),
            "offset_raw_mean": _json_tensor(offset_mean),
            "offset_raw_abs_mean": _json_tensor(offset_abs_mean),
            "edge_count_per_layer": _json_tensor(self.count),
            "offset_edge_count_per_layer": _json_tensor(self.offset_count),
        }

        self.current_seq_len = None
        return result

    def close(self) -> None:
        """Remove all hooks. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_q.clear()
        self.current_seq_len = None
        self._closed = True


def _matrix_from_optional_nested(values: Any) -> torch.Tensor:
    """Convert nested float/None data back to a float tensor with NaNs."""
    def replace_none(x):
        if isinstance(x, list):
            return [replace_none(v) for v in x]
        return float("nan") if x is None else x

    return torch.tensor(replace_none(values), dtype=torch.float32)


def top_heads(stats: dict[str, Any], *, k: int = 8) -> list[dict[str, Any]]:
    """Rank layer/head pairs by induction-specific local contrast.

    If local contrast is unavailable, falls back to absolute raw mean current.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    contrast = _matrix_from_optional_nested(stats["local_contrast_raw"])
    raw = _matrix_from_optional_nested(stats["raw_mean"])

    ranking = contrast
    if torch.isnan(ranking).all():
        ranking = raw

    score = torch.nan_to_num(
        ranking.abs(),
        nan=float("-inf"),
        posinf=float("-inf"),
        neginf=float("-inf"),
    )

    finite_count = int(torch.isfinite(score).sum().item())
    if finite_count == 0:
        return []

    n = min(k, finite_count)
    values, indices = torch.topk(score.flatten(), n)
    H = score.shape[1]

    out: list[dict[str, Any]] = []
    for value, flat in zip(values.tolist(), indices.tolist()):
        layer, head = divmod(flat, H)

        raw_value = raw[layer, head]
        contrast_value = contrast[layer, head]

        out.append(
            {
                "layer": layer,
                "head": head,
                "abs_rank_score": float(value),
                "raw_mean": (
                    float(raw_value) if torch.isfinite(raw_value) else None
                ),
                "local_contrast_raw": (
                    float(contrast_value)
                    if torch.isfinite(contrast_value)
                    else None
                ),
            }
        )

    return out
