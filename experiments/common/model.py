"""experiments/common/model.py
Canonical model construction and strict checkpoint loading for experiments.

This module is the bridge between experiment checkpoint/provenance metadata and
an instantiated :class:`nanochat.gpt.GPT`. It deliberately has no Hugging Face,
Modal, or network responsibilities: callers provide a local mirror root and a
:class:`~experiments.common.checkpoints.CheckpointRef`.

The central distinction is between the operator that *trained the weights* and
the operator used to *evaluate the same weights*::

    loaded = load_model(
        root="/cache/shared-repo",
        checkpoint=CheckpointRef("attention", "run-0001", 10_000),
        evaluation_arm="amap",
        device="cuda",
    )

Here the checkpoint identity remains ``attention/run-0001/...`` while the
forward pass runs under AMAP. This is an instantaneous fixed-weight operator
graft, not an AMAP-trained checkpoint.

Model architecture is reconstructed from ``run.json``. Canonical runs should
store the exact ``GPTConfig`` dictionary under ``model.model_config``; for
convenience this loader also accepts ``model.config`` or a direct mapping of
GPTConfig fields. The stored training operator is validated before any optional
evaluation-arm override is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from nanochat.gpt import GPT, GPTConfig

from experiments.common.arms import ArmSpec, get_arm
from experiments.common.checkpoints import (
    CheckpointFormatError,
    CheckpointRef,
    LocalCheckpoint,
    StateDictValidation,
    load_model_weights,
    read_json,
    resolve_local_checkpoint,
)
from experiments.common.provenance import (
    RunProvenance,
    load_local_provenance,
    validate_checkpoint_provenance,
)


# These fields determine parameter/buffer shapes or attention semantics and must
# be present in canonical run metadata. Operator coordinates are validated
# separately and then overridden only for fixed-weight evaluation grafts.
_REQUIRED_MODEL_CONFIG_FIELDS = (
    "sequence_len",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_kv_head",
    "n_embd",
    "window_pattern",
)


class ExperimentModelError(RuntimeError):
    """Base error for experiment model construction/loading."""


class ModelConfigError(ExperimentModelError):
    """Raised when run metadata cannot reconstruct the trained GPTConfig."""


@dataclass(slots=True)
class LoadedModel:
    """A concrete GPT plus the identities needed to interpret its forward pass."""

    model: GPT
    local_checkpoint: LocalCheckpoint
    run: RunProvenance
    checkpoint_metadata: dict[str, Any]
    trained_arm: ArmSpec
    evaluation_arm: ArmSpec
    model_config: GPTConfig
    state_dict_validation: StateDictValidation

    @property
    def checkpoint(self) -> CheckpointRef:
        return self.local_checkpoint.ref

    @property
    def is_operator_graft(self) -> bool:
        """Whether weights are being evaluated under a different operator."""
        return self.trained_arm.name != self.evaluation_arm.name

    @property
    def device(self) -> Any:
        """Return the model device without imposing a torch type at import time."""
        if hasattr(self.model, "get_device"):
            return self.model.get_device()
        return next(self.model.parameters()).device

    def as_dict(self) -> dict[str, Any]:
        """Compact JSON-friendly identity for experiment result envelopes."""
        return {
            "checkpoint": self.checkpoint.as_dict(),
            "trained_arm": self.trained_arm.as_dict(),
            "evaluation_arm": self.evaluation_arm.as_dict(),
            "is_operator_graft": self.is_operator_graft,
            "model_config": {
                field.name: getattr(self.model_config, field.name)
                for field in fields(self.model_config)
            },
        }


def _config_field_names() -> set[str]:
    return {field.name for field in fields(GPTConfig)}


def _extract_model_config_mapping(run: RunProvenance) -> dict[str, Any]:
    """Extract the persisted GPTConfig mapping from ``run.json``.

    Preferred canonical form::

        "model": {
            "model_config": { ... exact GPTConfig fields ... }
        }

    ``model.config`` is accepted as a short alias, and a direct mapping is
    accepted for early experiment metadata where the model object itself is the
    config. Unknown keys inside an explicit nested config are rejected so typos
    do not silently fall back to GPTConfig defaults.
    """
    model_meta = run.model
    known = _config_field_names()

    for key in ("model_config", "config"):
        if key in model_meta:
            raw = model_meta[key]
            if not isinstance(raw, Mapping):
                raise ModelConfigError(f"run.json model.{key} must be an object")
            unknown = sorted(set(raw) - known)
            if unknown:
                raise ModelConfigError(
                    f"run.json model.{key} contains unknown GPTConfig fields: "
                    + ", ".join(unknown)
                )
            return dict(raw)

    # Early/canonical-simple form: model itself contains GPTConfig fields plus
    # optional descriptive metadata. Only known fields are extracted here.
    direct = {key: value for key, value in model_meta.items() if key in known}
    if direct:
        return direct

    raise ModelConfigError(
        "run.json does not contain a GPTConfig; expected model.model_config, "
        "model.config, or GPTConfig fields directly under model"
    )


def _validate_required_config_fields(config: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED_MODEL_CONFIG_FIELDS if key not in config]
    if missing:
        raise ModelConfigError(
            "run.json model config is missing required fields: " + ", ".join(missing)
        )


def _validate_basic_config(config: GPTConfig) -> None:
    if config.sequence_len <= 0:
        raise ModelConfigError("GPTConfig.sequence_len must be positive")
    if config.vocab_size <= 0:
        raise ModelConfigError("GPTConfig.vocab_size must be positive")
    if config.n_layer <= 0:
        raise ModelConfigError("GPTConfig.n_layer must be positive")
    if config.n_head <= 0 or config.n_kv_head <= 0:
        raise ModelConfigError("GPTConfig n_head and n_kv_head must be positive")
    if config.n_embd <= 0 or config.n_embd % config.n_head != 0:
        raise ModelConfigError(
            "GPTConfig.n_embd must be positive and divisible by n_head"
        )
    if not config.window_pattern:
        raise ModelConfigError("GPTConfig.window_pattern must be non-empty")


def _validate_trained_operator(config: GPTConfig, trained_arm: ArmSpec) -> None:
    """Require persisted model config to agree with canonical run provenance."""
    expected = trained_arm.config_overrides()
    actual = {
        "attn_variant": config.attn_variant,
        "hmap_beta": float(config.hmap_beta),
        "hmap_alpha": float(config.hmap_alpha),
    }
    mismatches = [
        f"{key}: stored={actual[key]!r}, expected={value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise ModelConfigError(
            "stored GPTConfig operator does not match run arm "
            f"{trained_arm.name!r}: " + "; ".join(mismatches)
        )


def _validate_evaluation_compatibility(config: GPTConfig, arm: ArmSpec) -> None:
    """Catch architecture/operator combinations the GPT implementation forbids."""
    if arm.attn_variant == "hmap" and config.n_kv_head != config.n_head:
        raise ModelConfigError(
            f"cannot evaluate {arm.name} with n_kv_head={config.n_kv_head} and "
            f"n_head={config.n_head}: the eager operator family requires "
            "n_kv_head == n_head for per-head q/k pairing"
        )


def trained_model_config(run: RunProvenance) -> GPTConfig:
    """Reconstruct and validate the GPTConfig under which a run was trained."""
    payload = _extract_model_config_mapping(run)
    _validate_required_config_fields(payload)
    try:
        config = GPTConfig(**payload)
    except TypeError as exc:
        raise ModelConfigError(f"invalid GPTConfig in run.json: {exc}") from exc

    _validate_basic_config(config)
    _validate_trained_operator(config, get_arm(str(run.arm)))
    return config


def evaluation_model_config(
    run: RunProvenance,
    evaluation_arm: str | ArmSpec | None = None,
) -> tuple[GPTConfig, ArmSpec, ArmSpec]:
    """Return config plus trained/evaluation arm identities.

    ``evaluation_arm=None`` means normal evaluation under the trained operator.
    Supplying another arm changes only the operator fields; all architecture and
    learned-parameter shape settings remain those of the source checkpoint.
    """
    trained_arm = get_arm(str(run.arm))
    target_arm = (
        trained_arm
        if evaluation_arm is None
        else evaluation_arm
        if isinstance(evaluation_arm, ArmSpec)
        else get_arm(evaluation_arm)
    )

    trained_config = trained_model_config(run)
    overrides = target_arm.config_overrides()
    config = replace(trained_config, **overrides)
    _validate_evaluation_compatibility(config, target_arm)
    return config, trained_arm, target_arm


def build_model(
    config: GPTConfig,
    *,
    device: str | Any,
    initialize: bool = True,
) -> GPT:
    """Construct a GPT using nanochat's meta -> to_empty initialization pattern.

    ``initialize=True`` is the safe default even when weights will immediately
    be loaded: ``GPT.init_weights`` also materializes non-persistent rotary
    buffers that are intentionally absent from state dicts.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - nanochat requires torch
        raise ExperimentModelError("constructing GPT requires torch") from exc

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    if initialize:
        model.init_weights()
    return model


def load_local_model(
    local: LocalCheckpoint,
    *,
    evaluation_arm: str | ArmSpec | None = None,
    device: str | Any = "cuda",
    weights_device: str = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> LoadedModel:
    """Load one canonical local checkpoint under a requested evaluation arm.

    The state dict is loaded on ``weights_device`` (CPU by default) and copied
    into the already-materialized target model. This avoids keeping two full
    copies of the model on GPU during loading. Strict key/shape validation is
    the default and should remain enabled for scientific operator-graft runs.
    """
    local.require_portable_files(require_run_metadata=True)

    checkpoint_metadata = read_json(local.metadata_path)
    run = load_local_provenance(local)
    validate_checkpoint_provenance(local.ref, checkpoint_metadata, run)

    config, trained_arm, target_arm = evaluation_model_config(run, evaluation_arm)
    model = build_model(config, device=device, initialize=True)

    report = load_model_weights(
        model,
        local.model_path,
        device=weights_device,
        strict=strict,
        assign=False,
    )

    if eval_mode:
        model.eval()

    return LoadedModel(
        model=model,
        local_checkpoint=local,
        run=run,
        checkpoint_metadata=checkpoint_metadata,
        trained_arm=trained_arm,
        evaluation_arm=target_arm,
        model_config=config,
        state_dict_validation=report,
    )


def load_model(
    root: str | Path,
    checkpoint: CheckpointRef,
    *,
    evaluation_arm: str | ArmSpec | None = None,
    device: str | Any = "cuda",
    weights_device: str = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> LoadedModel:
    """Resolve and load a checkpoint from a local mirror of the shared HF repo."""
    local = resolve_local_checkpoint(
        root,
        checkpoint,
        require_files=True,
        require_run_metadata=True,
        validate_metadata=True,
    )
    return load_local_model(
        local,
        evaluation_arm=evaluation_arm,
        device=device,
        weights_device=weights_device,
        strict=strict,
        eval_mode=eval_mode,
    )


__all__ = [
    "ExperimentModelError",
    "ModelConfigError",
    "LoadedModel",
    "trained_model_config",
    "evaluation_model_config",
    "build_model",
    "load_local_model",
    "load_model",
]
