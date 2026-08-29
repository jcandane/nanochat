"""experiments/common/checkpoints.py
Checkpoint identities, paths, and strict state-dict I/O for experiments.

This module defines the *portable* checkpoint contract used by the research
layer. It deliberately does **not** know how to talk to Hugging Face, Modal, or
any other remote service. Remote code can download/upload these paths; the
scientific code only needs a stable identity and local files.

Canonical shared-repo layout
----------------------------

::

    <ARM>/
      run-0001/
        run.json
        checkpoints/
          step-001000/
            model.safetensors
            checkpoint.json
            resume/               # optional, trainer-specific state
        samples/

where ``<ARM>`` is ``attention``, ``AMAP``, ``HMAP``, or ``DMAP``.

``model.safetensors`` is the portable model artifact used for inference and
operator grafts. ``resume/`` is intentionally separate because optimizer and
dataloader state are only needed to resume the *same* training run.

Legacy nanochat ``.pt`` state dicts are supported for migration/loading, but
they are not part of the canonical storage layout above.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, MutableMapping

from experiments.common.arms import ArmName, get_arm, normalize_arm_name


CHECKPOINT_SCHEMA_VERSION = 1
MODEL_FILENAME = "model.safetensors"
CHECKPOINT_METADATA_FILENAME = "checkpoint.json"
RUN_METADATA_FILENAME = "run.json"
RESUME_DIRNAME = "resume"
SAMPLES_DIRNAME = "samples"

_RUN_ID_RE = re.compile(r"^run-(\d{4,})$")
_STEP_DIR_RE = re.compile(r"^step-(\d+)$")


def validate_run_id(run_id: str) -> str:
    """Validate and return a canonical run id such as ``run-0001``.

    This helper is intentionally public because provenance construction needs
    to validate run identities without constructing a synthetic checkpoint.
    """
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(
            f"invalid run_id {run_id!r}; expected 'run-' followed by at least "
            "four digits, e.g. 'run-0001'"
        )
    return run_id


class CheckpointError(RuntimeError):
    """Base class for experiment checkpoint errors."""


class CheckpointFormatError(CheckpointError):
    """Raised when a checkpoint path or metadata file violates the schema."""


class StateDictMismatchError(CheckpointError):
    """Raised when a state dict is not key/shape compatible with a model."""


@dataclass(frozen=True, slots=True, order=True)
class CheckpointRef:
    """Stable identity of one model checkpoint in the shared repository.

    ``arm`` means the operator under which these weights were *trained* at this
    point in their lineage. It does **not** mean the operator currently used to
    evaluate them. Fixed-weight operator swaps therefore keep the same
    ``CheckpointRef`` and record the evaluation arm separately.
    """

    arm: ArmName | str
    run_id: str
    step: int

    def __post_init__(self) -> None:
        normalized = normalize_arm_name(str(self.arm))
        object.__setattr__(self, "arm", normalized)

        validate_run_id(self.run_id)
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"step must be a non-negative integer, got {self.step!r}")

    @property
    def arm_folder(self) -> str:
        return get_arm(str(self.arm)).hf_folder

    @property
    def step_dirname(self) -> str:
        return f"step-{self.step:06d}"

    @property
    def run_dir(self) -> PurePosixPath:
        return PurePosixPath(self.arm_folder, self.run_id)

    @property
    def checkpoints_dir(self) -> PurePosixPath:
        return self.run_dir / "checkpoints"

    @property
    def directory(self) -> PurePosixPath:
        """HF-relative directory containing this checkpoint."""
        return self.checkpoints_dir / self.step_dirname

    @property
    def model_path(self) -> PurePosixPath:
        return self.directory / MODEL_FILENAME

    @property
    def metadata_path(self) -> PurePosixPath:
        return self.directory / CHECKPOINT_METADATA_FILENAME

    @property
    def run_metadata_path(self) -> PurePosixPath:
        return self.run_dir / RUN_METADATA_FILENAME

    @property
    def resume_dir(self) -> PurePosixPath:
        return self.directory / RESUME_DIRNAME

    @property
    def samples_dir(self) -> PurePosixPath:
        return self.run_dir / SAMPLES_DIRNAME

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "run_id": self.run_id,
            "step": self.step,
            "path": self.directory.as_posix(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CheckpointRef":
        try:
            arm = str(value["arm"])
            run_id = str(value["run_id"])
            step = int(value["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                "checkpoint reference must contain arm, run_id, and integer step"
            ) from exc
        return cls(arm=arm, run_id=run_id, step=step)

    @classmethod
    def from_repo_path(cls, path: str | os.PathLike[str]) -> "CheckpointRef":
        """Parse a canonical repo-relative checkpoint path.

        Accepted inputs may point to the checkpoint directory itself or to a
        file inside it, for example both of these are valid::

            AMAP/run-0003/checkpoints/step-001000
            AMAP/run-0003/checkpoints/step-001000/model.safetensors
        """
        parts = PurePosixPath(str(path).replace("\\", "/")).parts
        if len(parts) >= 1 and parts[-1] in {
            MODEL_FILENAME,
            CHECKPOINT_METADATA_FILENAME,
        }:
            parts = parts[:-1]

        if len(parts) != 4 or parts[2] != "checkpoints":
            raise CheckpointFormatError(
                "expected '<ARM>/run-XXXX/checkpoints/step-XXXXXX' "
                f"(optionally followed by a checkpoint filename), got {path!r}"
            )

        arm_folder, run_id, _, step_dir = parts
        # Folder presentation is intentionally distinct from the normalized arm
        # name. Resolve by comparing against the canonical ArmSpec.hf_folder.
        folder_to_arm = {
            get_arm(name).hf_folder.lower(): name
            for name in ("attention", "amap", "hmap", "dmap")
        }
        try:
            arm = folder_to_arm[arm_folder.lower()]
        except KeyError as exc:
            raise CheckpointFormatError(f"unknown arm folder {arm_folder!r}") from exc

        match = _STEP_DIR_RE.fullmatch(step_dir)
        if match is None:
            raise CheckpointFormatError(
                f"invalid checkpoint step directory {step_dir!r}; expected 'step-XXXXXX'"
            )
        return cls(arm=arm, run_id=run_id, step=int(match.group(1)))


@dataclass(frozen=True, slots=True)
class LocalCheckpoint:
    """Resolved local files for a :class:`CheckpointRef`."""

    root: Path
    ref: CheckpointRef

    @property
    def directory(self) -> Path:
        return self.root / Path(*self.ref.directory.parts)

    @property
    def model_path(self) -> Path:
        return self.directory / MODEL_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.directory / CHECKPOINT_METADATA_FILENAME

    @property
    def run_metadata_path(self) -> Path:
        return self.root / Path(*self.ref.run_metadata_path.parts)

    @property
    def resume_dir(self) -> Path:
        return self.directory / RESUME_DIRNAME

    @property
    def samples_dir(self) -> Path:
        return self.root / Path(*self.ref.samples_dir.parts)

    def require_portable_files(self, *, require_run_metadata: bool = True) -> None:
        """Require the files needed for portable evaluation/grafting."""
        missing: list[Path] = []
        if not self.model_path.is_file():
            missing.append(self.model_path)
        if not self.metadata_path.is_file():
            missing.append(self.metadata_path)
        if require_run_metadata and not self.run_metadata_path.is_file():
            missing.append(self.run_metadata_path)
        if missing:
            rendered = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(f"checkpoint is incomplete; missing:\n  {rendered}")


@dataclass(frozen=True, slots=True)
class ShapeMismatch:
    key: str
    checkpoint_shape: tuple[int, ...]
    model_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StateDictValidation:
    """Key/shape compatibility report for a model and checkpoint state dict."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)

    def summary(self, *, max_items: int = 8) -> str:
        if self.ok:
            return "state_dict is key-for-key and shape-for-shape compatible"

        chunks: list[str] = []
        if self.missing_keys:
            shown = ", ".join(self.missing_keys[:max_items])
            suffix = " ..." if len(self.missing_keys) > max_items else ""
            chunks.append(f"missing keys: {shown}{suffix}")
        if self.unexpected_keys:
            shown = ", ".join(self.unexpected_keys[:max_items])
            suffix = " ..." if len(self.unexpected_keys) > max_items else ""
            chunks.append(f"unexpected keys: {shown}{suffix}")
        if self.shape_mismatches:
            shown_items = self.shape_mismatches[:max_items]
            shown = ", ".join(
                f"{item.key}: ckpt{item.checkpoint_shape} != model{item.model_shape}"
                for item in shown_items
            )
            suffix = " ..." if len(self.shape_mismatches) > max_items else ""
            chunks.append(f"shape mismatches: {shown}{suffix}")
        return "; ".join(chunks)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise StateDictMismatchError(self.summary())


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except (TypeError, ValueError):
        return None


def validate_state_dict(
    model_or_state: Any,
    checkpoint_state: Mapping[str, Any],
) -> StateDictValidation:
    """Compare state-dict keys and tensor shapes without mutating a model.

    ``model_or_state`` may be an ``nn.Module`` or an already materialized state
    dict. This keeps the validator easy to unit-test and useful in migration
    tooling.
    """
    if hasattr(model_or_state, "state_dict") and callable(model_or_state.state_dict):
        model_state = model_or_state.state_dict()
    elif isinstance(model_or_state, Mapping):
        model_state = model_or_state
    else:
        raise TypeError("model_or_state must be an nn.Module-like object or Mapping")

    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)

    missing = tuple(sorted(model_keys - checkpoint_keys))
    unexpected = tuple(sorted(checkpoint_keys - model_keys))

    shape_bad: list[ShapeMismatch] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = _shape_of(model_state[key])
        checkpoint_shape = _shape_of(checkpoint_state[key])
        if model_shape is None or checkpoint_shape is None:
            continue
        if model_shape != checkpoint_shape:
            shape_bad.append(
                ShapeMismatch(
                    key=key,
                    checkpoint_shape=checkpoint_shape,
                    model_shape=model_shape,
                )
            )

    return StateDictValidation(
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=tuple(shape_bad),
    )


def load_state_dict_file(
    path: str | os.PathLike[str],
    *,
    device: str = "cpu",
) -> MutableMapping[str, Any]:
    """Load a portable safetensors file or a legacy nanochat ``.pt`` state dict.

    ``.pt`` support exists only to migrate/evaluate existing nanochat runs. New
    experiment artifacts should be written as ``model.safetensors``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CheckpointError(
                "loading .safetensors requires the 'safetensors' package"
            ) from exc
        state = load_file(str(path), device=device)
    elif path.suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - nanochat requires torch
            raise CheckpointError("loading legacy .pt checkpoints requires torch") from exc
        state = torch.load(path, map_location=device, weights_only=True)
    else:
        raise CheckpointFormatError(
            f"unsupported model file extension {path.suffix!r}; expected .safetensors or .pt"
        )

    if not isinstance(state, MutableMapping):
        raise CheckpointFormatError(
            f"model file {path} did not contain a state_dict mapping; got {type(state).__name__}"
        )
    return state


def save_state_dict_safetensors(
    state_dict: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write a portable model state dict in canonical safetensors format.

    Tensors are detached, made contiguous, and moved to CPU before writing so
    the artifact is independent of the training device. Non-tensor state-dict
    values are rejected rather than silently discarded.
    """
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CheckpointError(
            "saving model.safetensors requires torch and the 'safetensors' package"
        ) from exc

    tensors: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"state_dict[{key!r}] is {type(value).__name__}, expected torch.Tensor"
            )
        tensors[key] = value.detach().cpu().contiguous()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=dict(metadata or {}))
    return path


def load_model_weights(
    model: Any,
    path: str | os.PathLike[str],
    *,
    device: str = "cpu",
    strict: bool = True,
    assign: bool = False,
) -> StateDictValidation:
    """Load model weights, performing an explicit key/shape check first.

    The explicit validation produces a much more useful failure than relying on
    ``nn.Module.load_state_dict`` alone, and it certifies the invariant required
    for pure operator grafts: identical parameter names and shapes.
    """
    state = load_state_dict_file(path, device=device)
    report = validate_state_dict(model, state)
    if strict:
        report.raise_if_invalid()

    # ``assign`` is useful for meta/to_empty construction but is only available
    # in newer PyTorch versions. Let an older torch raise a clear TypeError if a
    # caller explicitly requests it.
    try:
        model.load_state_dict(state, strict=strict, assign=assign)
    except TypeError:
        if assign:
            raise
        model.load_state_dict(state, strict=strict)
    return report


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError(f"could not read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointFormatError(
            f"metadata {path} must contain a JSON object, got {type(value).__name__}"
        )
    return value


def write_json_atomic(path: str | os.PathLike[str], value: Mapping[str, Any]) -> Path:
    """Atomically replace a JSON metadata file on the local filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # NamedTemporaryFile in the destination directory keeps os.replace atomic on
    # ordinary local filesystems and avoids exposing a half-written manifest.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
    return path


def validate_checkpoint_metadata(
    ref: CheckpointRef,
    metadata: Mapping[str, Any],
) -> None:
    """Validate identity fields in ``checkpoint.json`` against its path.

    We intentionally validate only fields owned by this module. Training
    configuration and lineage live in richer metadata/provenance structures and
    may evolve without changing checkpoint-path semantics.

    Canonical ``checkpoint.json`` identity fields are::

        {
          "schema_version": 1,
          "arm": "amap",
          "run_id": "run-0003",
          "step": 1000,
          ...
        }
    """
    version = metadata.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"checkpoint schema_version={version!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    try:
        meta_ref = CheckpointRef(
            arm=str(metadata["arm"]),
            run_id=str(metadata["run_id"]),
            step=int(metadata["step"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError(
            "checkpoint.json must contain valid arm, run_id, and step identity fields"
        ) from exc

    if meta_ref != ref:
        raise CheckpointFormatError(
            "checkpoint metadata identity does not match its path: "
            f"path={ref.as_dict()}, metadata={meta_ref.as_dict()}"
        )


def resolve_local_checkpoint(
    root: str | os.PathLike[str],
    ref: CheckpointRef,
    *,
    require_files: bool = True,
    require_run_metadata: bool = True,
    validate_metadata: bool = True,
) -> LocalCheckpoint:
    """Resolve a checkpoint under a local mirror of the shared repo."""
    local = LocalCheckpoint(root=Path(root), ref=ref)
    if require_files:
        local.require_portable_files(require_run_metadata=require_run_metadata)
    if validate_metadata and local.metadata_path.is_file():
        validate_checkpoint_metadata(ref, read_json(local.metadata_path))
    return local


def discover_checkpoints(
    root: str | os.PathLike[str],
    *,
    arm: str | None = None,
    run_id: str | None = None,
) -> list[CheckpointRef]:
    """Discover canonical checkpoint directories in a local repository mirror.

    Only directory identity is inspected here; use ``resolve_local_checkpoint``
    when file completeness and metadata validation are required.
    """
    root = Path(root)
    arms = [get_arm(arm)] if arm is not None else [
        get_arm(name) for name in ("attention", "amap", "hmap", "dmap")
    ]

    refs: list[CheckpointRef] = []
    for arm_spec in arms:
        arm_dir = root / arm_spec.hf_folder
        if not arm_dir.is_dir():
            continue
        run_dirs = [arm_dir / run_id] if run_id is not None else sorted(arm_dir.glob("run-*"))
        for run_dir in run_dirs:
            if not run_dir.is_dir() or not _RUN_ID_RE.fullmatch(run_dir.name):
                continue
            checkpoints_dir = run_dir / "checkpoints"
            if not checkpoints_dir.is_dir():
                continue
            for step_dir in checkpoints_dir.iterdir():
                if not step_dir.is_dir():
                    continue
                match = _STEP_DIR_RE.fullmatch(step_dir.name)
                if match is None:
                    continue
                refs.append(
                    CheckpointRef(
                        arm=arm_spec.name,
                        run_id=run_dir.name,
                        step=int(match.group(1)),
                    )
                )

    return sorted(refs, key=lambda ref: (ref.arm_folder, ref.run_id, ref.step))


def latest_checkpoint(
    root: str | os.PathLike[str],
    *,
    arm: str,
    run_id: str,
) -> CheckpointRef | None:
    """Return the highest-step checkpoint for one local run, if any."""
    refs = discover_checkpoints(root, arm=arm, run_id=run_id)
    if not refs:
        return None
    return max(refs, key=lambda ref: ref.step)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MODEL_FILENAME",
    "CHECKPOINT_METADATA_FILENAME",
    "RUN_METADATA_FILENAME",
    "RESUME_DIRNAME",
    "SAMPLES_DIRNAME",
    "CheckpointError",
    "CheckpointFormatError",
    "StateDictMismatchError",
    "validate_run_id",
    "CheckpointRef",
    "LocalCheckpoint",
    "ShapeMismatch",
    "StateDictValidation",
    "validate_state_dict",
    "load_state_dict_file",
    "save_state_dict_safetensors",
    "load_model_weights",
    "read_json",
    "write_json_atomic",
    "validate_checkpoint_metadata",
    "resolve_local_checkpoint",
    "discover_checkpoints",
    "latest_checkpoint",
]
