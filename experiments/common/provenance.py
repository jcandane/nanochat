"""experiments/common/provenance.py
Run-level lineage and checkpoint provenance for nanochat experiments.

This module answers a different question from ``checkpoints.py``:

* ``checkpoints.py`` says **where a checkpoint lives and whether its weights fit**.
* ``provenance.py`` says **how those weights came to exist**.

The distinction matters for operator-graft experiments. A checkpoint stored at
``AMAP/run-0003/...`` is AMAP-trained at that point in its history, but it may
have been initialized from an attention checkpoint with an instantaneous
operator swap before continued pretraining.

Canonical ``run.json`` records that ancestry explicitly. Graft runs can inherit
their source run's lineage, so chains such as::

    attention -> AMAP -> attention -> HMAP

remain self-describing without encoding scientific history into directory names.
The immediate source checkpoint is always retained as a concrete pointer even
when inherited lineage is also present.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from experiments.common.arms import ArmName, get_arm, normalize_arm_name
from experiments.common.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    RUN_METADATA_FILENAME,
    CheckpointFormatError,
    CheckpointRef,
    LocalCheckpoint,
    read_json,
    validate_checkpoint_metadata,
    validate_run_id,
    write_json_atomic,
)


RUN_SCHEMA_VERSION = 1
InitializationKind = Literal["scratch", "graft"]


def utc_now_iso() -> str:
    """Return an RFC-3339-like UTC timestamp suitable for JSON metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _copy_json_object(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True, slots=True)
class SourceCheckpoint:
    """Concrete parent checkpoint used to initialize a graft/reconditioning run.

    ``repo_id`` and ``revision`` are optional because most experiments use the
    same shared Hugging Face repository. When supplied, they make cross-repo or
    pinned-revision ancestry unambiguous.
    """

    ref: CheckpointRef
    repo_id: str | None = None
    revision: str | None = None
    model_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"checkpoint": self.ref.as_dict()}
        if self.repo_id:
            out["repo_id"] = self.repo_id
        if self.revision:
            out["revision"] = self.revision
        if self.model_sha256:
            out["model_sha256"] = self.model_sha256
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceCheckpoint":
        checkpoint = value.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise CheckpointFormatError(
                "graft initialization must contain a 'checkpoint' object"
            )
        return cls(
            ref=CheckpointRef.from_dict(checkpoint),
            repo_id=str(value["repo_id"]) if value.get("repo_id") else None,
            revision=str(value["revision"]) if value.get("revision") else None,
            model_sha256=(
                str(value["model_sha256"]) if value.get("model_sha256") else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ScratchInitialization:
    """Initialization of a training run from freshly initialized model weights."""

    seed: int | None = None

    @property
    def kind(self) -> Literal["scratch"]:
        return "scratch"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind}
        if self.seed is not None:
            out["seed"] = int(self.seed)
        return out


@dataclass(frozen=True, slots=True)
class GraftInitialization:
    """Weights-only initialization under a target operator.

    A graft changes the forward operator but does not modify the source weights.
    Continued pretraining happens *after* this initialization with a fresh
    optimizer/data stream unless the training harness explicitly records a
    different policy in ``training``.
    """

    source: SourceCheckpoint

    @property
    def kind(self) -> Literal["graft"]:
        return "graft"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "source": self.source.as_dict(),
        }


Initialization = ScratchInitialization | GraftInitialization


def initialization_from_dict(value: Mapping[str, Any]) -> Initialization:
    kind = value.get("type")
    if kind == "scratch":
        seed = value.get("seed")
        if seed is not None:
            if isinstance(seed, bool):
                raise CheckpointFormatError("scratch initialization seed must be an integer")
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise CheckpointFormatError(
                    "scratch initialization seed must be an integer"
                ) from exc
        return ScratchInitialization(seed=seed)

    if kind == "graft":
        source = value.get("source")
        if not isinstance(source, Mapping):
            raise CheckpointFormatError(
                "graft initialization must contain a 'source' object"
            )
        return GraftInitialization(source=SourceCheckpoint.from_dict(source))

    raise CheckpointFormatError(
        f"unknown initialization type {kind!r}; expected 'scratch' or 'graft'"
    )


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Canonical, JSON-serializable provenance for one training run."""

    run_id: str
    arm: ArmName | str
    initialization: Initialization
    lineage: tuple[dict[str, Any], ...]
    model: dict[str, Any]
    training: dict[str, Any]
    created_at: str
    git_commit: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        object.__setattr__(self, "arm", normalize_arm_name(str(self.arm)))

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty timestamp string")

        # A graft's concrete parent must point to a different run identity or a
        # previous checkpoint. We allow same-arm continued-pretraining grafts;
        # scientific direction is represented by source/target arm, not blocked.
        if isinstance(self.initialization, GraftInitialization):
            source_ref = self.initialization.source.ref
            if source_ref.run_id == self.run_id and source_ref.arm == self.arm:
                raise ValueError(
                    "a graft run cannot use a checkpoint from the same arm/run_id; "
                    "resume the existing run instead"
                )

    @property
    def operator(self) -> dict[str, object]:
        return get_arm(str(self.arm)).as_dict()

    @property
    def lineage_depth(self) -> int:
        return len(self.lineage)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "arm": self.arm,
            "operator": self.operator,
            "initialization": self.initialization.as_dict(),
            "lineage": [dict(event) for event in self.lineage],
            "model": dict(self.model),
            "training": dict(self.training),
            "created_at": self.created_at,
        }
        if self.git_commit:
            out["git_commit"] = self.git_commit
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunProvenance":
        version = value.get("schema_version")
        if version != RUN_SCHEMA_VERSION:
            raise CheckpointFormatError(
                f"run schema_version={version!r}; expected {RUN_SCHEMA_VERSION}"
            )

        initialization = value.get("initialization")
        if not isinstance(initialization, Mapping):
            raise CheckpointFormatError("run.json must contain an initialization object")

        lineage_raw = value.get("lineage", [])
        if not isinstance(lineage_raw, list) or not all(
            isinstance(event, Mapping) for event in lineage_raw
        ):
            raise CheckpointFormatError("run.json lineage must be a list of objects")

        model = value.get("model", {})
        training = value.get("training", {})
        if not isinstance(model, Mapping):
            raise CheckpointFormatError("run.json model must be an object")
        if not isinstance(training, Mapping):
            raise CheckpointFormatError("run.json training must be an object")

        try:
            run_id = str(value["run_id"])
            arm = str(value["arm"])
            created_at = str(value["created_at"])
        except KeyError as exc:
            raise CheckpointFormatError(
                f"run.json missing required field {exc.args[0]!r}"
            ) from exc

        run = cls(
            run_id=run_id,
            arm=arm,
            initialization=initialization_from_dict(initialization),
            lineage=tuple(dict(event) for event in lineage_raw),
            model=dict(model),
            training=dict(training),
            created_at=created_at,
            git_commit=str(value["git_commit"]) if value.get("git_commit") else None,
            notes=str(value["notes"]) if value.get("notes") else None,
        )
        _validate_operator_snapshot(run, value.get("operator"))
        return run


def _validate_operator_snapshot(
    run: RunProvenance,
    snapshot: Any,
) -> None:
    """Catch stale alpha/beta conventions in persisted metadata."""
    if snapshot is None:
        return
    if not isinstance(snapshot, Mapping):
        raise CheckpointFormatError("run.json operator must be an object")

    expected = get_arm(str(run.arm))
    checks = {
        "name": expected.name,
        "beta": expected.beta,
        "alpha": expected.alpha,
        "attn_variant": expected.attn_variant,
    }
    for key, expected_value in checks.items():
        if key in snapshot and snapshot[key] != expected_value:
            raise CheckpointFormatError(
                f"run.json operator.{key}={snapshot[key]!r} does not match "
                f"canonical {run.arm} value {expected_value!r}"
            )


def _scratch_event(
    *, run_id: str, arm: str, seed: int | None
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event": "scratch_init",
        "run_id": run_id,
        "arm": normalize_arm_name(arm),
    }
    if seed is not None:
        event["seed"] = int(seed)
    return event


def _graft_event(
    *,
    run_id: str,
    target_arm: str,
    source: SourceCheckpoint,
) -> dict[str, Any]:
    target = get_arm(target_arm)
    return {
        "event": "operator_graft",
        "run_id": run_id,
        "source": source.as_dict(),
        "operator_swap": {
            "from": source.ref.arm,
            "to": target.name,
        },
        "target_operator": target.as_dict(),
    }


def new_scratch_run(
    *,
    run_id: str,
    arm: str,
    model: Mapping[str, Any],
    training: Mapping[str, Any],
    seed: int | None = None,
    git_commit: str | None = None,
    created_at: str | None = None,
    notes: str | None = None,
) -> RunProvenance:
    """Create provenance for a run initialized from fresh model weights."""
    validate_run_id(run_id)
    arm_name = normalize_arm_name(arm)
    initialization = ScratchInitialization(seed=seed)
    return RunProvenance(
        run_id=run_id,
        arm=arm_name,
        initialization=initialization,
        lineage=(_scratch_event(run_id=run_id, arm=arm_name, seed=seed),),
        model=_copy_json_object(model),
        training=_copy_json_object(training),
        created_at=created_at or utc_now_iso(),
        git_commit=git_commit,
        notes=notes,
    )


def new_graft_run(
    *,
    run_id: str,
    target_arm: str,
    source_checkpoint: CheckpointRef,
    model: Mapping[str, Any],
    training: Mapping[str, Any],
    source_run: RunProvenance | None = None,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
    source_model_sha256: str | None = None,
    git_commit: str | None = None,
    created_at: str | None = None,
    notes: str | None = None,
) -> RunProvenance:
    """Create a weights-only operator-graft/reconditioning run.

    If ``source_run`` is available, its lineage is copied and the new graft
    event appended. This keeps the new ``run.json`` self-contained while the
    immediate source checkpoint remains the authoritative parent pointer.
    """
    validate_run_id(run_id)
    target = normalize_arm_name(target_arm)

    if source_run is not None:
        if source_run.run_id != source_checkpoint.run_id:
            raise ValueError(
                "source_run.run_id does not match source_checkpoint.run_id: "
                f"{source_run.run_id!r} != {source_checkpoint.run_id!r}"
            )
        if source_run.arm != source_checkpoint.arm:
            raise ValueError(
                "source_run.arm does not match source_checkpoint.arm: "
                f"{source_run.arm!r} != {source_checkpoint.arm!r}"
            )

    source = SourceCheckpoint(
        ref=source_checkpoint,
        repo_id=source_repo_id,
        revision=source_revision,
        model_sha256=source_model_sha256,
    )
    inherited = tuple(source_run.lineage) if source_run is not None else ()
    lineage = inherited + (
        _graft_event(run_id=run_id, target_arm=target, source=source),
    )

    return RunProvenance(
        run_id=run_id,
        arm=target,
        initialization=GraftInitialization(source=source),
        lineage=lineage,
        model=_copy_json_object(model),
        training=_copy_json_object(training),
        created_at=created_at or utc_now_iso(),
        git_commit=git_commit,
        notes=notes,
    )


def validate_run_for_checkpoint(run: RunProvenance, ref: CheckpointRef) -> None:
    """Require a checkpoint path to belong to the run that claims it."""
    if run.run_id != ref.run_id or run.arm != ref.arm:
        raise CheckpointFormatError(
            "checkpoint does not belong to run provenance: "
            f"checkpoint={ref.as_dict()}, run={{'arm': {run.arm!r}, "
            f"'run_id': {run.run_id!r}}}"
        )


def build_checkpoint_metadata(
    ref: CheckpointRef,
    run: RunProvenance,
    *,
    tokens_seen: int | None = None,
    validation_bpb: float | None = None,
    model_sha256: str | None = None,
    created_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical ``checkpoint.json`` payload for one saved step."""
    validate_run_for_checkpoint(run, ref)

    out: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "arm": ref.arm,
        "run_id": ref.run_id,
        "step": ref.step,
        "path": ref.directory.as_posix(),
        "model_file": ref.model_path.name,
        "trained_operator": get_arm(str(ref.arm)).as_dict(),
        "initialization_type": run.initialization.kind,
        "lineage_depth": run.lineage_depth,
        "created_at": created_at or utc_now_iso(),
    }
    if tokens_seen is not None:
        if isinstance(tokens_seen, bool) or int(tokens_seen) < 0:
            raise ValueError("tokens_seen must be a non-negative integer")
        out["tokens_seen"] = int(tokens_seen)
    if validation_bpb is not None:
        out["validation_bpb"] = float(validation_bpb)
    if model_sha256:
        out["model_sha256"] = model_sha256
    if run.git_commit:
        out["git_commit"] = run.git_commit

    if isinstance(run.initialization, GraftInitialization):
        out["parent_checkpoint"] = run.initialization.source.as_dict()

    if extra:
        protected = set(out)
        overlap = protected & set(extra)
        if overlap:
            raise ValueError(
                "extra checkpoint metadata cannot overwrite canonical fields: "
                + ", ".join(sorted(overlap))
            )
        out.update(dict(extra))
    return out


def validate_checkpoint_provenance(
    ref: CheckpointRef,
    checkpoint_metadata: Mapping[str, Any],
    run: RunProvenance,
) -> None:
    """Validate ``checkpoint.json`` against both its path and ``run.json``."""
    validate_checkpoint_metadata(ref, checkpoint_metadata)
    validate_run_for_checkpoint(run, ref)

    init_type = checkpoint_metadata.get("initialization_type")
    if init_type is not None and init_type != run.initialization.kind:
        raise CheckpointFormatError(
            f"checkpoint initialization_type={init_type!r} does not match "
            f"run initialization {run.initialization.kind!r}"
        )

    depth = checkpoint_metadata.get("lineage_depth")
    if depth is not None:
        try:
            depth_int = int(depth)
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError("checkpoint lineage_depth must be an integer") from exc
        if depth_int != run.lineage_depth:
            raise CheckpointFormatError(
                f"checkpoint lineage_depth={depth_int} does not match "
                f"run lineage depth {run.lineage_depth}"
            )

    if isinstance(run.initialization, GraftInitialization):
        parent = checkpoint_metadata.get("parent_checkpoint")
        if parent is not None:
            if not isinstance(parent, Mapping):
                raise CheckpointFormatError("checkpoint parent_checkpoint must be an object")
            parsed = SourceCheckpoint.from_dict(parent)
            if parsed.ref != run.initialization.source.ref:
                raise CheckpointFormatError(
                    "checkpoint parent_checkpoint does not match run initialization source"
                )


def read_run_provenance(path: str | Path) -> RunProvenance:
    """Read and validate a canonical ``run.json`` file."""
    return RunProvenance.from_dict(read_json(path))


def write_run_provenance(path: str | Path, run: RunProvenance) -> Path:
    """Atomically write a canonical ``run.json`` file."""
    path = Path(path)
    if path.name != RUN_METADATA_FILENAME:
        raise ValueError(
            f"run provenance must be written as {RUN_METADATA_FILENAME!r}, got {path.name!r}"
        )
    return write_json_atomic(path, run.as_dict())


def load_local_provenance(local: LocalCheckpoint) -> RunProvenance:
    """Load ``run.json`` and validate it against a resolved local checkpoint."""
    run = read_run_provenance(local.run_metadata_path)
    validate_run_for_checkpoint(run, local.ref)
    return run


__all__ = [
    "RUN_SCHEMA_VERSION",
    "InitializationKind",
    "SourceCheckpoint",
    "ScratchInitialization",
    "GraftInitialization",
    "Initialization",
    "RunProvenance",
    "utc_now_iso",
    "initialization_from_dict",
    "new_scratch_run",
    "new_graft_run",
    "validate_run_for_checkpoint",
    "build_checkpoint_metadata",
    "validate_checkpoint_provenance",
    "read_run_provenance",
    "write_run_provenance",
    "load_local_provenance",
]
