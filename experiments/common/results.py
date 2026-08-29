"""experiments/common/results.py
Canonical experiment-result envelopes and repository paths.

Every scientific evaluation in ``experiments/`` should emit the same outer JSON
shape regardless of whether the payload is induction, ICL, CORE, DAC, or a
future metric. The result envelope keeps three identities separate:

* the checkpoint whose **weights** are being evaluated,
* the operator those weights were **trained under**, and
* the operator used for the **current forward pass**.

That separation is essential for fixed-weight operator-graft experiments. For
example, attention-trained weights evaluated under DMAP remain an ``attention``
checkpoint while ``evaluation_operator`` is DMAP.

Canonical shared-repo layout
----------------------------

Results live outside checkpoint directories so evaluation code can evolve
without mutating model artifacts::

    metrics/
      induction/
        attention/
          run-0001/
            step-010000/
              as-AMAP__v001-a1b2c3d4e5f6.json

The suffix is a deterministic fingerprint of the experiment name/version and
its evaluation configuration. Re-running an identical evaluation therefore has
a stable destination, while changing lengths, seeds, task limits, etc. creates
a distinct artifact.

``results.py`` contains no Hugging Face, Modal, or torch-specific I/O. It only
defines JSON-safe metadata and local filesystem writes; remote launchers can
upload the resulting repo-relative path verbatim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePath, PurePosixPath
import re
import tempfile
from typing import TYPE_CHECKING, Any, Mapping

from experiments.common.arms import ArmSpec, get_arm
from experiments.common.checkpoints import CheckpointRef

if TYPE_CHECKING:
    from experiments.common.model import LoadedModel


RESULT_SCHEMA_VERSION = 1
METRICS_DIRNAME = "metrics"
_RESULT_HASH_HEX = 12
_EXPERIMENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_RESULT_FILENAME_RE = re.compile(
    r"^as-([A-Za-z0-9_-]+)__v(\d{3,})-([0-9a-f]{12})\.json$"
)


class ResultError(RuntimeError):
    """Base class for experiment-result errors."""


class ResultFormatError(ResultError):
    """Raised when a result cannot satisfy the canonical JSON schema."""


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for result metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def validate_experiment_name(name: str) -> str:
    """Validate the stable slug used beneath ``metrics/``.

    Examples: ``induction``, ``icl``, ``core``, ``dac``.
    Human-readable titles belong inside metric payloads, not in paths.
    """
    if not isinstance(name, str):
        raise ValueError("experiment name must be a string")
    normalized = name.strip().lower()
    if not _EXPERIMENT_NAME_RE.fullmatch(normalized):
        raise ValueError(
            f"invalid experiment name {name!r}; expected lowercase letters, "
            "digits, '_' or '-', beginning with a letter"
        )
    return normalized


def validate_experiment_version(version: int) -> int:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"experiment version must be a positive integer, got {version!r}")
    return version


def _jsonable(value: Any, *, where: str = "$") -> Any:
    """Convert common scientific-Python values to strict JSON-compatible data.

    Non-finite floats are rejected rather than emitting non-standard JSON NaN or
    Infinity tokens. A metric that is undefined should record ``None`` and, when
    useful, an accompanying reason field.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultFormatError(
                f"non-finite float at {where}: {value!r}; use None for undefined metrics"
            )
        return value

    if isinstance(value, PurePath):
        return value.as_posix()

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResultFormatError(
                    f"JSON object key at {where} must be a string, got {key!r}"
                )
            out[key] = _jsonable(item, where=f"{where}.{key}")
        return out

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, where=f"{where}[{idx}]")
            for idx, item in enumerate(value)
        ]

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value), where=where)

    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict(), where=where)

    # numpy / torch scalar-like values.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (ValueError, RuntimeError):
            scalar = value
        if scalar is not value:
            return _jsonable(scalar, where=where)

    # numpy arrays / torch tensors are accepted only through their explicit
    # JSON-sized representation; experiment code controls whether this is wise.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist(), where=where)
        except (ValueError, RuntimeError) as exc:
            raise ResultFormatError(f"could not serialize value at {where}") from exc

    raise ResultFormatError(
        f"unsupported result value at {where}: {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    clean = _jsonable(value)
    return json.dumps(
        clean,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Stable identity and evaluation configuration for one metric family."""

    name: str
    version: int = 1
    config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_experiment_name(self.name))
        validate_experiment_version(self.version)
        object.__setattr__(self, "config", _jsonable(dict(self.config or {}), where="$.config"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "config": dict(self.config or {}),
        }

    @property
    def fingerprint(self) -> str:
        """Short deterministic hash of metric identity + evaluation config."""
        digest = hashlib.sha256(_canonical_json_bytes(self.as_dict())).hexdigest()
        return digest[:_RESULT_HASH_HEX]

    @property
    def result_id(self) -> str:
        return f"v{self.version:03d}-{self.fingerprint}"


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Canonical JSON envelope emitted by every experiment metric."""

    experiment: ExperimentSpec
    checkpoint: CheckpointRef
    trained_arm: ArmSpec
    evaluation_arm: ArmSpec
    model: Mapping[str, Any]
    metrics: Mapping[str, Any]
    created_at: str
    git_commit: str | None = None
    weights_provenance: Mapping[str, Any] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.checkpoint.arm != self.trained_arm.name:
            raise ResultFormatError(
                f"checkpoint arm {self.checkpoint.arm!r} does not match trained_arm "
                f"{self.trained_arm.name!r}"
            )
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ResultFormatError("created_at must be a non-empty timestamp string")

        object.__setattr__(self, "model", _jsonable(dict(self.model), where="$.model"))
        object.__setattr__(self, "metrics", _jsonable(dict(self.metrics), where="$.metrics"))
        object.__setattr__(
            self,
            "weights_provenance",
            _jsonable(dict(self.weights_provenance or {}), where="$.weights.provenance"),
        )

    @property
    def is_operator_graft(self) -> bool:
        return self.trained_arm.name != self.evaluation_arm.name

    @property
    def result_id(self) -> str:
        return self.experiment.result_id

    @property
    def repo_path(self) -> PurePosixPath:
        """Canonical HF-relative location for this metric artifact."""
        eval_label = self.evaluation_arm.hf_folder
        filename = f"as-{eval_label}__{self.result_id}.json"
        return PurePosixPath(
            METRICS_DIRNAME,
            self.experiment.name,
            self.trained_arm.hf_folder,
            self.checkpoint.run_id,
            self.checkpoint.step_dirname,
            filename,
        )

    def as_dict(self) -> dict[str, Any]:
        weights: dict[str, Any] = {
            "checkpoint": self.checkpoint.as_dict(),
            "trained_arm": self.trained_arm.as_dict(),
        }
        if self.weights_provenance:
            weights["provenance"] = dict(self.weights_provenance)

        out: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "result_id": self.result_id,
            "experiment": self.experiment.as_dict(),
            "weights": weights,
            "evaluation_operator": self.evaluation_arm.as_dict(),
            "is_operator_graft": self.is_operator_graft,
            "model": dict(self.model),
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
        }
        if self.git_commit:
            out["git_commit"] = self.git_commit
        if self.notes:
            out["notes"] = self.notes
        return _jsonable(out)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentResult":
        version = value.get("schema_version")
        if version != RESULT_SCHEMA_VERSION:
            raise ResultFormatError(
                f"result schema_version={version!r}; expected {RESULT_SCHEMA_VERSION}"
            )

        experiment_raw = value.get("experiment")
        weights_raw = value.get("weights")
        evaluation_raw = value.get("evaluation_operator")
        model_raw = value.get("model")
        metrics_raw = value.get("metrics")
        if not isinstance(experiment_raw, Mapping):
            raise ResultFormatError("result experiment must be an object")
        if not isinstance(weights_raw, Mapping):
            raise ResultFormatError("result weights must be an object")
        if not isinstance(evaluation_raw, Mapping):
            raise ResultFormatError("result evaluation_operator must be an object")
        if not isinstance(model_raw, Mapping):
            raise ResultFormatError("result model must be an object")
        if not isinstance(metrics_raw, Mapping):
            raise ResultFormatError("result metrics must be an object")

        config_raw = experiment_raw.get("config", {})
        if not isinstance(config_raw, Mapping):
            raise ResultFormatError("result experiment.config must be an object")
        experiment = ExperimentSpec(
            name=str(experiment_raw.get("name", "")),
            version=int(experiment_raw.get("version", 0)),
            config=dict(config_raw),
        )

        checkpoint_raw = weights_raw.get("checkpoint")
        trained_raw = weights_raw.get("trained_arm")
        if not isinstance(checkpoint_raw, Mapping):
            raise ResultFormatError("result weights.checkpoint must be an object")
        if not isinstance(trained_raw, Mapping):
            raise ResultFormatError("result weights.trained_arm must be an object")

        checkpoint = CheckpointRef.from_dict(checkpoint_raw)
        trained_arm = get_arm(str(trained_raw.get("name", checkpoint.arm)))
        evaluation_arm = get_arm(str(evaluation_raw.get("name", "")))

        provenance_raw = weights_raw.get("provenance", {})
        if not isinstance(provenance_raw, Mapping):
            raise ResultFormatError("result weights.provenance must be an object")

        try:
            created_at = str(value["created_at"])
        except KeyError as exc:
            raise ResultFormatError("result missing required field 'created_at'") from exc

        result = cls(
            experiment=experiment,
            checkpoint=checkpoint,
            trained_arm=trained_arm,
            evaluation_arm=evaluation_arm,
            model=dict(model_raw),
            metrics=dict(metrics_raw),
            created_at=created_at,
            git_commit=str(value["git_commit"]) if value.get("git_commit") else None,
            weights_provenance=dict(provenance_raw),
            notes=str(value["notes"]) if value.get("notes") else None,
        )

        # Detect hand-edited/stale operator snapshots or result ids.
        _validate_arm_snapshot("weights.trained_arm", trained_raw, result.trained_arm)
        _validate_arm_snapshot("evaluation_operator", evaluation_raw, result.evaluation_arm)
        if value.get("result_id") != result.result_id:
            raise ResultFormatError(
                f"result_id={value.get('result_id')!r} does not match canonical "
                f"{result.result_id!r} for experiment/config"
            )
        if "is_operator_graft" in value and bool(value["is_operator_graft"]) != result.is_operator_graft:
            raise ResultFormatError("result is_operator_graft is inconsistent with arm identities")
        return result


def _validate_arm_snapshot(label: str, raw: Mapping[str, Any], expected: ArmSpec) -> None:
    checks = {
        "name": expected.name,
        "beta": expected.beta,
        "alpha": expected.alpha,
        "attn_variant": expected.attn_variant,
    }
    for key, expected_value in checks.items():
        if key in raw and raw[key] != expected_value:
            raise ResultFormatError(
                f"{label}.{key}={raw[key]!r} does not match canonical "
                f"{expected.name} value {expected_value!r}"
            )


def build_result(
    loaded: "LoadedModel",
    *,
    experiment: ExperimentSpec,
    metrics: Mapping[str, Any],
    created_at: str | None = None,
    git_commit: str | None = None,
    notes: str | None = None,
) -> ExperimentResult:
    """Build an envelope from :class:`experiments.common.model.LoadedModel`.

    ``weights_provenance`` intentionally stores a compact run summary rather
    than copying the full lineage. The checkpoint's ``run.json`` remains the
    authoritative ancestry record, while the result is still interpretable on
    its own.
    """
    model_identity = loaded.as_dict()
    model_config = model_identity.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ResultFormatError("LoadedModel.as_dict().model_config must be an object")

    run = loaded.run
    run_git_commit = getattr(run, "git_commit", None)
    initialization = getattr(run, "initialization", None)
    init_as_dict = getattr(initialization, "as_dict", None)
    provenance = {
        "run_id": run.run_id,
        "initialization": init_as_dict() if callable(init_as_dict) else None,
        "lineage_depth": run.lineage_depth,
        "run_created_at": run.created_at,
    }

    return ExperimentResult(
        experiment=experiment,
        checkpoint=loaded.checkpoint,
        trained_arm=loaded.trained_arm,
        evaluation_arm=loaded.evaluation_arm,
        model={"model_config": dict(model_config)},
        metrics=metrics,
        created_at=created_at or utc_now_iso(),
        git_commit=git_commit or run_git_commit,
        weights_provenance=provenance,
        notes=notes,
    )


def write_result(
    root: str | os.PathLike[str],
    result: ExperimentResult,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write a result beneath its canonical repo-relative path."""
    root_path = Path(root)
    destination = root_path / Path(*result.repo_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    payload = result.as_dict()
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def read_result(path: str | os.PathLike[str]) -> ExperimentResult:
    """Read and validate a canonical result JSON file."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultFormatError(f"could not read result JSON {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ResultFormatError(f"result JSON {source} must contain an object")
    result = ExperimentResult.from_dict(raw)

    # If the filename follows our canonical convention, validate it too. This
    # catches results moved/renamed into a misleading evaluation-arm identity.
    match = _RESULT_FILENAME_RE.fullmatch(source.name)
    if match is not None:
        eval_label, version_text, fingerprint = match.groups()
        if eval_label.lower() != result.evaluation_arm.hf_folder.lower():
            raise ResultFormatError(
                f"result filename evaluation arm {eval_label!r} does not match "
                f"payload {result.evaluation_arm.hf_folder!r}"
            )
        if int(version_text) != result.experiment.version:
            raise ResultFormatError("result filename version does not match payload")
        if fingerprint != result.experiment.fingerprint:
            raise ResultFormatError("result filename fingerprint does not match payload")
    return result
