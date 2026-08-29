# experiments/common/__init__.py
"""experiments/common/__init__.py
Shared experiment infrastructure.

This package provides the common contracts used by induction, ICL, CORE,
transition, and other research experiments.
"""

from .arms import (
    ARM_NAMES,
    ArmName,
    ArmSpec,
    get_arm,
    iter_arms,
    normalize_arm_name,
)

from .checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_FILENAME,
    CHECKPOINT_METADATA_FILENAME,
    RUN_METADATA_FILENAME,
    RESUME_DIRNAME,
    SAMPLES_DIRNAME,
    CheckpointError,
    CheckpointFormatError,
    StateDictMismatchError,
    CheckpointRef,
    LocalCheckpoint,
    ShapeMismatch,
    StateDictValidation,
    validate_state_dict,
    load_state_dict_file,
    save_state_dict_safetensors,
    load_model_weights,
    read_json,
    write_json_atomic,
    validate_checkpoint_metadata,
    resolve_local_checkpoint,
    discover_checkpoints,
    latest_checkpoint,
)

from .provenance import (
    RUN_SCHEMA_VERSION,
    InitializationKind,
    SourceCheckpoint,
    ScratchInitialization,
    GraftInitialization,
    Initialization,
    RunProvenance,
    new_scratch_run,
    new_graft_run,
    validate_run_for_checkpoint,
    build_checkpoint_metadata,
    validate_checkpoint_provenance,
    read_run_provenance,
    write_run_provenance,
    load_local_provenance,
)

from .model import (
    ExperimentModelError,
    ModelConfigError,
    LoadedModel,
    trained_model_config,
    evaluation_model_config,
    build_model,
    load_local_model,
    load_model,
)

from .results import (
    ResultError,
    ResultFormatError,
    ExperimentSpec,
    ExperimentResult,
    build_result,
    write_result,
    read_result,
)


__all__ = [
    # arms
    "ARM_NAMES",
    "ArmName",
    "ArmSpec",
    "get_arm",
    "iter_arms",
    "normalize_arm_name",

    # checkpoints
    "CHECKPOINT_SCHEMA_VERSION",
    "MODEL_FILENAME",
    "CHECKPOINT_METADATA_FILENAME",
    "RUN_METADATA_FILENAME",
    "RESUME_DIRNAME",
    "SAMPLES_DIRNAME",
    "CheckpointError",
    "CheckpointFormatError",
    "StateDictMismatchError",
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

    # provenance
    "RUN_SCHEMA_VERSION",
    "InitializationKind",
    "SourceCheckpoint",
    "ScratchInitialization",
    "GraftInitialization",
    "Initialization",
    "RunProvenance",
    "new_scratch_run",
    "new_graft_run",
    "validate_run_for_checkpoint",
    "build_checkpoint_metadata",
    "validate_checkpoint_provenance",
    "read_run_provenance",
    "write_run_provenance",
    "load_local_provenance",

    # model
    "ExperimentModelError",
    "ModelConfigError",
    "LoadedModel",
    "trained_model_config",
    "evaluation_model_config",
    "build_model",
    "load_local_model",
    "load_model",

    # results
    "ResultError",
    "ResultFormatError",
    "ExperimentSpec",
    "ExperimentResult",
    "build_result",
    "write_result",
    "read_result",
]
