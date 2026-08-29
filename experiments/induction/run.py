"""experiments/induction/run.py
Scientific runner for repeated-random induction behavior and DAC measurements.

This module connects the pure experiment components::

    CheckpointRef
        -> load_model(...)
        -> evaluate_induction_behavior(...) + DACInductionCollector
        -> ExperimentResult
        -> metrics/induction/...

It deliberately has no Hugging Face or Modal networking. ``repo_root`` is a
local mirror/staging directory using the canonical shared-repository layout.
A future ``modal/induction.py`` wrapper can download the required checkpoint,
invoke this runner, then upload the emitted metric files verbatim.

The ``evaluation_arm=\"all\"`` mode evaluates the same checkpoint weights under
all four canonical operators. Each arm restarts the deterministic synthetic RNG
from the same seed, so all operators see the same logical examples.

If a CUDA forward OOMs, only the physical microbatch is reduced. The current
length is restarted from the same seed, and the DAC collector is reset before
retrying, preserving both the logical benchmark and edge counts.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from experiments.common.arms import ARM_NAMES, ArmName, get_arm, normalize_arm_name
from experiments.common.checkpoints import CheckpointRef
from experiments.common.model import LoadedModel, load_model
from experiments.common.results import (
    ExperimentResult,
    ExperimentSpec,
    build_result,
    write_result,
)
from experiments.induction.behavior import (
    DEFAULT_MAX_FORWARD_TOKENS,
    InductionBehaviorResult,
    auto_micro_batch_size,
    evaluate_induction_behavior,
)
from experiments.induction.dac import (
    DEFAULT_OFFSETS,
    DACInductionCollector,
    top_heads,
)
from nanochat.tokenizer import get_tokenizer


EXPERIMENT_NAME = "induction"
EXPERIMENT_VERSION = 1
DEFAULT_LENGTHS = (32, 64, 128, 256, 512)


@dataclass(frozen=True, slots=True)
class InductionRunConfig:
    """Configuration shared across operator evaluations of one checkpoint."""

    lengths: tuple[int, ...] = DEFAULT_LENGTHS
    offsets: tuple[int, ...] = DEFAULT_OFFSETS
    logical_batch_size: int = 16
    num_batches: int = 4
    seed: int = 1234
    micro_batch_size: int | None = None
    max_forward_tokens: int = DEFAULT_MAX_FORWARD_TOKENS
    include_dac: bool = True
    top_k: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lengths",
            _validate_positive_unique_ints("lengths", self.lengths, minimum=2),
        )
        object.__setattr__(self, "offsets", _validate_unique_ints("offsets", self.offsets))
        _positive_int("logical_batch_size", self.logical_batch_size)
        _positive_int("num_batches", self.num_batches)
        _positive_int("max_forward_tokens", self.max_forward_tokens)
        _positive_int("top_k", self.top_k)
        if self.micro_batch_size is not None:
            _positive_int("micro_batch_size", self.micro_batch_size)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")

    def experiment_config(self) -> dict[str, Any]:
        """Stable configuration used in the canonical result fingerprint."""
        return {
            "benchmark": "repeated-random-induction",
            "sequence": "[s ; s]",
            "lengths": list(self.lengths),
            "logical_batch_size": self.logical_batch_size,
            "num_batches": self.num_batches,
            "seed": self.seed,
            "requested_micro_batch_size": self.micro_batch_size,
            "max_forward_tokens": self.max_forward_tokens,
            "dac": {
                "enabled": self.include_dac,
                "offsets": list(self.offsets) if self.include_dac else [],
                "top_k": self.top_k if self.include_dac else 0,
                "shared_behavior_forward_pass": self.include_dac,
            },
        }


@dataclass(frozen=True, slots=True)
class InductionRunArtifact:
    """One operator evaluation and the result file written for it."""

    result: ExperimentResult
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_arm": self.result.evaluation_arm.name,
            "is_operator_graft": self.result.is_operator_graft,
            "result_id": self.result.result_id,
            "repo_path": self.result.repo_path.as_posix(),
            "local_path": str(self.path),
        }


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        requirement = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {requirement}, got {value!r}")
    return value


def _validate_unique_ints(name: str, values: Iterable[int]) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{name} must contain at least one integer")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in resolved):
        raise ValueError(f"{name} must contain only integers, got {resolved!r}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates, got {resolved!r}")
    return resolved


def _validate_positive_unique_ints(
    name: str,
    values: Iterable[int],
    *,
    minimum: int = 1,
) -> tuple[int, ...]:
    resolved = _validate_unique_ints(name, values)
    for value in resolved:
        _positive_int(name, value, minimum=minimum)
    return resolved


def _parse_csv_ints(text: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated list of integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    return values


def resolve_evaluation_arms(value: str) -> tuple[ArmName, ...]:
    """Resolve one arm name or the special ``all`` sweep."""
    normalized = value.strip().lower()
    if normalized == "all":
        return ARM_NAMES
    return (normalize_arm_name(normalized),)


def _tokenizer_vocab_size() -> int:
    tokenizer = get_tokenizer()
    vocab_size = int(tokenizer.get_vocab_size())
    if vocab_size <= 0:
        raise RuntimeError(f"tokenizer returned invalid vocab size {vocab_size}")
    return vocab_size


def _validate_vocab_against_model(vocab_size: int, loaded: LoadedModel) -> None:
    model_vocab = int(loaded.model_config.vocab_size)
    if vocab_size > model_vocab:
        raise RuntimeError(
            f"tokenizer vocab_size={vocab_size} exceeds model vocab_size={model_vocab}"
        )


def _validate_lengths_against_model(
    lengths: Sequence[int],
    loaded: LoadedModel,
) -> None:
    context_len = int(loaded.model_config.sequence_len)
    too_long = [L for L in lengths if 2 * L - 1 > context_len]
    if too_long:
        rendered = ", ".join(str(L) for L in too_long)
        raise ValueError(
            f"induction half-length(s) {rendered} exceed model context "
            f"sequence_len={context_len}; require 2*L-1 <= sequence_len"
        )


def _clear_cuda_cache(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _initial_micro_batch(config: InductionRunConfig, seq_len: int) -> int:
    if config.micro_batch_size is not None:
        return min(config.logical_batch_size, config.micro_batch_size)
    return auto_micro_batch_size(
        config.logical_batch_size,
        seq_len,
        max_forward_tokens=config.max_forward_tokens,
    )


def _run_one_length(
    loaded: LoadedModel,
    *,
    vocab_size: int,
    seq_len: int,
    config: InductionRunConfig,
    collector: DACInductionCollector | None,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run one L, retrying from the same seed if CUDA OOM requires a smaller mb."""
    micro_batch = _initial_micro_batch(config, seq_len)

    while True:
        if collector is not None:
            collector.begin(seq_len)

        try:
            behavior: InductionBehaviorResult = evaluate_induction_behavior(
                loaded.model,
                vocab_size,
                seq_len=seq_len,
                logical_batch_size=config.logical_batch_size,
                num_batches=config.num_batches,
                seed=config.seed,
                micro_batch_size=micro_batch,
                max_forward_tokens=config.max_forward_tokens,
                device=device,
            )

            payload: dict[str, Any] = {"behavior": behavior.as_dict()}
            if collector is not None:
                dac_stats = collector.finish()
                payload["dac"] = dac_stats
                payload["top_heads"] = top_heads(dac_stats, k=config.top_k)
            return payload

        except torch.OutOfMemoryError:
            if torch.device(device).type != "cuda" or micro_batch <= 1:
                raise

            old_micro_batch = micro_batch
            micro_batch = max(1, micro_batch // 2)
            gc.collect()
            _clear_cuda_cache(device)
            print(
                "[induction] CUDA OOM "
                f"arm={loaded.evaluation_arm.name} L={seq_len} "
                f"micro_batch={old_micro_batch}; retrying with "
                f"micro_batch={micro_batch}",
                flush=True,
            )


def evaluate_loaded_model(
    loaded: LoadedModel,
    *,
    vocab_size: int,
    config: InductionRunConfig,
    device: str | torch.device,
) -> dict[str, Any]:
    """Evaluate all requested lengths for one already-loaded operator."""
    _validate_vocab_against_model(vocab_size, loaded)
    _validate_lengths_against_model(config.lengths, loaded)
    by_length: dict[str, Any] = {}

    collector_context = (
        DACInductionCollector(
            loaded.model,
            offsets=config.offsets,
            active_flux_coefficient=loaded.evaluation_arm.active_flux_coefficient,
        )
        if config.include_dac
        else nullcontext(None)
    )

    with collector_context as collector:
        for seq_len in config.lengths:
            values = _run_one_length(
                loaded,
                vocab_size=vocab_size,
                seq_len=seq_len,
                config=config,
                collector=collector,
                device=device,
            )
            by_length[str(seq_len)] = values

            behavior = values["behavior"]
            message = (
                f"[induction] arm={loaded.evaluation_arm.name} "
                f"L={seq_len} "
                f"loss={behavior['induction_loss']:.6f} "
                f"acc={behavior['induction_acc']:.6f} "
                f"microB={behavior['micro_batch_size']}"
            )
            if config.include_dac and values.get("top_heads"):
                head = values["top_heads"][0]
                contrast = head["local_contrast_raw"]
                contrast_text = "undefined" if contrast is None else f"{contrast:+.6f}"
                message += f" top=L{head['layer']}:H{head['head']}:{contrast_text}"
            print(message, flush=True)

    return {"by_length": by_length}


def run_checkpoint(
    *,
    repo_root: str | Path,
    checkpoint: CheckpointRef,
    evaluation_arm: str = "all",
    config: InductionRunConfig | None = None,
    output_root: str | Path | None = None,
    device: str | torch.device = "cuda",
    weights_device: str = "cpu",
    strict: bool = True,
    overwrite: bool = True,
    vocab_size: int | None = None,
) -> list[InductionRunArtifact]:
    """Evaluate a checkpoint under one or all canonical attention operators."""
    repo_root = Path(repo_root)
    output_root = repo_root if output_root is None else Path(output_root)
    config = config or InductionRunConfig()

    eval_arms = resolve_evaluation_arms(evaluation_arm)
    resolved_vocab_size = _tokenizer_vocab_size() if vocab_size is None else int(vocab_size)
    _positive_int("vocab_size", resolved_vocab_size)

    experiment = ExperimentSpec(
        name=EXPERIMENT_NAME,
        version=EXPERIMENT_VERSION,
        config=config.experiment_config(),
    )

    artifacts: list[InductionRunArtifact] = []
    for arm_name in eval_arms:
        loaded: LoadedModel | None = None
        try:
            loaded = load_model(
                repo_root,
                checkpoint,
                evaluation_arm=get_arm(arm_name),
                device=device,
                weights_device=weights_device,
                strict=strict,
                eval_mode=True,
            )

            metrics = evaluate_loaded_model(
                loaded,
                vocab_size=resolved_vocab_size,
                config=config,
                device=device,
            )

            result = build_result(
                loaded,
                experiment=experiment,
                metrics=metrics,
                notes=(
                    "Repeated-random [s;s] induction behavior"
                    + (
                        " and DAC were measured on the same model forwards."
                        if config.include_dac
                        else "."
                    )
                ),
            )
            path = write_result(output_root, result, overwrite=overwrite)
            artifacts.append(InductionRunArtifact(result=result, path=path))
            print(f"[induction] wrote {result.repo_path.as_posix()}", flush=True)

        finally:
            if loaded is not None:
                del loaded
            gc.collect()
            _clear_cuda_cache(device)

    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate repeated-random induction behavior/DAC for a canonical "
            "experiment checkpoint."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="local mirror/staging root containing model folders and metrics/",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="optional separate root for metrics/; defaults to --repo-root",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "canonical repo-relative checkpoint path, e.g. "
            "attention/run-0001/checkpoints/step-010000"
        ),
    )
    parser.add_argument(
        "--evaluation-arm",
        default="all",
        choices=("all", *ARM_NAMES),
        help="operator used for inference; 'all' evaluates all four corners",
    )
    parser.add_argument(
        "--lengths",
        default=",".join(str(x) for x in DEFAULT_LENGTHS),
        help="comma-separated repeated-sequence half-lengths",
    )
    parser.add_argument(
        "--offsets",
        default=",".join(str(x) for x in DEFAULT_OFFSETS),
        help="comma-separated DAC source offsets",
    )
    parser.add_argument("--logical-batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="physical forward batch; 0 = auto from --max-forward-tokens",
    )
    parser.add_argument(
        "--max-forward-tokens",
        type=int,
        default=DEFAULT_MAX_FORWARD_TOKENS,
        help="auto microbatch heuristic: microB*(2L-1) <= this budget",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--behavior-only",
        action="store_true",
        help="skip DAC hooks and run only behavioral induction metrics",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-device", default="cpu")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help="synthetic token vocabulary; 0 = tokenizer.get_vocab_size()",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="disable strict checkpoint key/shape validation (not recommended)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail rather than overwrite an identical canonical result path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        checkpoint = CheckpointRef.from_repo_path(args.checkpoint)
        config = InductionRunConfig(
            lengths=_parse_csv_ints(args.lengths, name="lengths"),
            offsets=_parse_csv_ints(args.offsets, name="offsets"),
            logical_batch_size=args.logical_batch_size,
            num_batches=args.num_batches,
            seed=args.seed,
            micro_batch_size=(None if args.micro_batch_size == 0 else args.micro_batch_size),
            max_forward_tokens=args.max_forward_tokens,
            include_dac=not args.behavior_only,
            top_k=args.top_k,
        )

        artifacts = run_checkpoint(
            repo_root=args.repo_root,
            output_root=args.output_root or None,
            checkpoint=checkpoint,
            evaluation_arm=args.evaluation_arm,
            config=config,
            device=args.device,
            weights_device=args.weights_device,
            strict=not args.no_strict,
            overwrite=not args.no_overwrite,
            vocab_size=None if args.vocab_size == 0 else args.vocab_size,
        )

    except (ValueError, RuntimeError, FileNotFoundError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
        return 2

    print(
        json.dumps(
            {
                "checkpoint": checkpoint.as_dict(),
                "artifacts": [artifact.as_dict() for artifact in artifacts],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
