"""experiments/simulation/run.py
Native-operator milestone evaluation for one continuous operator-square simulation.

This module is scientific code only: no Modal imports, no Hugging Face network
access, and no repository ids. The caller provides a local canonical checkpoint
mirror and this runner evaluates that checkpoint under the operator it was
trained for.

Each milestone measures:

* validation bits-per-byte (BPB), using nanochat's validation loader;
* CORE, using nanochat's base evaluator;
* repeated-random induction behavior plus DAC, by reusing the validated
  experiments.induction implementation.

One immutable JSON record is written per milestone under:

    metrics/simulation/<ARM>/run-XXXX/step-XXXXXX.json

The outer Modal harness is free to discard intermediate model weights after this
record and the next rolling resume checkpoint are safely persisted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.common.checkpoints import CheckpointRef
from experiments.common.model import LoadedModel, load_model
from experiments.induction.dac import DEFAULT_OFFSETS
from experiments.induction.run import (
    DEFAULT_LENGTHS,
    InductionRunConfig,
    evaluate_loaded_model,
)
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_core


SCHEMA_VERSION = 1
DEFAULT_EVAL_TOKENS = 4_194_304


@dataclass(frozen=True, slots=True)
class SimulationEvalConfig:
    """Scientific protocol held fixed across all trajectory milestones."""

    eval_tokens: int = DEFAULT_EVAL_TOKENS
    eval_batch_size: int = 1
    core_max_per_task: int = 500
    induction_lengths: tuple[int, ...] = DEFAULT_LENGTHS
    induction_offsets: tuple[int, ...] = DEFAULT_OFFSETS
    induction_logical_batch_size: int = 16
    induction_micro_batch_size: int | None = None
    induction_max_forward_tokens: int = 8192
    induction_num_batches: int = 4
    induction_seed: int = 1234
    induction_top_k: int = 8

    def __post_init__(self) -> None:
        positive = (
            "eval_tokens",
            "eval_batch_size",
            "core_max_per_task",
            "induction_logical_batch_size",
            "induction_max_forward_tokens",
            "induction_num_batches",
            "induction_top_k",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.induction_micro_batch_size is not None:
            value = self.induction_micro_batch_size
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "induction_micro_batch_size must be None or a positive integer"
                )

    def induction_config(self) -> InductionRunConfig:
        return InductionRunConfig(
            lengths=self.induction_lengths,
            offsets=self.induction_offsets,
            logical_batch_size=self.induction_logical_batch_size,
            num_batches=self.induction_num_batches,
            seed=self.induction_seed,
            micro_batch_size=self.induction_micro_batch_size,
            max_forward_tokens=self.induction_max_forward_tokens,
            include_dac=True,
            top_k=self.induction_top_k,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation": {
                "requested_tokens": self.eval_tokens,
                "device_batch_size": self.eval_batch_size,
            },
            "core": {"max_per_task": self.core_max_per_task},
            "induction": self.induction_config().experiment_config(),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


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


def _strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        return _strict_json(data.item() if data.numel() == 1 else data.tolist())
    if hasattr(value, "item"):
        try:
            return _strict_json(value.item())
        except Exception:
            pass
    raise TypeError(f"cannot serialize {type(value).__name__} to strict JSON")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            _strict_json(dict(payload)),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    tmp.replace(path)
    return path


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def _validation_bpb(
    loaded: LoadedModel,
    *,
    config: SimulationEvalConfig,
    device: str | torch.device,
) -> tuple[float, int, int]:
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    sequence_len = int(loaded.model_config.sequence_len)
    tokens_per_step = config.eval_batch_size * sequence_len
    eval_steps = max(1, config.eval_tokens // tokens_per_step)
    actual_tokens = eval_steps * tokens_per_step
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        config.eval_batch_size,
        sequence_len,
        split="val",
        device=device,
    )
    loaded.model.eval()
    value = evaluate_bpb(loaded.model, loader, eval_steps, token_bytes)
    return float(value), int(eval_steps), int(actual_tokens)


@torch.inference_mode()
def _core(
    loaded: LoadedModel,
    *,
    config: SimulationEvalConfig,
    device: str | torch.device,
) -> dict[str, Any]:
    tokenizer = get_tokenizer()
    loaded.model.eval()
    result = evaluate_core(
        loaded.model,
        tokenizer,
        device,
        max_per_task=config.core_max_per_task,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("evaluate_core returned a non-mapping result")
    return dict(result)


def evaluate_milestone(
    *,
    repo_root: str | Path,
    checkpoint: CheckpointRef,
    output_root: str | Path,
    config: SimulationEvalConfig | None = None,
    device: str | torch.device = "cuda",
    weights_device: str = "cpu",
    strict: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Evaluate BPB, CORE, induction behavior and DAC for one milestone."""
    config = config or SimulationEvalConfig()
    repo_root = Path(repo_root)
    output_root = Path(output_root)

    loaded: LoadedModel | None = None
    try:
        loaded = load_model(
            repo_root,
            checkpoint,
            evaluation_arm=checkpoint.arm,
            device=device,
            weights_device=weights_device,
            strict=strict,
            eval_mode=True,
        )
        if loaded.trained_arm.name != loaded.evaluation_arm.name:
            raise RuntimeError("simulation milestone must use the native operator")

        print(
            f"[simulation/eval] arm={loaded.evaluation_arm.name} "
            f"step={checkpoint.step:06d}",
            flush=True,
        )

        bpb, bpb_steps, actual_eval_tokens = _validation_bpb(
            loaded,
            config=config,
            device=device,
        )
        print(
            f"[simulation/eval] validation_bpb={bpb:.6f} "
            f"tokens={actual_eval_tokens:,}",
            flush=True,
        )
        _clear_cuda()

        core = _core(loaded, config=config, device=device)
        if isinstance(core.get("core_metric"), (float, int)):
            print(
                f"[simulation/eval] core={float(core['core_metric']):.6f}",
                flush=True,
            )
        else:
            print("[simulation/eval] CORE complete", flush=True)
        _clear_cuda()

        vocab_size = int(get_tokenizer().get_vocab_size())
        induction = evaluate_loaded_model(
            loaded,
            vocab_size=vocab_size,
            config=config.induction_config(),
            device=device,
        )
        _clear_cuda()

        metadata = loaded.checkpoint_metadata
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "simulation",
            "created_at": _utc_now_iso(),
            "arm": loaded.trained_arm.name,
            "run_id": checkpoint.run_id,
            "step": checkpoint.step,
            "tokens_seen": metadata.get("tokens_seen"),
            "checkpoint": checkpoint.as_dict(),
            "model_sha256": metadata.get("model_sha256"),
            "trained_operator": loaded.trained_arm.as_dict(),
            "evaluation_operator": loaded.evaluation_arm.as_dict(),
            "config": {
                **config.as_dict(),
                "validation": {
                    **config.as_dict()["validation"],
                    "eval_steps": bpb_steps,
                    "actual_tokens": actual_eval_tokens,
                },
            },
            "metrics": {
                "validation_bpb": bpb,
                "core": core,
                "induction": induction,
            },
        }

        output = (
            output_root
            / "metrics"
            / "simulation"
            / loaded.trained_arm.hf_folder
            / checkpoint.run_id
            / f"step-{checkpoint.step:06d}.json"
        )
        _write_json_atomic(output, payload)
        print(f"[simulation/eval] wrote {output}", flush=True)
        return payload, output
    finally:
        if loaded is not None:
            del loaded
        _clear_cuda()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one continuous simulation trajectory milestone."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--core-max-per-task", type=int, default=500)
    parser.add_argument(
        "--induction-lengths",
        default=",".join(str(value) for value in DEFAULT_LENGTHS),
    )
    parser.add_argument(
        "--induction-offsets",
        default=",".join(str(value) for value in DEFAULT_OFFSETS),
    )
    parser.add_argument("--induction-logical-batch-size", type=int, default=16)
    parser.add_argument("--induction-micro-batch-size", type=int, default=0)
    parser.add_argument("--induction-max-forward-tokens", type=int, default=8192)
    parser.add_argument("--induction-num-batches", type=int, default=4)
    parser.add_argument("--induction-seed", type=int, default=1234)
    parser.add_argument("--induction-top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-device", default="cpu")
    parser.add_argument("--no-strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        checkpoint = CheckpointRef.from_repo_path(args.checkpoint)
        config = SimulationEvalConfig(
            eval_tokens=args.eval_tokens,
            eval_batch_size=args.eval_batch_size,
            core_max_per_task=args.core_max_per_task,
            induction_lengths=_parse_csv_ints(
                args.induction_lengths, name="induction_lengths"
            ),
            induction_offsets=_parse_csv_ints(
                args.induction_offsets, name="induction_offsets"
            ),
            induction_logical_batch_size=args.induction_logical_batch_size,
            induction_micro_batch_size=(
                None
                if args.induction_micro_batch_size == 0
                else args.induction_micro_batch_size
            ),
            induction_max_forward_tokens=args.induction_max_forward_tokens,
            induction_num_batches=args.induction_num_batches,
            induction_seed=args.induction_seed,
            induction_top_k=args.induction_top_k,
        )
        payload, output = evaluate_milestone(
            repo_root=args.repo_root,
            checkpoint=checkpoint,
            output_root=args.output_root,
            config=config,
            device=args.device,
            weights_device=args.weights_device,
            strict=not args.no_strict,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "arm": payload["arm"],
                "run_id": payload["run_id"],
                "step": payload["step"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
