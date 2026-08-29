"""modal/train.py
Generic one-run training bridge for the nanochat attention-operator square.

A run is defined by exactly two scientific inputs:

    1. a canonical source checkpoint already stored in HF_REPO;
    2. a target operator arm: attention, AMAP, HMAP, or DMAP.

The destination name is automatic. For the requested target arm, this launcher
finds the next unused canonical run id:

    attention/run-0001/...
    AMAP/run-0001/...
    HMAP/run-0001/...
    DMAP/run-0001/...

and allocates ``run-XXXX`` = max(existing)+1.

Important semantics
-------------------
* Every invocation creates a NEW run. Nothing resumes or overwrites the source.
* Source and target arms may be arbitrary, including the same arm.
* The source portable ``model.safetensors`` is converted to a temporary raw
  ``.pt`` state_dict only because ``scripts.base_train --init-from-model``
  currently accepts nanochat raw state_dict files.
* ``base_train`` performs a strict key/shape warm start, then creates a FRESH
  optimizer and FRESH dataloader at local step 0.
* ``--save-every=-1`` is forced: nanochat saves only the final local checkpoint.
* No Hugging Face write happens while training is running.
* After the GPU training function returns successfully, a separate CPU Modal
  function converts the completed raw checkpoint to canonical safetensors,
  writes run/checkpoint provenance, and uploads the new run in ONE HF commit.
* Optimizer/dataloader state is deliberately not published. A later action that
  continues from this checkpoint is another weights-only run with a new run id.

Examples
--------
attention -> AMAP::

    modal run modal/train.py \
      --hf-repo owner/repo \
      --source-checkpoint attention/run-0001/checkpoints/step-000000 \
      --target-arm amap \
      --steps 20

DMAP -> AMAP::

    modal run modal/train.py \
      --hf-repo owner/repo \
      --source-checkpoint DMAP/run-0003/checkpoints/step-001000 \
      --target-arm amap \
      --steps 1000

AMAP -> AMAP (fresh run, not optimizer resume)::

    modal run modal/train.py \
      --hf-repo owner/repo \
      --source-checkpoint AMAP/run-0001/checkpoints/step-001000 \
      --target-arm amap \
      --steps 1000
"""

from __future__ import annotations

import sys


_RUNTIME = "--runtime" in sys.argv


if _RUNTIME:
    # -------------------------------------------------------------------------
    # Inner helper runtime: executes inside /root/nanochat/.venv.
    #
    # This file is run directly from modal/train.py, so add the repository root
    # explicitly. Do NOT make repo/modal a Python package; doing so would shadow
    # the installed ``modal`` SDK.
    # -------------------------------------------------------------------------
    from pathlib import Path as _BootstrapPath

    _REPO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import argparse
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path
    from typing import Any

    import torch

    from experiments.common.arms import get_arm
    from experiments.common.checkpoints import (
        CheckpointRef,
        load_state_dict_file,
        read_json,
        resolve_local_checkpoint,
        save_state_dict_safetensors,
        validate_checkpoint_metadata,
        write_json_atomic,
    )
    from experiments.common.provenance import (
        GraftInitialization,
        RunProvenance,
        SourceCheckpoint,
        build_checkpoint_metadata,
        read_run_provenance,
        validate_checkpoint_provenance,
        write_run_provenance,
    )


    def _utc_now_iso() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )


    def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


    def _json_line(value: dict[str, Any]) -> None:
        print("__NANOCHAT_JSON__" + json.dumps(value, sort_keys=True), flush=True)


    def _arm_info(arm_name: str) -> None:
        arm = get_arm(arm_name)
        _json_line(
            {
                "name": arm.name,
                "hf_folder": arm.hf_folder,
                "operator": arm.as_dict(),
            }
        )


    def _prepare_seed(
        *,
        staging_root: Path,
        source_checkpoint: str,
        output_pt: Path,
    ) -> None:
        """Validate a canonical source and emit a temporary raw state_dict .pt."""
        ref = CheckpointRef.from_repo_path(source_checkpoint)
        local = resolve_local_checkpoint(staging_root, ref)

        run = read_run_provenance(local.run_metadata_path)
        checkpoint_meta = read_json(local.metadata_path)
        validate_checkpoint_provenance(ref, checkpoint_meta, run)

        expected_sha = checkpoint_meta.get("model_sha256")
        actual_sha = _sha256(local.model_path)
        if expected_sha and str(expected_sha) != actual_sha:
            raise RuntimeError(
                "source model SHA-256 does not match checkpoint.json:\n"
                f"  checkpoint.json: {expected_sha}\n"
                f"  downloaded file: {actual_sha}"
            )

        state = load_state_dict_file(local.model_path, device="cpu")
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state), output_pt)

        _json_line(
            {
                "source_checkpoint": ref.as_dict(),
                "source_model_sha256": actual_sha,
                "raw_seed_pt": str(output_pt),
                "tensor_count": len(state),
            }
        )


    def _publish_canonical(
        *,
        source_checkpoint: str,
        source_run_json: Path,
        source_checkpoint_json: Path,
        trained_model_pt: Path,
        trained_meta_json: Path,
        target_arm_name: str,
        target_run_id: str,
        steps: int,
        hf_repo: str,
        source_revision: str,
        git_commit: str,
        output_root: Path,
    ) -> None:
        """Convert a completed nanochat run into one canonical final checkpoint."""
        if steps <= 0:
            raise ValueError("steps must be positive")

        source_ref = CheckpointRef.from_repo_path(source_checkpoint)
        source_run = read_run_provenance(source_run_json)
        source_checkpoint_meta = read_json(source_checkpoint_json)
        validate_checkpoint_metadata(source_ref, source_checkpoint_meta)

        if source_run.run_id != source_ref.run_id or source_run.arm != source_ref.arm:
            raise RuntimeError(
                "source run.json identity does not match source checkpoint path"
            )

        target_arm = get_arm(target_arm_name)
        target_ref = CheckpointRef(
            arm=target_arm.name,
            run_id=target_run_id,
            step=steps,
        )

        trainer_meta = read_json(trained_meta_json)
        if int(trainer_meta.get("step", -1)) != steps:
            raise RuntimeError(
                f"nanochat final meta step={trainer_meta.get('step')!r}; "
                f"expected {steps}"
            )

        user_config = trainer_meta.get("user_config")
        if not isinstance(user_config, dict):
            raise RuntimeError("nanochat final meta has no user_config object")

        if str(user_config.get("arm", "")).lower() != target_arm.name:
            raise RuntimeError(
                "nanochat final metadata target arm does not match requested arm: "
                f"{user_config.get('arm')!r} != {target_arm.name!r}"
            )

        operator_checks = {
            "attn_variant": target_arm.attn_variant,
            "hmap_beta": target_arm.beta,
            "hmap_alpha": target_arm.alpha,
        }
        for key, expected in operator_checks.items():
            got = user_config.get(key)
            if got != expected:
                raise RuntimeError(
                    f"nanochat final user_config.{key}={got!r}; "
                    f"expected canonical {expected!r} for {target_arm.name}"
                )

        model_config = trainer_meta.get("model_config")
        if not isinstance(model_config, dict):
            raise RuntimeError("nanochat final meta has no model_config object")

        # Load only after all metadata checks pass. Clone to independent compact
        # CPU storages before safetensors writing.
        trained_state = torch.load(
            trained_model_pt,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(trained_state, dict):
            raise RuntimeError("nanochat final model file is not a state_dict")

        compact_state: dict[str, torch.Tensor] = {}
        for key, value in trained_state.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"trained state_dict[{key!r}] is not a tensor"
                )
            compact_state[key] = value.detach().cpu().clone().contiguous()
        del trained_state

        model_out = output_root / Path(*target_ref.model_path.parts)
        save_state_dict_safetensors(
            compact_state,
            model_out,
            metadata={
                "format": "nanochat-operator-square",
                "trained_arm": target_arm.name,
                "run_id": target_run_id,
                "local_step": str(steps),
                "parent_checkpoint": source_ref.directory.as_posix(),
                "source_repo": hf_repo,
            },
        )
        del compact_state

        model_sha256 = _sha256(model_out)
        source_model_sha = source_checkpoint_meta.get("model_sha256")
        source = SourceCheckpoint(
            ref=source_ref,
            repo_id=hf_repo,
            revision=source_revision,
            model_sha256=(
                str(source_model_sha) if source_model_sha else None
            ),
        )

        same_arm = source_ref.arm == target_arm.name
        training_mode = "continued_pretrain" if same_arm else "recondition"

        if same_arm:
            lineage_event = {
                "event": "weights_only_continuation",
                "run_id": target_run_id,
                "source": source.as_dict(),
                "operator": target_arm.as_dict(),
                "optimizer_policy": "fresh",
                "dataloader_policy": "fresh",
            }
            notes = (
                f"Same-arm weights-only continuation from "
                f"{source_ref.directory.as_posix()}; fresh optimizer and "
                f"dataloader; {steps} local optimization steps."
            )
        else:
            lineage_event = {
                "event": "operator_graft",
                "run_id": target_run_id,
                "source": source.as_dict(),
                "operator_swap": {
                    "from": source_ref.arm,
                    "to": target_arm.name,
                },
                "target_operator": target_arm.as_dict(),
                "optimizer_policy": "fresh",
                "dataloader_policy": "fresh",
            }
            notes = (
                f"Operator graft {source_ref.arm}->{target_arm.name} from "
                f"{source_ref.directory.as_posix()}, followed by {steps} local "
                "optimization steps with fresh optimizer and dataloader."
            )

        created_at = _utc_now_iso()
        training = {
            "mode": training_mode,
            "source_checkpoint": source_ref.as_dict(),
            "source_trained_arm": source_ref.arm,
            "target_arm": target_arm.name,
            "fresh_optimizer": True,
            "fresh_dataloader": True,
            "local_start_step": 0,
            "local_end_step": steps,
            "optimizer_steps": steps,
            "total_batch_size": trainer_meta.get("total_batch_size"),
            "device_batch_size": trainer_meta.get("device_batch_size"),
            "max_seq_len": trainer_meta.get("max_seq_len"),
            "base_train_user_config": user_config,
        }

        run = RunProvenance(
            run_id=target_run_id,
            arm=target_arm.name,
            initialization=GraftInitialization(source=source),
            lineage=tuple(source_run.lineage) + (lineage_event,),
            model={
                "model_config": dict(model_config),
            },
            training=training,
            created_at=created_at,
            git_commit=git_commit or None,
            notes=notes,
        )

        run_json_out = output_root / Path(*target_ref.run_metadata_path.parts)
        checkpoint_json_out = output_root / Path(*target_ref.metadata_path.parts)

        write_run_provenance(run_json_out, run)

        total_batch_size = trainer_meta.get("total_batch_size")
        tokens_seen = None
        if isinstance(total_batch_size, int) and total_batch_size >= 0:
            tokens_seen = total_batch_size * steps

        val_bpb = trainer_meta.get("val_bpb")
        if not isinstance(val_bpb, (int, float)):
            val_bpb = None

        checkpoint_payload = build_checkpoint_metadata(
            target_ref,
            run,
            tokens_seen=tokens_seen,
            validation_bpb=val_bpb,
            model_sha256=model_sha256,
            created_at=created_at,
            extra={
                "local_optimizer_steps": steps,
                "training_mode": training_mode,
                "source_checkpoint": source_ref.as_dict(),
                "optimizer_state_published": False,
                "dataloader_state_published": False,
            },
        )
        write_json_atomic(checkpoint_json_out, checkpoint_payload)

        _json_line(
            {
                "target_checkpoint": target_ref.as_dict(),
                "target_run_dir": target_ref.run_dir.as_posix(),
                "model_sha256": model_sha256,
                "training_mode": training_mode,
                "source_checkpoint": source_ref.as_dict(),
                "files": [
                    target_ref.run_metadata_path.as_posix(),
                    target_ref.model_path.as_posix(),
                    target_ref.metadata_path.as_posix(),
                ],
            }
        )


    def runtime_main() -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command", required=True)

        arm_parser = sub.add_parser("arm-info")
        arm_parser.add_argument("--arm", required=True)

        seed_parser = sub.add_parser("prepare-seed")
        seed_parser.add_argument("--staging-root", required=True)
        seed_parser.add_argument("--source-checkpoint", required=True)
        seed_parser.add_argument("--output-pt", required=True)

        publish_parser = sub.add_parser("publish")
        publish_parser.add_argument("--source-checkpoint", required=True)
        publish_parser.add_argument("--source-run-json", required=True)
        publish_parser.add_argument("--source-checkpoint-json", required=True)
        publish_parser.add_argument("--trained-model-pt", required=True)
        publish_parser.add_argument("--trained-meta-json", required=True)
        publish_parser.add_argument("--target-arm", required=True)
        publish_parser.add_argument("--target-run-id", required=True)
        publish_parser.add_argument("--steps", required=True, type=int)
        publish_parser.add_argument("--hf-repo", required=True)
        publish_parser.add_argument("--source-revision", default="main")
        publish_parser.add_argument("--git-commit", default="")
        publish_parser.add_argument("--output-root", required=True)

        args = parser.parse_args()

        if args.command == "arm-info":
            _arm_info(args.arm)
            return

        if args.command == "prepare-seed":
            _prepare_seed(
                staging_root=Path(args.staging_root),
                source_checkpoint=args.source_checkpoint,
                output_pt=Path(args.output_pt),
            )
            return

        if args.command == "publish":
            _publish_canonical(
                source_checkpoint=args.source_checkpoint,
                source_run_json=Path(args.source_run_json),
                source_checkpoint_json=Path(args.source_checkpoint_json),
                trained_model_pt=Path(args.trained_model_pt),
                trained_meta_json=Path(args.trained_meta_json),
                target_arm_name=args.target_arm,
                target_run_id=args.target_run_id,
                steps=args.steps,
                hf_repo=args.hf_repo,
                source_revision=args.source_revision,
                git_commit=args.git_commit,
                output_root=Path(args.output_root),
            )
            return

        raise AssertionError(args.command)


    if __name__ == "__main__":
        sys.argv.remove("--runtime")
        runtime_main()


else:
    # -------------------------------------------------------------------------
    # Outer Modal / HF orchestration.
    # -------------------------------------------------------------------------
    import json
    import os
    from pathlib import Path, PurePosixPath
    import re
    import shutil
    import subprocess
    import tempfile
    from typing import Any
    import uuid

    import modal


    APP_NAME = "nanochat-train"
    REPO_DIR = "/root/nanochat"
    VENV = f"{REPO_DIR}/.venv"

    CACHE_DIR = "/root/.cache/nanochat-research"
    CACHE_VOLUME_NAME = "nanochat-research-cache"

    TOKENIZER_REPO = "karpathy/nanochat-d32"
    TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")

    app = modal.App(APP_NAME)

    cache_volume = modal.Volume.from_name(
        CACHE_VOLUME_NAME,
        create_if_missing=True,
    )

    if modal.is_local():
        hf_secret = modal.Secret.from_dict(
            {"HF_TOKEN": os.environ.get("HF_TOKEN")}
        )
    else:
        hf_secret = modal.Secret.from_dict({})

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "build-essential")
        .pip_install("huggingface_hub", "hf-xet")
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "curl --proto '=https' --tlsv1.2 -sSf "
            "https://sh.rustup.rs | sh -s -- -y",
        )
        .env(
            {
                "PATH": (
                    "/root/.cargo/bin:/root/.local/bin:/usr/local/bin:"
                    "/usr/bin:/bin"
                ),
                "OMP_NUM_THREADS": "1",
                "NANOCHAT_BASE_DIR": CACHE_DIR,
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor_cache",
                "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton_cache",
            }
        )
        .add_local_dir(
            ".",
            remote_path=REPO_DIR,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".mypy_cache",
                "wandb",
            ],
        )
        .run_commands(
            f"cd {REPO_DIR} && uv sync --extra gpu",
            f"cd {REPO_DIR} && "
            f"uv pip install --python {VENV}/bin/python safetensors",
        )
    )


    def _subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            REPO_DIR if not existing else REPO_DIR + os.pathsep + existing
        )
        return env


    def _run_streamed(
        cmd: list[str],
        *,
        log_path: Path | None = None,
    ) -> None:
        print("[modal/train] running:", " ".join(cmd), flush=True)

        log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_DIR,
                env=_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None

            for line in proc.stdout:
                print(line, end="", flush=True)
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()

            return_code = proc.wait()
        finally:
            if log_handle is not None:
                log_handle.close()

        if return_code != 0:
            raise RuntimeError(
                f"command failed with exit code {return_code}: "
                + " ".join(cmd)
            )


    def _run_capture_json(cmd: list[str]) -> dict[str, Any]:
        proc = subprocess.run(
            cmd,
            cwd=REPO_DIR,
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"command failed with exit code {proc.returncode}: "
                + " ".join(cmd)
            )

        prefix = "__NANOCHAT_JSON__"
        lines = [
            line[len(prefix):]
            for line in proc.stdout.splitlines()
            if line.startswith(prefix)
        ]
        if not lines:
            raise RuntimeError("helper command returned no structured JSON")
        value = json.loads(lines[-1])
        if not isinstance(value, dict):
            raise RuntimeError("helper JSON was not an object")
        return value


    def _normalize_hf_repo_id(value: str) -> str:
        repo_id = value.strip().rstrip("/")
        prefix = "https://huggingface.co/"
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix):]
        if repo_id.endswith(".git"):
            repo_id = repo_id[:-4]
        parts = [part for part in repo_id.split("/") if part]
        if len(parts) != 2:
            raise ValueError(
                "hf_repo must be 'owner/repo' or its Hugging Face URL"
            )
        return "/".join(parts)


    def _canonical_checkpoint_dir(value: str) -> PurePosixPath:
        raw = value.strip().replace("\\", "/")
        if not raw:
            raise ValueError("source_checkpoint must be non-empty")

        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_checkpoint must be a safe repo-relative path")
        if path.name in {"model.safetensors", "checkpoint.json"}:
            path = path.parent

        if len(path.parts) != 4 or path.parts[2] != "checkpoints":
            raise ValueError(
                "source_checkpoint must have shape "
                "'<ARM>/run-XXXX/checkpoints/step-XXXXXX'"
            )
        return path


    def _source_files(checkpoint: str) -> tuple[PurePosixPath, ...]:
        checkpoint_dir = _canonical_checkpoint_dir(checkpoint)
        run_dir = checkpoint_dir.parents[1]
        return (
            run_dir / "run.json",
            checkpoint_dir / "checkpoint.json",
            checkpoint_dir / "model.safetensors",
        )


    def _hf_download_into(
        *,
        repo_id: str,
        filename: str,
        destination: Path,
        token: str | None,
        revision: str,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            token=token,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        return destination


    def _ensure_tokenizer() -> None:
        tokenizer_dir = Path(CACHE_DIR) / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        for filename in TOKENIZER_FILES:
            _hf_download_into(
                repo_id=TOKENIZER_REPO,
                filename=filename,
                destination=tokenizer_dir / filename,
                token=None,
                revision="main",
            )
        print(
            f"[modal/train] tokenizer ready from {TOKENIZER_REPO} -> "
            f"{tokenizer_dir}",
            flush=True,
        )


    def _next_run_id(repo_files: list[str], target_folder: str) -> str:
        pattern = re.compile(
            rf"^{re.escape(target_folder)}/run-(\d{{4,}})/"
        )
        numbers = []
        for filename in repo_files:
            match = pattern.match(filename)
            if match:
                numbers.append(int(match.group(1)))

        next_number = max(numbers, default=0) + 1
        return f"run-{next_number:04d}"


    def _model_args_from_run_json(run_json: Path) -> dict[str, Any]:
        payload = json.loads(run_json.read_text())
        model = payload.get("model")
        if not isinstance(model, dict):
            raise RuntimeError("source run.json has no model object")
        config = model.get("model_config")
        if not isinstance(config, dict):
            raise RuntimeError(
                "source run.json has no model.model_config object"
            )

        required = (
            "sequence_len",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "window_pattern",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise RuntimeError(
                "source model_config is missing: " + ", ".join(missing)
            )

        n_layer = int(config["n_layer"])
        n_head = int(config["n_head"])
        n_kv_head = int(config["n_kv_head"])
        n_embd = int(config["n_embd"])
        if n_kv_head != n_head:
            raise RuntimeError(
                "operator-square training currently requires n_kv_head == n_head"
            )
        if n_embd % n_layer != 0 or n_embd % n_head != 0:
            raise RuntimeError(
                "cannot derive nanochat aspect_ratio/head_dim from source config"
            )

        return {
            "depth": n_layer,
            "aspect_ratio": n_embd // n_layer,
            "head_dim": n_embd // n_head,
            "max_seq_len": int(config["sequence_len"]),
            "window_pattern": str(config["window_pattern"]),
        }


    def _visible_gpu_count() -> int:
        proc = subprocess.run(
            [
                f"{VENV}/bin/python",
                "-c",
                "import torch; print(torch.cuda.device_count())",
            ],
            cwd=REPO_DIR,
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        try:
            return int(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f"could not determine visible GPU count from: {proc.stdout!r}"
            ) from exc


    @app.function(
        image=image,
        cpu=32,
        memory=131072,
        timeout=24 * 60 * 60,
        retries=0,
        scaledown_window=5,
        volumes={CACHE_DIR: cache_volume},
        secrets=[hf_secret],
    )
    def train_run(
        hf_repo: str,
        source_checkpoint: str,
        target_arm: str,
        steps: int = 20,
        num_gpus: int = 1,
        device_batch_size: int = 1,
        total_batch_size: int = -1,
        data_shards: int = 240,
        revision: str = "main",
        git_commit: str = "",
    ) -> dict[str, Any]:
        """GPU stage. Trains locally only; performs no HF writes."""
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if num_gpus not in {1, 2, 4, 8}:
            raise ValueError("num_gpus must be one of 1, 2, 4, 8")
        if device_batch_size <= 0:
            raise ValueError("device_batch_size must be > 0")
        if total_batch_size == 0 or total_batch_size < -1:
            raise ValueError("total_batch_size must be -1 or a positive integer")
        if data_shards <= 0:
            raise ValueError("data_shards must be > 0")

        hf_repo = _normalize_hf_repo_id(hf_repo)
        source_dir = _canonical_checkpoint_dir(source_checkpoint)
        source_checkpoint = source_dir.as_posix()

        token = os.environ.get("HF_TOKEN") or None
        if token is None:
            raise RuntimeError("HF_TOKEN is required")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.auth_check(
            repo_id=hf_repo,
            repo_type="model",
            token=token,
            write=True,
        )

        visible_gpus = _visible_gpu_count()
        if visible_gpus != num_gpus:
            raise RuntimeError(
                f"Modal provisioned {visible_gpus} visible GPU(s), "
                f"but num_gpus={num_gpus}"
            )

        cache_volume.reload()

        python = f"{VENV}/bin/python"
        helper = f"{REPO_DIR}/modal/train.py"

        arm_info = _run_capture_json(
            [
                python,
                helper,
                "--runtime",
                "arm-info",
                "--arm",
                target_arm,
            ]
        )
        target_arm = str(arm_info["name"])
        target_folder = str(arm_info["hf_folder"])

        repo_files = list(
            api.list_repo_files(
                repo_id=hf_repo,
                repo_type="model",
                revision=revision,
                token=token,
            )
        )
        target_run_id = _next_run_id(repo_files, target_folder)
        target_prefix = f"{target_folder}/{target_run_id}/"
        if any(path.startswith(target_prefix) for path in repo_files):
            raise RuntimeError(
                f"automatic target allocation collision: {target_prefix}"
            )

        print(
            f"[modal/train] source: {source_checkpoint}",
            flush=True,
        )
        print(
            f"[modal/train] target allocated: "
            f"{target_folder}/{target_run_id}/checkpoints/step-{steps:06d}",
            flush=True,
        )
        print(
            "[modal/train] policy: fresh optimizer + fresh dataloader; "
            "source is immutable",
            flush=True,
        )

        job_id = (
            f"{target_arm}-{target_run_id}-"
            f"{uuid.uuid4().hex[:10]}"
        )
        tmp_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir="/tmp"))
        staging_root = tmp_root / "source"
        staging_root.mkdir(parents=True, exist_ok=True)

        staged: dict[str, Path] = {}
        for relative in _source_files(source_checkpoint):
            destination = staging_root / Path(*relative.parts)
            _hf_download_into(
                repo_id=hf_repo,
                filename=relative.as_posix(),
                destination=destination,
                token=token,
                revision=revision,
            )
            staged[relative.name] = destination
            print(
                f"[modal/train] staged {relative.as_posix()}",
                flush=True,
            )

        source_run_json = staged["run.json"]
        source_checkpoint_json = staged["checkpoint.json"]
        model_args = _model_args_from_run_json(source_run_json)

        raw_seed_pt = tmp_root / "seed_state_dict.pt"
        _run_streamed(
            [
                python,
                helper,
                "--runtime",
                "prepare-seed",
                "--staging-root",
                str(staging_root),
                "--source-checkpoint",
                source_checkpoint,
                "--output-pt",
                str(raw_seed_pt),
            ]
        )

        _ensure_tokenizer()

        print(
            f"[modal/train] ensuring {data_shards} nanochat data shards...",
            flush=True,
        )
        _run_streamed(
            [
                python,
                "-m",
                "nanochat.dataset",
                "-n",
                str(data_shards),
            ]
        )
        # Persist expensive tokenizer/data downloads even if later training fails.
        cache_volume.commit()

        # Raw nanochat outputs live only on the Modal Volume and use a UUID tag.
        # They are never uploaded to Hugging Face.
        local_tag = f"transient-{job_id}"
        raw_checkpoint_dir = (
            Path(CACHE_DIR) / "base_checkpoints" / local_tag
        )
        if raw_checkpoint_dir.exists():
            shutil.rmtree(raw_checkpoint_dir)

        job_dir = Path(CACHE_DIR) / "train_jobs" / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "train.log"

        # Preserve the exact source metadata needed by the CPU publisher.
        shutil.copy2(source_run_json, job_dir / "source_run.json")
        shutil.copy2(
            source_checkpoint_json,
            job_dir / "source_checkpoint.json",
        )

        base_cmd = [
            "-m",
            "scripts.base_train",
            "--run=dummy",
            "--device-type=cuda",
            f"--depth={model_args['depth']}",
            f"--aspect-ratio={model_args['aspect_ratio']}",
            f"--head-dim={model_args['head_dim']}",
            f"--max-seq-len={model_args['max_seq_len']}",
            f"--window-pattern={model_args['window_pattern']}",
            f"--model-tag={local_tag}",
            f"--arm={target_arm}",
            f"--init-from-model={raw_seed_pt}",
            f"--num-iterations={steps}",
            "--save-every=-1",
            "--eval-every=-1",
            "--core-metric-every=-1",
            "--sample-every=-1",
            f"--device-batch-size={device_batch_size}",
            f"--total-batch-size={total_batch_size}",
        ]

        if num_gpus == 1:
            cmd = [python, *base_cmd]
        else:
            cmd = [
                f"{VENV}/bin/torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                *base_cmd,
            ]

        print(
            "[modal/train] HF publication is disabled during GPU training; "
            "only the final local checkpoint will be staged afterward.",
            flush=True,
        )
        _run_streamed(cmd, log_path=log_path)

        final_model = raw_checkpoint_dir / f"model_{steps:06d}.pt"
        final_meta = raw_checkpoint_dir / f"meta_{steps:06d}.json"
        missing = [
            path for path in (final_model, final_meta)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "training completed without expected final checkpoint files:\n  "
                + "\n  ".join(str(path) for path in missing)
            )

        job_payload = {
            "job_id": job_id,
            "hf_repo": hf_repo,
            "revision": revision,
            "source_checkpoint": source_checkpoint,
            "target_arm": target_arm,
            "target_folder": target_folder,
            "target_run_id": target_run_id,
            "steps": steps,
            "raw_checkpoint_dir": str(raw_checkpoint_dir),
            "trained_model_pt": str(final_model),
            "trained_meta_json": str(final_meta),
            "source_run_json": str(job_dir / "source_run.json"),
            "source_checkpoint_json": str(
                job_dir / "source_checkpoint.json"
            ),
            "train_log": str(log_path),
            "git_commit": git_commit,
        }
        (job_dir / "job.json").write_text(
            json.dumps(job_payload, indent=2, sort_keys=True) + "\n"
        )

        # This commit is to the Modal Volume only, not Hugging Face.
        cache_volume.commit()

        print(
            f"[modal/train] GPU training complete: local step {steps}",
            flush=True,
        )
        print(
            "[modal/train] no Hugging Face checkpoint has been written yet",
            flush=True,
        )
        return job_payload


    @app.function(
        image=image,
        cpu=16,
        memory=131072,
        timeout=6 * 60 * 60,
        retries=0,
        scaledown_window=5,
        volumes={CACHE_DIR: cache_volume},
        secrets=[hf_secret],
    )
    def publish_run(job: dict[str, Any]) -> dict[str, Any]:
        """CPU stage. Canonicalizes and uploads only after GPU success."""
        token = os.environ.get("HF_TOKEN") or None
        if token is None:
            raise RuntimeError("HF_TOKEN is required")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        cache_volume.reload()

        hf_repo = _normalize_hf_repo_id(str(job["hf_repo"]))
        target_folder = str(job["target_folder"])
        target_run_id = str(job["target_run_id"])
        steps = int(job["steps"])
        target_prefix = f"{target_folder}/{target_run_id}/"

        # Re-check immediately before the one-and-only HF write. This protects
        # against another process allocating the same run while training.
        repo_files = list(
            api.list_repo_files(
                repo_id=hf_repo,
                repo_type="model",
                revision=str(job["revision"]),
                token=token,
            )
        )
        collisions = [
            path for path in repo_files
            if path.startswith(target_prefix)
        ]
        if collisions:
            raise RuntimeError(
                "target run appeared while training; refusing to overwrite:\n  "
                + "\n  ".join(collisions[:20])
            )

        for key in (
            "trained_model_pt",
            "trained_meta_json",
            "source_run_json",
            "source_checkpoint_json",
        ):
            path = Path(str(job[key]))
            if not path.is_file():
                raise RuntimeError(
                    f"completed training artifact vanished from Volume: {path}"
                )

        output_root = Path(
            tempfile.mkdtemp(prefix="nanochat-publish-", dir="/tmp")
        )

        python = f"{VENV}/bin/python"
        helper = f"{REPO_DIR}/modal/train.py"
        publish_info = _run_capture_json(
            [
                python,
                helper,
                "--runtime",
                "publish",
                "--source-checkpoint",
                str(job["source_checkpoint"]),
                "--source-run-json",
                str(job["source_run_json"]),
                "--source-checkpoint-json",
                str(job["source_checkpoint_json"]),
                "--trained-model-pt",
                str(job["trained_model_pt"]),
                "--trained-meta-json",
                str(job["trained_meta_json"]),
                "--target-arm",
                str(job["target_arm"]),
                "--target-run-id",
                target_run_id,
                "--steps",
                str(steps),
                "--hf-repo",
                hf_repo,
                "--source-revision",
                str(job["revision"]),
                "--git-commit",
                str(job.get("git_commit") or ""),
                "--output-root",
                str(output_root),
            ]
        )

        run_folder = output_root / target_folder / target_run_id
        if not run_folder.is_dir():
            raise RuntimeError(
                f"canonical publisher did not create {run_folder}"
            )

        local_files = sorted(
            path.relative_to(output_root).as_posix()
            for path in run_folder.rglob("*")
            if path.is_file()
        )
        expected_count = 3
        if len(local_files) != expected_count:
            raise RuntimeError(
                "canonical run must contain exactly run.json, "
                "model.safetensors, checkpoint.json; got:\n  "
                + "\n  ".join(local_files)
            )

        # ONE Hugging Face commit, only now that training and canonicalization
        # have both completed successfully.
        api.upload_folder(
            repo_id=hf_repo,
            repo_type="model",
            revision=str(job["revision"]),
            token=token,
            folder_path=str(run_folder),
            path_in_repo=f"{target_folder}/{target_run_id}",
            commit_message=(
                f"{target_folder}/{target_run_id}: "
                f"{job['source_checkpoint']} -> {job['target_arm']} "
                f"for {steps} steps"
            ),
        )

        print(
            f"[modal/train] published canonical run: "
            f"{target_folder}/{target_run_id}",
            flush=True,
        )
        for relative in local_files:
            print(f"[modal/train]   {relative}", flush=True)

        # Raw trainer files are implementation details. Remove them only after
        # Hugging Face accepted the canonical run.
        raw_checkpoint_dir = Path(str(job["raw_checkpoint_dir"]))
        job_dir = Path(str(job["source_run_json"])).parent
        if raw_checkpoint_dir.exists():
            shutil.rmtree(raw_checkpoint_dir)
        if job_dir.exists():
            shutil.rmtree(job_dir)
        cache_volume.commit()

        return {
            "ok": True,
            "hf_repo": hf_repo,
            "source_checkpoint": job["source_checkpoint"],
            "target_arm": job["target_arm"],
            "target_run_id": target_run_id,
            "target_checkpoint": (
                f"{target_folder}/{target_run_id}/checkpoints/"
                f"step-{steps:06d}"
            ),
            "steps": steps,
            "training_mode": publish_info["training_mode"],
            "model_sha256": publish_info["model_sha256"],
            "uploaded_files": local_files,
        }


    @app.local_entrypoint()
    def main(
        hf_repo: str,
        source_checkpoint: str,
        target_arm: str,
        steps: int = 20,
        gpu: str = "H100",
        num_gpus: int = 1,
        device_batch_size: int = 1,
        total_batch_size: int = -1,
        data_shards: int = 240,
        revision: str = "main",
    ) -> None:
        """GitHub/local entrypoint: GPU train first, CPU publish second."""
        if num_gpus not in {1, 2, 4, 8}:
            raise ValueError("num_gpus must be one of 1, 2, 4, 8")

        gpu_request = gpu if num_gpus == 1 else f"{gpu}:{num_gpus}"
        print(
            f"[modal/train] requesting Modal GPU resource: {gpu_request}",
            flush=True,
        )

        # Dynamic GPU options let one generic workflow choose 1/2/4/8 GPUs.
        trained = train_run.with_options(gpu=gpu_request).remote(
            hf_repo=hf_repo,
            source_checkpoint=source_checkpoint,
            target_arm=target_arm,
            steps=steps,
            num_gpus=num_gpus,
            device_batch_size=device_batch_size,
            total_batch_size=total_batch_size,
            data_shards=data_shards,
            revision=revision,
            git_commit=os.environ.get("GITHUB_SHA", ""),
        )

        # This call cannot happen unless the GPU function returned successfully.
        result = publish_run.remote(trained)
        print(json.dumps(result, indent=2, sort_keys=True))
