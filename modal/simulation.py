"""modal/simulation.py
Continuous single-arm trajectory runner for the nanochat operator square.

One invocation selects exactly one target arm and follows one fixed optimization
trajectory from the canonical source checkpoint. The intended source is the
untouched Karpathy bootstrap checkpoint already stored in the shared model repo:

    attention/run-0001/checkpoints/step-000000

Protocol for a 10k / 1k simulation:

    step 0:     source weights + target operator, zero optimizer steps
                -> BPB + CORE + induction/DAC
    step 1000:  resume-continuous training state
                -> BPB + CORE + induction/DAC
    ...
    step 10000: final BPB + CORE + induction/DAC

The training trajectory is continuous. A private ephemeral copy of
``scripts/base_train.py`` adds only ``--stop-after-step``. Every subprocess is
still launched with the SAME ``--num-iterations=<horizon>`` and resumes the
previous optimizer + dataloader state. Therefore the LR, Muon momentum, and
weight-decay schedules always see the full horizon and are not restarted at
milestones.

Intermediate weights are rolling recovery state on the Modal Volume: once the
next milestone checkpoint and metrics are durable, the previous large raw
checkpoint is deleted. Metric JSON files are immutable. No Hugging Face model
checkpoint is written during training. Only after the full trajectory and final
evaluation succeed is the final canonical model plus all milestone metrics
published.

Examples:

    modal run modal/simulation.py --hf-repo owner/repo --arm hmap
    modal run modal/simulation.py --hf-repo owner/repo --arm dmap
    modal run modal/simulation.py --hf-repo owner/repo --mode recover \
        --arm hmap --pending-job latest
"""

from __future__ import annotations

import sys


_RUNTIME = "--runtime" in sys.argv


if _RUNTIME:
    # ---------------------------------------------------------------------
    # Inner helper runtime. This executes under the repo uv venv, where torch,
    # safetensors, nanochat, and experiments are installed.
    # ---------------------------------------------------------------------
    import argparse
    from dataclasses import asdict, replace
    import hashlib
    import json
    from pathlib import Path
    import shutil
    from typing import Any

    import torch

    from experiments.common.arms import get_arm
    from experiments.common.checkpoints import (
        CheckpointRef,
        load_state_dict_file,
        read_json,
        resolve_local_checkpoint,
        save_state_dict_safetensors,
        write_json_atomic,
    )
    from experiments.common.model import trained_model_config
    from experiments.common.provenance import (
        build_checkpoint_metadata,
        new_graft_run,
        read_run_provenance,
        validate_checkpoint_provenance,
        write_run_provenance,
    )


    def _sha256(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


    def _emit(payload: dict[str, Any]) -> None:
        print("__NANOCHAT_JSON__" + json.dumps(payload, sort_keys=True), flush=True)


    def _arm_info(name: str) -> None:
        arm = get_arm(name)
        _emit(
            {
                "name": arm.name,
                "hf_folder": arm.hf_folder,
                "operator": arm.as_dict(),
            }
        )


    def _prepare_seed(
        *,
        source_root: Path,
        source_checkpoint: str,
        output_pt: Path,
    ) -> None:
        ref = CheckpointRef.from_repo_path(source_checkpoint)
        local = resolve_local_checkpoint(source_root, ref)
        run = read_run_provenance(local.run_metadata_path)
        metadata = read_json(local.metadata_path)
        validate_checkpoint_provenance(ref, metadata, run)

        expected_sha = metadata.get("model_sha256")
        actual_sha = _sha256(local.model_path)
        if expected_sha and str(expected_sha) != actual_sha:
            raise RuntimeError(
                "source model SHA mismatch: "
                f"checkpoint.json={expected_sha!r}, actual={actual_sha!r}"
            )

        state = load_state_dict_file(local.model_path, device="cpu")
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state), output_pt)
        _emit(
            {
                "source_checkpoint": ref.as_dict(),
                "source_model_sha256": actual_sha,
                "output_pt": str(output_pt),
                "tensor_count": len(state),
            }
        )


    def _create_run(
        *,
        source_root: Path,
        source_checkpoint: str,
        target_arm_name: str,
        target_run_id: str,
        hf_repo: str,
        revision: str,
        horizon: int,
        milestone_every: int,
        total_batch_size: int,
        device_batch_size: int,
        data_shards: int,
        git_commit: str,
        output_run_json: Path,
    ) -> None:
        source_ref = CheckpointRef.from_repo_path(source_checkpoint)
        source_local = resolve_local_checkpoint(source_root, source_ref)
        source_run = read_run_provenance(source_local.run_metadata_path)
        source_metadata = read_json(source_local.metadata_path)
        validate_checkpoint_provenance(source_ref, source_metadata, source_run)

        source_sha = source_metadata.get("model_sha256") or _sha256(
            source_local.model_path
        )
        target_arm = get_arm(target_arm_name)
        source_config = trained_model_config(source_run)
        target_config = replace(source_config, **target_arm.config_overrides())

        training = {
            "mode": (
                "continued_pretrain"
                if source_ref.arm == target_arm.name
                else "recondition"
            ),
            "simulation": True,
            "source_checkpoint": source_ref.as_dict(),
            "source_trained_arm": source_ref.arm,
            "target_arm": target_arm.name,
            "fresh_optimizer_at_step_0": True,
            "fresh_dataloader_at_step_0": True,
            "resume_optimizer_between_milestones": True,
            "resume_dataloader_between_milestones": True,
            "fixed_schedule_horizon": horizon,
            "milestone_every": milestone_every,
            "optimizer_steps": horizon,
            "total_batch_size": total_batch_size,
            "device_batch_size": device_batch_size,
            "data_shards": data_shards,
            "intermediate_checkpoint_policy": "rolling_modal_volume",
            "hf_checkpoint_policy": "final_only_after_full_success",
            "milestone_metrics": ["validation_bpb", "core", "induction", "dac"],
        }

        run = new_graft_run(
            run_id=target_run_id,
            target_arm=target_arm.name,
            source_checkpoint=source_ref,
            source_run=source_run,
            source_repo_id=hf_repo,
            source_revision=revision,
            source_model_sha256=str(source_sha),
            model={"model_config": asdict(target_config)},
            training=training,
            git_commit=git_commit or None,
            notes=(
                f"Continuous {horizon}-step {target_arm.name} simulation from "
                f"{source_ref.directory.as_posix()}; native BPB/CORE/induction+DAC "
                f"at step 0 and every {milestone_every} steps. Intermediate weights "
                "are rolling recovery state; final HF model only."
            ),
        )
        output_run_json.parent.mkdir(parents=True, exist_ok=True)
        write_run_provenance(output_run_json, run)
        _emit(
            {
                "run_id": run.run_id,
                "arm": run.arm,
                "hf_folder": target_arm.hf_folder,
                "training_mode": training["mode"],
                "run_json": str(output_run_json),
            }
        )


    def _stage_milestone(
        *,
        run_json: Path,
        output_root: Path,
        step: int,
        horizon: int,
        milestone_every: int,
        total_batch_size: int,
        model_safetensors: Path | None,
        model_pt: Path | None,
    ) -> None:
        if (model_safetensors is None) == (model_pt is None):
            raise ValueError(
                "exactly one of --model-safetensors or --model-pt is required"
            )

        run = read_run_provenance(run_json)
        ref = CheckpointRef(arm=run.arm, run_id=run.run_id, step=step)
        target_model = output_root / Path(*ref.model_path.parts)
        target_model.parent.mkdir(parents=True, exist_ok=True)

        if model_safetensors is not None:
            shutil.copy2(model_safetensors, target_model)
        else:
            assert model_pt is not None
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            save_state_dict_safetensors(state, target_model)
            del state

        target_run_json = output_root / Path(*ref.run_metadata_path.parts)
        target_run_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_json, target_run_json)

        model_sha = _sha256(target_model)
        metadata = build_checkpoint_metadata(
            ref,
            run,
            tokens_seen=step * total_batch_size,
            model_sha256=model_sha,
            extra={
                "simulation": True,
                "simulation_horizon": horizon,
                "milestone_every": milestone_every,
                "rolling_intermediate": step < horizon,
                "optimizer_state_published": False,
                "dataloader_state_published": False,
            },
        )
        target_metadata = output_root / Path(*ref.metadata_path.parts)
        write_json_atomic(target_metadata, metadata)
        _emit(
            {
                "checkpoint": ref.as_dict(),
                "checkpoint_path": ref.directory.as_posix(),
                "model_sha256": model_sha,
                "run_json": ref.run_metadata_path.as_posix(),
                "model_file": ref.model_path.as_posix(),
                "checkpoint_json": ref.metadata_path.as_posix(),
            }
        )


    def runtime_main() -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)

        command = subparsers.add_parser("arm-info")
        command.add_argument("--arm", required=True)

        command = subparsers.add_parser("prepare-seed")
        command.add_argument("--source-root", required=True)
        command.add_argument("--source-checkpoint", required=True)
        command.add_argument("--output-pt", required=True)

        command = subparsers.add_parser("create-run")
        command.add_argument("--source-root", required=True)
        command.add_argument("--source-checkpoint", required=True)
        command.add_argument("--target-arm", required=True)
        command.add_argument("--target-run-id", required=True)
        command.add_argument("--hf-repo", required=True)
        command.add_argument("--revision", default="main")
        command.add_argument("--horizon", type=int, required=True)
        command.add_argument("--milestone-every", type=int, required=True)
        command.add_argument("--total-batch-size", type=int, required=True)
        command.add_argument("--device-batch-size", type=int, required=True)
        command.add_argument("--data-shards", type=int, required=True)
        command.add_argument("--git-commit", default="")
        command.add_argument("--output-run-json", required=True)

        command = subparsers.add_parser("stage-milestone")
        command.add_argument("--run-json", required=True)
        command.add_argument("--output-root", required=True)
        command.add_argument("--step", type=int, required=True)
        command.add_argument("--horizon", type=int, required=True)
        command.add_argument("--milestone-every", type=int, required=True)
        command.add_argument("--total-batch-size", type=int, required=True)
        group = command.add_mutually_exclusive_group(required=True)
        group.add_argument("--model-safetensors")
        group.add_argument("--model-pt")

        args = parser.parse_args()
        if args.command == "arm-info":
            _arm_info(args.arm)
        elif args.command == "prepare-seed":
            _prepare_seed(
                source_root=Path(args.source_root),
                source_checkpoint=args.source_checkpoint,
                output_pt=Path(args.output_pt),
            )
        elif args.command == "create-run":
            _create_run(
                source_root=Path(args.source_root),
                source_checkpoint=args.source_checkpoint,
                target_arm_name=args.target_arm,
                target_run_id=args.target_run_id,
                hf_repo=args.hf_repo,
                revision=args.revision,
                horizon=args.horizon,
                milestone_every=args.milestone_every,
                total_batch_size=args.total_batch_size,
                device_batch_size=args.device_batch_size,
                data_shards=args.data_shards,
                git_commit=args.git_commit,
                output_run_json=Path(args.output_run_json),
            )
        elif args.command == "stage-milestone":
            _stage_milestone(
                run_json=Path(args.run_json),
                output_root=Path(args.output_root),
                step=args.step,
                horizon=args.horizon,
                milestone_every=args.milestone_every,
                total_batch_size=args.total_batch_size,
                model_safetensors=(
                    Path(args.model_safetensors) if args.model_safetensors else None
                ),
                model_pt=Path(args.model_pt) if args.model_pt else None,
            )
        else:
            raise AssertionError(args.command)


    if __name__ == "__main__":
        sys.argv.remove("--runtime")
        runtime_main()


else:
    # ---------------------------------------------------------------------
    # Outer Modal / Hugging Face orchestration. Torch work remains inside the
    # repo venv or the base_train subprocess.
    # ---------------------------------------------------------------------
    import json
    import os
    from pathlib import Path, PurePosixPath
    import re
    import shutil
    import subprocess
    import tempfile
    import time
    from typing import Any
    import uuid

    import modal


    APP_NAME = "nanochat-simulation"
    REPO_DIR = "/root/nanochat"
    VENV = f"{REPO_DIR}/.venv"
    CACHE_DIR = "/root/.cache/nanochat-research"
    TOKENIZER_REPO = "karpathy/nanochat-d32"
    TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")

    app = modal.App(APP_NAME)
    cache_volume = modal.Volume.from_name(
        "nanochat-research-cache", create_if_missing=True
    )

    if modal.is_local():
        hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])
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
                "HF_XET_HIGH_PERFORMANCE": "1",
            }
        )
        .add_local_dir(
            ".",
            remote_path=REPO_DIR,
            copy=True,
            ignore=[".git", ".venv", "__pycache__", "*.pyc", ".github"],
        )
        .run_commands(
            f"cd {REPO_DIR} && uv sync --extra gpu",
            (
                f"uv pip install --python {VENV}/bin/python safetensors "
                f"&& {VENV}/bin/python -c \"import safetensors; "
                "print('safetensors', safetensors.__version__)\""
            ),
        )
    )


    def _env() -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["NANOCHAT_BASE_DIR"] = CACHE_DIR
        env["TORCHINDUCTOR_CACHE_DIR"] = f"{CACHE_DIR}/inductor_cache"
        env["TRITON_CACHE_DIR"] = f"{CACHE_DIR}/triton_cache"
        return env


    def _run_streamed(cmd: list[str], *, log_path: Path | None = None) -> None:
        print("[modal/simulation] running:", " ".join(cmd), flush=True)
        log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_DIR,
                env=_env(),
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
            returncode = proc.wait()
        finally:
            if log_handle is not None:
                log_handle.close()
        if returncode != 0:
            raise RuntimeError(
                f"command failed with exit code {returncode}: " + " ".join(cmd)
            )


    def _run_json(cmd: list[str]) -> dict[str, Any]:
        proc = subprocess.run(
            cmd,
            cwd=REPO_DIR,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"command failed with exit code {proc.returncode}: " + " ".join(cmd)
            )
        prefix = "__NANOCHAT_JSON__"
        payloads = [
            json.loads(line[len(prefix) :])
            for line in proc.stdout.splitlines()
            if line.startswith(prefix)
        ]
        if not payloads:
            raise RuntimeError("helper subprocess produced no structured payload")
        return payloads[-1]


    def _normalize_repo(value: str) -> str:
        value = value.strip().rstrip("/")
        prefix = "https://huggingface.co/"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        if value.endswith(".git"):
            value = value[:-4]
        parts = [part for part in value.split("/") if part]
        if len(parts) != 2:
            raise ValueError("hf_repo must be owner/repo or its Hugging Face URL")
        return "/".join(parts)


    def _canonical_checkpoint(value: str) -> str:
        path = PurePosixPath(value.strip().replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_checkpoint must be repo-relative")
        if path.name in {"model.safetensors", "checkpoint.json"}:
            path = path.parent
        if len(path.parts) != 4 or path.parts[2] != "checkpoints":
            raise ValueError(
                "source_checkpoint must be <ARM>/run-XXXX/checkpoints/step-XXXXXX"
            )
        if re.fullmatch(r"run-\d{4,}", path.parts[1]) is None:
            raise ValueError("source_checkpoint has invalid run id")
        if re.fullmatch(r"step-\d{6,}", path.parts[3]) is None:
            raise ValueError("source_checkpoint has invalid step")
        return path.as_posix()


    def _source_paths(checkpoint: str) -> tuple[PurePosixPath, ...]:
        directory = PurePosixPath(_canonical_checkpoint(checkpoint))
        run_dir = directory.parent.parent
        return (
            run_dir / "run.json",
            directory / "checkpoint.json",
            directory / "model.safetensors",
        )


    def _download(
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
            repo_type="model",
            filename=filename,
            revision=revision,
            token=token,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        return destination


    def _stage_source(
        *,
        hf_repo: str,
        checkpoint: str,
        token: str,
        revision: str,
        root: Path,
    ) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for relative in _source_paths(checkpoint):
            destination = root / Path(*relative.parts)
            _download(
                repo_id=hf_repo,
                filename=relative.as_posix(),
                destination=destination,
                token=token,
                revision=revision,
            )
            files[relative.name] = destination
            print(
                f"[modal/simulation] staged {hf_repo}/{relative.as_posix()}",
                flush=True,
            )
        return files


    def _ensure_tokenizer() -> None:
        destination_root = Path(CACHE_DIR) / "tokenizer"
        for name in TOKENIZER_FILES:
            _download(
                repo_id=TOKENIZER_REPO,
                filename=name,
                destination=destination_root / name,
                token=None,
                revision="main",
            )
        print(
            f"[modal/simulation] tokenizer ready from {TOKENIZER_REPO} -> "
            f"{destination_root}",
            flush=True,
        )


    def _model_args(run_json: Path) -> dict[str, Any]:
        payload = json.loads(run_json.read_text())
        model = payload.get("model")
        config = model.get("model_config") if isinstance(model, dict) else None
        if not isinstance(config, dict):
            raise RuntimeError("source run.json has no model.model_config")
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
            raise RuntimeError("source model config missing: " + ", ".join(missing))
        depth = int(config["n_layer"])
        n_head = int(config["n_head"])
        n_kv_head = int(config["n_kv_head"])
        n_embd = int(config["n_embd"])
        if n_kv_head != n_head:
            raise RuntimeError("operator-square simulation requires MHA, not GQA")
        if n_embd % depth or n_embd % n_head:
            raise RuntimeError("cannot derive aspect_ratio/head_dim from source")
        return {
            "depth": depth,
            "aspect_ratio": n_embd // depth,
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
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        return int(proc.stdout.strip().splitlines()[-1])


    def _install_segment_trainer() -> str:
        """Patch a private base_train copy with process-only stop_after_step."""
        source = Path(REPO_DIR) / "scripts" / "base_train.py"
        destination = Path(REPO_DIR) / "scripts" / "_simulation_base_train.py"
        text = source.read_text()

        arg_anchor = (
            'parser.add_argument("--num-iterations", type=int, default=-1, '
            'help="explicit number of optimization steps (-1 = disable)")'
        )
        if arg_anchor not in text:
            raise RuntimeError(
                "base_train.py num-iterations anchor changed; refusing to patch"
            )
        text = text.replace(
            arg_anchor,
            arg_anchor
            + '\nparser.add_argument("--stop-after-step", type=int, default=-1, '
            'help="simulation process stop; scheduler still uses --num-iterations")',
            1,
        )

        horizon_anchor = 'print0(f"Total number of training tokens: {total_tokens:,}")'
        if horizon_anchor not in text:
            raise RuntimeError(
                "base_train.py training-horizon anchor changed; refusing to patch"
            )
        horizon_check = '''
if args.stop_after_step >= 0:
    if args.stop_after_step > num_iterations:
        raise ValueError("--stop-after-step cannot exceed --num-iterations")
    if args.resume_from_step >= 0 and args.stop_after_step <= args.resume_from_step:
        raise ValueError(
            "--stop-after-step must be greater than --resume-from-step"
        )
    print0(
        f"[simulation] fixed scheduler horizon={num_iterations}; "
        f"this process stops at step={args.stop_after_step}"
    )
'''
        text = text.replace(horizon_anchor, horizon_anchor + horizon_check, 1)

        loop_anchor = (
            "    last_step = step == num_iterations # loop runs "
            "num_iterations+1 times so that we can eval/save at the end"
        )
        if loop_anchor not in text:
            raise RuntimeError(
                "base_train.py loop anchor changed; refusing to patch"
            )
        loop_patch = '''    simulation_stop = (
        args.stop_after_step >= 0 and step == args.stop_after_step
    )
    last_step = step == num_iterations or simulation_stop
    # stop_after_step only exits this process. LR/momentum/WD schedulers above
    # continue to use num_iterations, which is the fixed full simulation horizon.'''
        text = text.replace(loop_anchor, loop_patch, 1)

        destination.write_text(text)
        print(
            f"[modal/simulation] installed ephemeral segment trainer: {destination}",
            flush=True,
        )
        return "scripts._simulation_base_train"


    def _next_run_id(
        *,
        repo_files: list[str],
        folder: str,
        reserved: list[str],
    ) -> str:
        pattern = re.compile(rf"^{re.escape(folder)}/run-(\d{{4,}})/")
        values = [
            int(match.group(1))
            for filename in repo_files
            if (match := pattern.match(filename))
        ]
        for run_id in reserved:
            match = re.fullmatch(r"run-(\d{4,})", run_id)
            if match:
                values.append(int(match.group(1)))
        return f"run-{max(values, default=0) + 1:04d}"


    def _jobs_root() -> Path:
        return Path(CACHE_DIR) / "simulation_jobs"


    def _write_job(job: dict[str, Any]) -> None:
        directory = Path(job["job_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "job.json.tmp"
        tmp.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        tmp.replace(directory / "job.json")


    def _read_jobs(hf_repo: str) -> list[dict[str, Any]]:
        root = _jobs_root()
        if not root.is_dir():
            return []
        jobs: list[dict[str, Any]] = []
        for path in root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if _normalize_repo(str(job.get("hf_repo", ""))) == hf_repo:
                    job["_job_json"] = str(path)
                    jobs.append(job)
            except Exception as exc:
                print(
                    f"[modal/simulation] ignoring unreadable job {path}: {exc}",
                    flush=True,
                )
        return jobs


    def _raw_resume_steps(raw_dir: Path, *, num_gpus: int) -> list[int]:
        if not raw_dir.is_dir():
            return []
        model_steps = {
            int(match.group(1))
            for path in raw_dir.glob("model_*.pt")
            if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))
        }
        meta_steps = {
            int(match.group(1))
            for path in raw_dir.glob("meta_*.json")
            if (match := re.fullmatch(r"meta_(\d+)\.json", path.name))
        }
        candidates = sorted(model_steps & meta_steps)
        complete: list[int] = []
        for step in candidates:
            if num_gpus == 1:
                variants = (
                    raw_dir / f"optim_{step:06d}_rank0.pt",
                    raw_dir / f"optim_{step:06d}.pt",
                )
                if any(path.is_file() for path in variants):
                    complete.append(step)
            else:
                expected = [
                    raw_dir / f"optim_{step:06d}_rank{rank}.pt"
                    for rank in range(num_gpus)
                ]
                if all(path.is_file() for path in expected):
                    complete.append(step)
        return complete


    def _delete_raw_step(raw_dir: Path, step: int) -> None:
        patterns = (
            f"model_{step:06d}.pt",
            f"meta_{step:06d}.json",
            f"optim_{step:06d}.pt",
            f"optim_{step:06d}_rank*.pt",
        )
        for pattern in patterns:
            for path in raw_dir.glob(pattern):
                path.unlink(missing_ok=True)


    def _training_command(
        *,
        trainer_module: str,
        model_args: dict[str, Any],
        raw_tag: str,
        target_arm: str,
        horizon: int,
        stop_after_step: int,
        resume_from_step: int,
        seed_pt: Path,
        num_gpus: int,
        device_batch_size: int,
        total_batch_size: int,
    ) -> list[str]:
        args = [
            "-m",
            trainer_module,
            "--run=dummy",
            "--device-type=cuda",
            f"--depth={model_args['depth']}",
            f"--aspect-ratio={model_args['aspect_ratio']}",
            f"--head-dim={model_args['head_dim']}",
            f"--max-seq-len={model_args['max_seq_len']}",
            f"--window-pattern={model_args['window_pattern']}",
            f"--model-tag={raw_tag}",
            f"--arm={target_arm}",
            f"--num-iterations={horizon}",
            f"--stop-after-step={stop_after_step}",
            "--save-every=-1",
            "--eval-every=-1",
            "--core-metric-every=-1",
            "--sample-every=-1",
            f"--device-batch-size={device_batch_size}",
            f"--total-batch-size={total_batch_size}",
        ]
        if resume_from_step > 0:
            args.append(f"--resume-from-step={resume_from_step}")
        else:
            args.append(f"--init-from-model={seed_pt}")

        if num_gpus == 1:
            return [f"{VENV}/bin/python", *args]
        return [
            f"{VENV}/bin/torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            *args,
        ]


    def _evaluate(
        *,
        canonical_root: Path,
        results_root: Path,
        checkpoint: str,
        eval_tokens: int,
        core_max_per_task: int,
        induction_lengths: str,
        induction_offsets: str,
        induction_logical_batch_size: int,
        induction_micro_batch_size: int,
        induction_max_forward_tokens: int,
        induction_num_batches: int,
        induction_seed: int,
        induction_top_k: int,
        log_path: Path,
    ) -> Path:
        cmd = [
            f"{VENV}/bin/python",
            "-m",
            "experiments.simulation.run",
            "--repo-root",
            str(canonical_root),
            "--output-root",
            str(results_root),
            "--checkpoint",
            checkpoint,
            "--eval-tokens",
            str(eval_tokens),
            "--eval-batch-size",
            "1",
            "--core-max-per-task",
            str(core_max_per_task),
            "--induction-lengths",
            induction_lengths,
            f"--induction-offsets={induction_offsets}",
            "--induction-logical-batch-size",
            str(induction_logical_batch_size),
            "--induction-micro-batch-size",
            str(induction_micro_batch_size),
            "--induction-max-forward-tokens",
            str(induction_max_forward_tokens),
            "--induction-num-batches",
            str(induction_num_batches),
            "--induction-seed",
            str(induction_seed),
            "--induction-top-k",
            str(induction_top_k),
            "--device",
            "cuda",
            "--weights-device",
            "cpu",
        ]
        _run_streamed(cmd, log_path=log_path)
        path = PurePosixPath(checkpoint)
        result = (
            results_root
            / "metrics"
            / "simulation"
            / path.parts[0]
            / path.parts[1]
            / f"{path.parts[3]}.json"
        )
        if not result.is_file():
            raise RuntimeError(f"missing milestone result after evaluation: {result}")
        return result


    def _stage_and_evaluate(
        *,
        helper: str,
        run_json: Path,
        tmp_root: Path,
        results_root: Path,
        step: int,
        horizon: int,
        milestone_every: int,
        total_batch_size: int,
        model_safetensors: Path | None,
        model_pt: Path | None,
        eval_tokens: int,
        core_max_per_task: int,
        induction_lengths: str,
        induction_offsets: str,
        induction_logical_batch_size: int,
        induction_micro_batch_size: int,
        induction_max_forward_tokens: int,
        induction_num_batches: int,
        induction_seed: int,
        induction_top_k: int,
        log_path: Path,
        durable_canonical_copy: Path | None = None,
    ) -> tuple[dict[str, Any], Path]:
        canonical_root = tmp_root / f"milestone-{step:06d}"
        shutil.rmtree(canonical_root, ignore_errors=True)
        cmd = [
            f"{VENV}/bin/python",
            helper,
            "--runtime",
            "stage-milestone",
            "--run-json",
            str(run_json),
            "--output-root",
            str(canonical_root),
            "--step",
            str(step),
            "--horizon",
            str(horizon),
            "--milestone-every",
            str(milestone_every),
            "--total-batch-size",
            str(total_batch_size),
        ]
        if model_safetensors is not None:
            cmd.extend(["--model-safetensors", str(model_safetensors)])
        else:
            assert model_pt is not None
            cmd.extend(["--model-pt", str(model_pt)])
        staged = _run_json(cmd)
        result = _evaluate(
            canonical_root=canonical_root,
            results_root=results_root,
            checkpoint=str(staged["checkpoint_path"]),
            eval_tokens=eval_tokens,
            core_max_per_task=core_max_per_task,
            induction_lengths=induction_lengths,
            induction_offsets=induction_offsets,
            induction_logical_batch_size=induction_logical_batch_size,
            induction_micro_batch_size=induction_micro_batch_size,
            induction_max_forward_tokens=induction_max_forward_tokens,
            induction_num_batches=induction_num_batches,
            induction_seed=induction_seed,
            induction_top_k=induction_top_k,
            log_path=log_path,
        )
        if durable_canonical_copy is not None:
            shutil.rmtree(durable_canonical_copy, ignore_errors=True)
            shutil.copytree(canonical_root, durable_canonical_copy)
        shutil.rmtree(canonical_root, ignore_errors=True)
        return staged, result


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
    def run_simulation(
        hf_repo: str,
        source_checkpoint: str,
        target_arm: str,
        horizon: int = 10_000,
        milestone_every: int = 1_000,
        num_gpus: int = 1,
        device_batch_size: int = 1,
        total_batch_size: int = 16_384,
        data_shards: int = 240,
        eval_tokens: int = 4_194_304,
        core_max_per_task: int = 500,
        induction_lengths: str = "32,64,128,256,512",
        induction_offsets: str = "-2,-1,0,1,2",
        induction_logical_batch_size: int = 16,
        induction_micro_batch_size: int = 0,
        induction_max_forward_tokens: int = 8192,
        induction_num_batches: int = 4,
        induction_seed: int = 1234,
        induction_top_k: int = 8,
        revision: str = "main",
        git_commit: str = "",
        resume_job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if milestone_every <= 0 or horizon % milestone_every != 0:
            raise ValueError("milestone_every must be positive and divide horizon")
        if num_gpus not in {1, 2, 4, 8}:
            raise ValueError("num_gpus must be one of 1,2,4,8")
        if device_batch_size <= 0 or total_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if data_shards <= 0 or eval_tokens <= 0 or core_max_per_task <= 0:
            raise ValueError("data/evaluation settings must be positive")

        hf_repo = _normalize_repo(hf_repo)
        source_checkpoint = _canonical_checkpoint(source_checkpoint)
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

        visible = _visible_gpu_count()
        if visible != num_gpus:
            raise RuntimeError(
                f"visible GPU count={visible}, requested num_gpus={num_gpus}"
            )

        cache_volume.reload()
        helper = f"{REPO_DIR}/modal/simulation.py"
        arm_info = _run_json(
            [
                f"{VENV}/bin/python",
                helper,
                "--runtime",
                "arm-info",
                "--arm",
                target_arm,
            ]
        )
        target_arm = str(arm_info["name"])
        target_folder = str(arm_info["hf_folder"])

        if resume_job is None:
            remote_files = list(
                api.list_repo_files(
                    repo_id=hf_repo,
                    repo_type="model",
                    revision=revision,
                    token=token,
                )
            )
            reserved = [
                str(job["target_run_id"])
                for job in _read_jobs(hf_repo)
                if job.get("target_folder") == target_folder
                and job.get("target_run_id")
            ]
            target_run_id = _next_run_id(
                repo_files=remote_files,
                folder=target_folder,
                reserved=reserved,
            )
            job_id = (
                f"simulation-{target_arm}-{target_run_id}-{uuid.uuid4().hex[:10]}"
            )
            job_dir = _jobs_root() / job_id
            raw_tag = f"transient-{job_id}"
            raw_dir = Path(CACHE_DIR) / "base_checkpoints" / raw_tag
            shutil.rmtree(job_dir, ignore_errors=True)
            shutil.rmtree(raw_dir, ignore_errors=True)
            job_dir.mkdir(parents=True, exist_ok=True)
            job: dict[str, Any] = {
                "job_id": job_id,
                "hf_repo": hf_repo,
                "revision": revision,
                "source_checkpoint": source_checkpoint,
                "target_arm": target_arm,
                "target_folder": target_folder,
                "target_run_id": target_run_id,
                "horizon": horizon,
                "milestone_every": milestone_every,
                "num_gpus": num_gpus,
                "device_batch_size": device_batch_size,
                "total_batch_size": total_batch_size,
                "data_shards": data_shards,
                "eval_tokens": eval_tokens,
                "core_max_per_task": core_max_per_task,
                "induction_lengths": induction_lengths,
                "induction_offsets": induction_offsets,
                "induction_logical_batch_size": induction_logical_batch_size,
                "induction_micro_batch_size": induction_micro_batch_size,
                "induction_max_forward_tokens": induction_max_forward_tokens,
                "induction_num_batches": induction_num_batches,
                "induction_seed": induction_seed,
                "induction_top_k": induction_top_k,
                "git_commit": git_commit,
                "job_dir": str(job_dir),
                "raw_tag": raw_tag,
                "raw_dir": str(raw_dir),
                "evaluated_steps": [],
                "ready_to_publish": False,
            }
            _write_job(job)
            cache_volume.commit()
        else:
            job = dict(resume_job)
            target_run_id = str(job["target_run_id"])
            job_dir = Path(job["job_dir"])
            raw_tag = str(job["raw_tag"])
            raw_dir = Path(job["raw_dir"])

        print(
            f"[modal/simulation] trajectory {target_folder}/{target_run_id}: "
            f"0 -> {horizon} by {milestone_every}",
            flush=True,
        )
        print(
            "[modal/simulation] fixed scheduler horizon across every training "
            "segment; optimizer+dataloader resume at milestones",
            flush=True,
        )
        print(
            "[modal/simulation] no Hugging Face checkpoint will be written "
            "until the whole trajectory succeeds",
            flush=True,
        )

        tmp_root = Path(tempfile.mkdtemp(prefix=f"{job['job_id']}-", dir="/tmp"))
        source_root = tmp_root / "source"
        source_root.mkdir(parents=True, exist_ok=True)
        staged_source = _stage_source(
            hf_repo=hf_repo,
            checkpoint=source_checkpoint,
            token=token,
            revision=revision,
            root=source_root,
        )
        model_args = _model_args(staged_source["run.json"])

        run_json = job_dir / "run.json"
        if not run_json.is_file():
            _run_json(
                [
                    f"{VENV}/bin/python",
                    helper,
                    "--runtime",
                    "create-run",
                    "--source-root",
                    str(source_root),
                    "--source-checkpoint",
                    source_checkpoint,
                    "--target-arm",
                    target_arm,
                    "--target-run-id",
                    target_run_id,
                    "--hf-repo",
                    hf_repo,
                    "--revision",
                    revision,
                    "--horizon",
                    str(horizon),
                    "--milestone-every",
                    str(milestone_every),
                    "--total-batch-size",
                    str(total_batch_size),
                    "--device-batch-size",
                    str(device_batch_size),
                    "--data-shards",
                    str(data_shards),
                    "--git-commit",
                    git_commit,
                    "--output-run-json",
                    str(run_json),
                ]
            )
            cache_volume.commit()

        _ensure_tokenizer()
        print(
            f"[modal/simulation] ensuring {data_shards} data shards...",
            flush=True,
        )
        _run_streamed(
            [f"{VENV}/bin/python", "-m", "nanochat.dataset", "-n", str(data_shards)]
        )
        cache_volume.commit()

        seed_pt = tmp_root / "source_state_dict.pt"
        _run_json(
            [
                f"{VENV}/bin/python",
                helper,
                "--runtime",
                "prepare-seed",
                "--source-root",
                str(source_root),
                "--source-checkpoint",
                source_checkpoint,
                "--output-pt",
                str(seed_pt),
            ]
        )

        trainer_module = _install_segment_trainer()
        results_root = job_dir / "results"
        evaluated = {int(value) for value in job.get("evaluated_steps", [])}

        if 0 not in evaluated:
            _, result_path = _stage_and_evaluate(
                helper=helper,
                run_json=run_json,
                tmp_root=tmp_root,
                results_root=results_root,
                step=0,
                horizon=horizon,
                milestone_every=milestone_every,
                total_batch_size=total_batch_size,
                model_safetensors=staged_source["model.safetensors"],
                model_pt=None,
                eval_tokens=eval_tokens,
                core_max_per_task=core_max_per_task,
                induction_lengths=induction_lengths,
                induction_offsets=induction_offsets,
                induction_logical_batch_size=induction_logical_batch_size,
                induction_micro_batch_size=induction_micro_batch_size,
                induction_max_forward_tokens=induction_max_forward_tokens,
                induction_num_batches=induction_num_batches,
                induction_seed=induction_seed,
                induction_top_k=induction_top_k,
                log_path=job_dir / "eval-000000.log",
            )
            evaluated.add(0)
            job["evaluated_steps"] = sorted(evaluated)
            job["last_metric"] = str(result_path)
            _write_job(job)
            cache_volume.commit()

        resume_steps = _raw_resume_steps(raw_dir, num_gpus=num_gpus)
        current_step = max(resume_steps, default=0)

        # If a crash happened after checkpointing but during evaluation, finish
        # that milestone before taking another optimizer step.
        if current_step > 0 and current_step not in evaluated:
            raw_model = raw_dir / f"model_{current_step:06d}.pt"
            _, result_path = _stage_and_evaluate(
                helper=helper,
                run_json=run_json,
                tmp_root=tmp_root,
                results_root=results_root,
                step=current_step,
                horizon=horizon,
                milestone_every=milestone_every,
                total_batch_size=total_batch_size,
                model_safetensors=None,
                model_pt=raw_model,
                eval_tokens=eval_tokens,
                core_max_per_task=core_max_per_task,
                induction_lengths=induction_lengths,
                induction_offsets=induction_offsets,
                induction_logical_batch_size=induction_logical_batch_size,
                induction_micro_batch_size=induction_micro_batch_size,
                induction_max_forward_tokens=induction_max_forward_tokens,
                induction_num_batches=induction_num_batches,
                induction_seed=induction_seed,
                induction_top_k=induction_top_k,
                log_path=job_dir / f"eval-{current_step:06d}.log",
                durable_canonical_copy=(
                    job_dir / "final_evaluated"
                    if current_step == horizon
                    else None
                ),
            )
            evaluated.add(current_step)
            job["evaluated_steps"] = sorted(evaluated)
            job["last_metric"] = str(result_path)
            _write_job(job)
            cache_volume.commit()

        while current_step < horizon:
            next_step = min(current_step + milestone_every, horizon)
            cmd = _training_command(
                trainer_module=trainer_module,
                model_args=model_args,
                raw_tag=raw_tag,
                target_arm=target_arm,
                horizon=horizon,
                stop_after_step=next_step,
                resume_from_step=current_step,
                seed_pt=seed_pt,
                num_gpus=num_gpus,
                device_batch_size=device_batch_size,
                total_batch_size=total_batch_size,
            )
            _run_streamed(
                cmd,
                log_path=job_dir / f"train-to-{next_step:06d}.log",
            )

            raw_model = raw_dir / f"model_{next_step:06d}.pt"
            raw_meta = raw_dir / f"meta_{next_step:06d}.json"
            if not raw_model.is_file() or not raw_meta.is_file():
                raise RuntimeError(
                    f"training segment ended without step {next_step} model/meta"
                )

            # Commit new rolling resume state before running expensive evals.
            job["latest_training_step"] = next_step
            _write_job(job)
            cache_volume.commit()

            staged, result_path = _stage_and_evaluate(
                helper=helper,
                run_json=run_json,
                tmp_root=tmp_root,
                results_root=results_root,
                step=next_step,
                horizon=horizon,
                milestone_every=milestone_every,
                total_batch_size=total_batch_size,
                model_safetensors=None,
                model_pt=raw_model,
                eval_tokens=eval_tokens,
                core_max_per_task=core_max_per_task,
                induction_lengths=induction_lengths,
                induction_offsets=induction_offsets,
                induction_logical_batch_size=induction_logical_batch_size,
                induction_micro_batch_size=induction_micro_batch_size,
                induction_max_forward_tokens=induction_max_forward_tokens,
                induction_num_batches=induction_num_batches,
                induction_seed=induction_seed,
                induction_top_k=induction_top_k,
                log_path=job_dir / f"eval-{next_step:06d}.log",
                durable_canonical_copy=(
                    job_dir / "final_evaluated"
                    if next_step == horizon
                    else None
                ),
            )
            evaluated.add(next_step)
            job["evaluated_steps"] = sorted(evaluated)
            job["last_metric"] = str(result_path)
            job["latest_model_sha256"] = str(staged["model_sha256"])
            _write_job(job)
            cache_volume.commit()

            # This implements the rolling/overwrite semantics safely: only after
            # the newer state AND its metrics are durable is the older large state
            # removed.
            if current_step > 0:
                _delete_raw_step(raw_dir, current_step)
                cache_volume.commit()
            current_step = next_step

        if horizon not in evaluated:
            raise RuntimeError("final milestone evaluation is missing")

        # Publish the exact canonical bytes that produced the final metrics.
        final_evaluated = job_dir / "final_evaluated"
        if not final_evaluated.is_dir():
            raise RuntimeError(
                "final milestone metrics exist but exact evaluated canonical "
                "checkpoint copy is missing"
            )
        bundle = job_dir / "publish_bundle"
        shutil.rmtree(bundle, ignore_errors=True)
        shutil.copytree(final_evaluated, bundle)

        final_checkpoint_path = (
            PurePosixPath(target_folder)
            / target_run_id
            / "checkpoints"
            / f"step-{horizon:06d}"
        ).as_posix()
        final_checkpoint_json = (
            bundle
            / target_folder
            / target_run_id
            / "checkpoints"
            / f"step-{horizon:06d}"
            / "checkpoint.json"
        )
        final_checkpoint_meta = json.loads(final_checkpoint_json.read_text())
        final_model_sha256 = str(final_checkpoint_meta["model_sha256"])

        metrics_source = results_root / "metrics"
        if not metrics_source.is_dir():
            raise RuntimeError("simulation produced no metric artifacts")
        shutil.copytree(metrics_source, bundle / "metrics", dirs_exist_ok=True)

        final_metric = (
            bundle
            / "metrics"
            / "simulation"
            / target_folder
            / target_run_id
            / f"step-{horizon:06d}.json"
        )
        final_metric_payload = json.loads(final_metric.read_text())
        if final_metric_payload.get("model_sha256") != final_model_sha256:
            raise RuntimeError(
                "final evaluated model SHA differs from publication model SHA"
            )

        job["ready_to_publish"] = True
        job["final_bundle"] = str(bundle)
        job["final_checkpoint"] = final_checkpoint_path
        job["final_model_sha256"] = final_model_sha256
        _write_job(job)
        cache_volume.commit()
        print(
            f"[modal/simulation] completed {target_folder}/{target_run_id} "
            f"through step {horizon}; final bundle durable, HF still untouched",
            flush=True,
        )
        return job


    @app.function(
        image=image,
        cpu=4,
        memory=8192,
        timeout=30 * 60,
        volumes={CACHE_DIR: cache_volume},
        secrets=[hf_secret],
    )
    def load_pending_job(
        hf_repo: str,
        pending_job: str = "latest",
        target_arm: str = "",
    ) -> dict[str, Any]:
        hf_repo = _normalize_repo(hf_repo)
        cache_volume.reload()
        jobs = _read_jobs(hf_repo)
        if target_arm.strip():
            target = target_arm.strip().lower()
            jobs = [
                job
                for job in jobs
                if str(job.get("target_arm", "")).lower() == target
            ]
        if not jobs:
            raise RuntimeError("no pending simulation jobs matched")
        if pending_job.strip().lower() == "latest":
            jobs.sort(
                key=lambda job: Path(job["_job_json"]).stat().st_mtime,
                reverse=True,
            )
            return jobs[0]
        needle = pending_job.strip()
        matches = [job for job in jobs if needle in str(job.get("job_id", ""))]
        if len(matches) != 1:
            raise RuntimeError(
                f"pending_job={needle!r} matched {len(matches)} jobs"
            )
        return matches[0]


    def _remote_complete(
        *,
        api,
        hf_repo: str,
        revision: str,
        token: str,
        paths: list[str],
    ) -> bool:
        remote = set(
            api.list_repo_files(
                repo_id=hf_repo,
                repo_type="model",
                revision=revision,
                token=token,
            )
        )
        return all(path in remote for path in paths)


    @app.function(
        image=image,
        cpu=16,
        memory=131072,
        timeout=2 * 60 * 60,
        volumes={CACHE_DIR: cache_volume},
        secrets=[hf_secret],
    )
    def publish_simulation(job: dict[str, Any]) -> dict[str, Any]:
        cache_volume.reload()
        token = os.environ.get("HF_TOKEN") or None
        if token is None:
            raise RuntimeError("HF_TOKEN is required")

        job_dir = Path(job["job_dir"])
        saved_job = job_dir / "job.json"
        if saved_job.is_file():
            job = json.loads(saved_job.read_text())
        if not job.get("ready_to_publish"):
            raise RuntimeError("simulation is not ready for publication")

        bundle = Path(job["final_bundle"])
        if not bundle.is_dir():
            raise RuntimeError(f"missing final bundle: {bundle}")
        paths = sorted(
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file()
        )
        if not paths:
            raise RuntimeError("final simulation bundle is empty")

        hf_repo = _normalize_repo(str(job["hf_repo"]))
        revision = str(job["revision"])
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.auth_check(
            repo_id=hf_repo,
            repo_type="model",
            token=token,
            write=True,
        )

        if not _remote_complete(
            api=api,
            hf_repo=hf_repo,
            revision=revision,
            token=token,
            paths=paths,
        ):
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    print(
                        f"[modal/simulation] HF Xet upload attempt {attempt}/3...",
                        flush=True,
                    )
                    api.upload_folder(
                        repo_id=hf_repo,
                        repo_type="model",
                        folder_path=str(bundle),
                        path_in_repo="",
                        revision=revision,
                        token=token,
                        commit_message=(
                            f"simulation {job['target_arm']} "
                            f"{job['target_run_id']} step {job['horizon']}"
                        ),
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[modal/simulation] upload attempt {attempt} raised: {exc}",
                        flush=True,
                    )
                    if _remote_complete(
                        api=api,
                        hf_repo=hf_repo,
                        revision=revision,
                        token=token,
                        paths=paths,
                    ):
                        last_error = None
                        break
                    time.sleep(min(20, 3 * attempt))
            if last_error is not None:
                raise last_error

        if not _remote_complete(
            api=api,
            hf_repo=hf_repo,
            revision=revision,
            token=token,
            paths=paths,
        ):
            raise RuntimeError("HF upload returned but final bundle is incomplete")

        raw_dir = Path(job["raw_dir"])
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(job_dir, ignore_errors=True)
        cache_volume.commit()

        result = {
            "ok": True,
            "hf_repo": hf_repo,
            "target_arm": job["target_arm"],
            "target_run_id": job["target_run_id"],
            "final_checkpoint": job["final_checkpoint"],
            "model_sha256": job["final_model_sha256"],
            "horizon": job["horizon"],
            "milestone_every": job["milestone_every"],
            "evaluated_steps": job["evaluated_steps"],
            "uploaded_files": paths,
        }
        print(
            f"[modal/simulation] published {job['target_folder']}/"
            f"{job['target_run_id']}",
            flush=True,
        )
        return result


    @app.local_entrypoint()
    def main(
        hf_repo: str,
        mode: str = "train",
        source_checkpoint: str = "attention/run-0001/checkpoints/step-000000",
        arm: str = "hmap",
        horizon: int = 10_000,
        milestone_every: int = 1_000,
        pending_job: str = "latest",
        gpu: str = "H100",
        num_gpus: int = 1,
        device_batch_size: int = 1,
        total_batch_size: int = 16_384,
        data_shards: int = 240,
        eval_tokens: int = 4_194_304,
        core_max_per_task: int = 500,
        induction_lengths: str = "32,64,128,256,512",
        induction_offsets: str = "-2,-1,0,1,2",
        induction_logical_batch_size: int = 16,
        induction_micro_batch_size: int = 0,
        induction_max_forward_tokens: int = 8192,
        induction_num_batches: int = 4,
        induction_seed: int = 1234,
        induction_top_k: int = 8,
        revision: str = "main",
    ) -> None:
        mode = mode.strip().lower()
        if mode not in {"train", "recover"}:
            raise ValueError("mode must be train or recover")
        if num_gpus not in {1, 2, 4, 8}:
            raise ValueError("num_gpus must be one of 1,2,4,8")

        common: dict[str, Any] = {
            "hf_repo": hf_repo,
            "source_checkpoint": source_checkpoint,
            "target_arm": arm,
            "horizon": horizon,
            "milestone_every": milestone_every,
            "num_gpus": num_gpus,
            "device_batch_size": device_batch_size,
            "total_batch_size": total_batch_size,
            "data_shards": data_shards,
            "eval_tokens": eval_tokens,
            "core_max_per_task": core_max_per_task,
            "induction_lengths": induction_lengths,
            "induction_offsets": induction_offsets,
            "induction_logical_batch_size": induction_logical_batch_size,
            "induction_micro_batch_size": induction_micro_batch_size,
            "induction_max_forward_tokens": induction_max_forward_tokens,
            "induction_num_batches": induction_num_batches,
            "induction_seed": induction_seed,
            "induction_top_k": induction_top_k,
            "revision": revision,
            "git_commit": os.environ.get("GITHUB_SHA", ""),
        }

        requested_num_gpus = num_gpus
        if mode == "recover":
            pending = load_pending_job.remote(
                hf_repo=hf_repo,
                pending_job=pending_job,
                target_arm=arm,
            )
            if pending.get("ready_to_publish"):
                result = publish_simulation.remote(pending)
                print(json.dumps(result, indent=2, sort_keys=True))
                return

            # Recovery is exact: persisted scientific settings win over workflow
            # defaults so a rerun cannot silently change the trajectory.
            saved_keys = (
                "source_checkpoint",
                "target_arm",
                "horizon",
                "milestone_every",
                "num_gpus",
                "device_batch_size",
                "total_batch_size",
                "data_shards",
                "eval_tokens",
                "core_max_per_task",
                "induction_lengths",
                "induction_offsets",
                "induction_logical_batch_size",
                "induction_micro_batch_size",
                "induction_max_forward_tokens",
                "induction_num_batches",
                "induction_seed",
                "induction_top_k",
                "revision",
            )
            for key in saved_keys:
                common[key] = pending[key]
            common["git_commit"] = pending.get("git_commit", "")
            common["resume_job"] = pending
            requested_num_gpus = int(pending["num_gpus"])

        gpu_request = (
            gpu if requested_num_gpus == 1 else f"{gpu}:{requested_num_gpus}"
        )
        print(
            f"[modal/simulation] requesting GPU resource {gpu_request}",
            flush=True,
        )
        completed = run_simulation.with_options(gpu=gpu_request).remote(**common)
        result = publish_simulation.remote(completed)
        print(json.dumps(result, indent=2, sort_keys=True))
