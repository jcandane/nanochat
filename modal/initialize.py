"""modal/initialize.py
One-time bootstrap of Karpathy's pretrained nanochat d32 into the canonical
operator-square Hugging Face repository layout.

This is intentionally a temporary migration tool, not a training script.

It imports the pinned upstream checkpoint:

    karpathy/nanochat-d32
    model_000650.pt
    meta_000650.json

and writes the canonical local-step-zero checkpoint:

    attention/
    └── run-0001/
        ├── run.json
        └── checkpoints/
            └── step-000000/
                ├── model.safetensors
                └── checkpoint.json

No optimizer steps are taken. Newly introduced parameters that do not exist in
the vintage d32 checkpoint are filled with explicit function-neutral values so
the converted checkpoint is key-for-key compatible with the current fork.

This file has two modes:

* outer mode: Modal + Hugging Face infrastructure;
* --runtime: conversion code executed inside the repo's uv GPU environment.

Keeping runtime mode in the same temporary file avoids adding another permanent
migration module to the repository.
"""

from __future__ import annotations

import sys


_RUNTIME = "--runtime" in sys.argv


if _RUNTIME:
    # -------------------------------------------------------------------------
    # Inner conversion runtime. Executed with /root/nanochat/.venv/bin/python.
    # Avoid importing Modal here.
    # -------------------------------------------------------------------------
    import argparse
    from dataclasses import asdict, fields
    from datetime import datetime, timezone
    import gc
    import hashlib
    import json
    from pathlib import Path
    from typing import Any

    import torch
    from safetensors.torch import save_file

    from experiments.common.arms import get_arm
    from experiments.common.checkpoints import (
        CHECKPOINT_SCHEMA_VERSION,
        CheckpointRef,
        validate_state_dict,
        write_json_atomic,
    )
    from nanochat.gpt import GPT, GPTConfig


    SOURCE_EXPECTED_CONFIG = {
        "sequence_len": 2048,
        "vocab_size": 65536,
        "n_layer": 32,
        "n_head": 16,
        "n_kv_head": 16,
        "n_embd": 2048,
    }

    # Karpathy's original HF upload, pinned below, stores this LFS object.
    SOURCE_EXPECTED_MODEL_SHA256 = (
        "761a83eb6f3e4798f535db280ca37ef4985ef246cb979e4cb84042c84c30598a"
    )

    # These are the only vintage-missing parameters accepted by this importer.
    # Every fill is chosen so the new path is functionally neutral at step 0.
    ALLOWED_MISSING_FRAGMENTS = (
        "value_embeds.",
        ".ve_gate.",
        "resid_lambdas",
        "x0_lambdas",
        "smear_gate.",
        "smear_lambda",
        "backout_lambda",
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


    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise RuntimeError(f"{path} must contain a JSON object")
        return value


    def _validate_source_meta(meta: dict[str, Any]) -> dict[str, Any]:
        source_config = meta.get("model_config")
        if not isinstance(source_config, dict):
            raise RuntimeError("Karpathy meta file has no model_config object")

        mismatches = {
            key: (source_config.get(key), expected)
            for key, expected in SOURCE_EXPECTED_CONFIG.items()
            if source_config.get(key) != expected
        }
        if mismatches:
            rendered = ", ".join(
                f"{key}: got={got!r}, expected={expected!r}"
                for key, (got, expected) in mismatches.items()
            )
            raise RuntimeError(
                "source checkpoint is not the pinned nanochat d32 architecture: "
                + rendered
            )

        if int(meta.get("step", -1)) != 650:
            raise RuntimeError(
                f"source meta step={meta.get('step')!r}; expected step 650"
            )
        return dict(source_config)


    def _current_attention_config(source_config: dict[str, Any]) -> GPTConfig:
        """Construct today's fork config while preserving d32 architecture."""
        available = {field.name for field in fields(GPTConfig)}
        arm = get_arm("attention")

        kwargs: dict[str, Any] = {
            **source_config,
            # Karpathy d32 / the existing pinned probe uses full context.
            "window_pattern": "L",
            **arm.config_overrides(),
        }

        # Current fork-only switches. Add them only when the local GPTConfig has
        # the field, so this bootstrap remains robust to small upstream changes.
        optional_neutral = {
            "witten": False,
            "bidirectional": False,
        }
        for key, value in optional_neutral.items():
            if key in available:
                kwargs[key] = value

        unknown = sorted(set(kwargs) - available)
        if unknown:
            raise RuntimeError(
                "current GPTConfig lacks fields required by d32 import: "
                + ", ".join(unknown)
            )

        config = GPTConfig(**kwargs)

        # Persist attention at the new operator-square corner (beta, alpha)=(1,1).
        if config.attn_variant != "standard":
            raise RuntimeError("bootstrap target must use standard attention")
        if float(config.hmap_beta) != 1.0 or float(config.hmap_alpha) != 1.0:
            raise RuntimeError(
                "current canonical attention arm must be (beta, alpha)=(1,1)"
            )
        if config.n_kv_head != config.n_head:
            raise RuntimeError("canonical d32 bootstrap must use MHA, not GQA")
        return config


    def _is_allowed_missing(key: str) -> bool:
        return any(fragment in key for fragment in ALLOWED_MISSING_FRAGMENTS)


    def _neutral_tensor(
        key: str,
        template: torch.Tensor,
        *,
        embedding_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, str]:
        """Create a deterministic function-neutral value for a vintage-missing key."""
        shape = tuple(template.shape)

        if key.startswith("value_embeds."):
            return (
                torch.zeros(shape, dtype=embedding_dtype, device="cpu"),
                "zeros (value residual disabled at import)",
            )

        if key == "resid_lambdas":
            return (
                torch.ones(shape, dtype=template.dtype, device="cpu"),
                "ones (identity residual scaling)",
            )

        if key == "x0_lambdas":
            return (
                torch.zeros(shape, dtype=template.dtype, device="cpu"),
                "zeros (x0 residual disabled)",
            )

        if key == "backout_lambda":
            return (
                torch.zeros(shape, dtype=template.dtype, device="cpu"),
                "zeros (backout disabled)",
            )

        if key == "smear_lambda":
            return (
                torch.zeros(shape, dtype=template.dtype, device="cpu"),
                "zeros (smear disabled)",
            )

        if key.startswith("smear_gate."):
            return (
                torch.zeros(shape, dtype=template.dtype, device="cpu"),
                "zeros (inactive because smear_lambda=0)",
            )

        if ".ve_gate." in key:
            return (
                torch.zeros(shape, dtype=template.dtype, device="cpu"),
                "zeros (value embedding itself is zero at import)",
            )

        raise RuntimeError(f"no neutral-fill rule for allowed key {key!r}")


    def _sanitize_for_safetensors(
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Detach storages from the pickle checkpoint before safetensors writing.

        The source .pt can contain storage layouts that are larger than the
        logical tensors. Cloning each tensor produces compact, independent,
        contiguous storage and avoids safetensors shared-storage rejection.
        """
        clean: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"state_dict[{key!r}] is {type(value).__name__}; expected Tensor"
                )
            clean[key] = value.detach().cpu().clone().contiguous()
        return clean


    def runtime_main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-model", required=True)
        parser.add_argument("--source-meta", required=True)
        parser.add_argument("--output-root", required=True)
        parser.add_argument("--run-id", default="run-0001")
        parser.add_argument("--source-repo", required=True)
        parser.add_argument("--source-revision", required=True)
        parser.add_argument("--source-model-file", required=True)
        parser.add_argument("--source-meta-file", required=True)
        parser.add_argument("--git-commit", default="")
        args = parser.parse_args()

        source_model_path = Path(args.source_model)
        source_meta_path = Path(args.source_meta)
        output_root = Path(args.output_root)

        source_sha256 = _sha256(source_model_path)
        if source_sha256 != SOURCE_EXPECTED_MODEL_SHA256:
            raise RuntimeError(
                "source model SHA-256 mismatch; refusing to canonicalize an "
                "unexpected upstream file:\n"
                f"  got      {source_sha256}\n"
                f"  expected {SOURCE_EXPECTED_MODEL_SHA256}"
            )

        source_meta = _load_json(source_meta_path)
        source_config = _validate_source_meta(source_meta)
        config = _current_attention_config(source_config)

        ref = CheckpointRef(
            arm="attention",
            run_id=args.run_id,
            step=0,
        )

        checkpoint_dir = output_root / Path(*ref.directory.parts)
        run_dir = output_root / Path(*ref.run_dir.parts)
        model_out = checkpoint_dir / "model.safetensors"
        checkpoint_json_out = checkpoint_dir / "checkpoint.json"
        run_json_out = run_dir / "run.json"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build only on meta: we need today's exact key/shape contract, not a
        # second full initialized model in RAM.
        with torch.device("meta"):
            meta_model = GPT(config)
        expected_state = meta_model.state_dict()

        source_state = torch.load(
            source_model_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(source_state, dict):
            raise RuntimeError(
                f"{source_model_path} did not contain a raw state_dict"
            )

        report = validate_state_dict(expected_state, source_state)

        if report.unexpected_keys:
            raise RuntimeError(
                "upstream d32 contains unexpected keys for this fork: "
                + ", ".join(report.unexpected_keys[:20])
            )
        if report.shape_mismatches:
            raise RuntimeError(
                "upstream d32 has shape mismatches for this fork: "
                + report.summary(max_items=20)
            )

        bad_missing = [
            key for key in report.missing_keys
            if not _is_allowed_missing(key)
        ]
        if bad_missing:
            raise RuntimeError(
                "upstream d32 is missing keys outside the explicit vintage "
                "compatibility allowlist: "
                + ", ".join(bad_missing[:20])
            )

        wte = source_state.get("transformer.wte.weight")
        if not isinstance(wte, torch.Tensor):
            raise RuntimeError(
                "source state_dict is missing transformer.wte.weight"
            )
        embedding_dtype = wte.dtype

        fill_report: dict[str, str] = {}
        for key in report.missing_keys:
            tensor, policy = _neutral_tensor(
                key,
                expected_state[key],
                embedding_dtype=embedding_dtype,
            )
            source_state[key] = tensor
            fill_report[key] = policy

        final_report = validate_state_dict(expected_state, source_state)
        final_report.raise_if_invalid()

        print(
            "[initialize] key/shape contract complete: "
            f"{len(source_state):,} tensors; "
            f"{len(fill_report):,} vintage keys neutral-filled",
            flush=True,
        )

        # Detach source pickle storages. This also compacts the old .pt into
        # safetensors and guarantees no storage aliasing in the canonical file.
        canonical_state = _sanitize_for_safetensors(source_state)
        del source_state
        gc.collect()

        save_file(
            canonical_state,
            str(model_out),
            metadata={
                "format": "nanochat-operator-square",
                "trained_arm": "attention",
                "run_id": args.run_id,
                "local_step": "0",
                "source_repo": args.source_repo,
                "source_revision": args.source_revision,
                "source_step": "650",
                "source_model_sha256": source_sha256,
            },
        )
        del canonical_state
        gc.collect()

        model_sha256 = _sha256(model_out)
        created_at = _utc_now_iso()
        attention = get_arm("attention")

        source_info = {
            "repo_id": args.source_repo,
            "revision": args.source_revision,
            "model_file": args.source_model_file,
            "meta_file": args.source_meta_file,
            "source_step": 650,
            "source_model_sha256": source_sha256,
            "source_meta": source_meta,
        }

        # Provenance schema v1 currently recognizes scratch/graft only. Keep the
        # initialization field parser-compatible, while making the authoritative
        # external import explicit in lineage + training. This can later migrate
        # losslessly if provenance.py gains an external_import kind.
        run_payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": ref.run_id,
            "arm": ref.arm,
            "operator": attention.as_dict(),
            "initialization": {
                "type": "scratch",
            },
            "lineage": [
                {
                    "event": "external_import",
                    "target_checkpoint": ref.as_dict(),
                    "source": source_info,
                    "local_step_origin": 0,
                    "optimizer_steps": 0,
                }
            ],
            "model": {
                "model_config": asdict(config),
                "source_model_config": source_config,
                "compatibility_mode": "karpathy-d32-vintage-neutral-fill",
            },
            "training": {
                "mode": "external_import",
                "optimizer_steps": 0,
                "tokens_seen_since_import": 0,
                "source": source_info,
                "migration": {
                    "missing_key_count": len(fill_report),
                    "neutral_fill": fill_report,
                },
            },
            "created_at": created_at,
            "notes": (
                "One-time import of Karpathy nanochat-d32 into the canonical "
                "attention/run checkpoint namespace. No optimizer step was taken. "
                "Schema-v1 initialization.type remains 'scratch' only for parser "
                "compatibility; lineage.event='external_import' is authoritative."
            ),
        }
        if args.git_commit:
            run_payload["git_commit"] = args.git_commit

        checkpoint_payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "arm": ref.arm,
            "run_id": ref.run_id,
            "step": ref.step,
            "path": ref.directory.as_posix(),
            "model_file": ref.model_path.name,
            "trained_operator": attention.as_dict(),
            "initialization_type": "scratch",
            "lineage_depth": 1,
            "created_at": created_at,
            "model_sha256": model_sha256,
            "local_optimizer_steps": 0,
            "tokens_seen_since_import": 0,
            "source": source_info,
            "migration": {
                "compatibility_mode": "karpathy-d32-vintage-neutral-fill",
                "missing_key_count": len(fill_report),
                "neutral_fill": fill_report,
            },
        }
        if args.git_commit:
            checkpoint_payload["git_commit"] = args.git_commit

        write_json_atomic(run_json_out, run_payload)
        write_json_atomic(checkpoint_json_out, checkpoint_payload)

        summary = {
            "checkpoint": ref.as_dict(),
            "source_model_sha256": source_sha256,
            "model_sha256": model_sha256,
            "model_bytes": model_out.stat().st_size,
            "neutral_filled_keys": len(fill_report),
            "files": [
                ref.run_metadata_path.as_posix(),
                ref.model_path.as_posix(),
                ref.metadata_path.as_posix(),
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


    if __name__ == "__main__":
        sys.argv.remove("--runtime")
        runtime_main()


else:
    # -------------------------------------------------------------------------
    # Outer Modal/Hugging Face bridge.
    # -------------------------------------------------------------------------
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    from typing import Any

    import modal


    APP_NAME = "nanochat-initialize-d32"
    REPO_DIR = "/root/nanochat"
    VENV = f"{REPO_DIR}/.venv"

    CACHE_DIR = "/root/.cache/nanochat-research"
    CACHE_VOLUME_NAME = "nanochat-research-cache"

    SOURCE_REPO = "karpathy/nanochat-d32"
    SOURCE_REVISION = "03560e49d2c73c2abe4c2760f01b193c27154e61"
    SOURCE_MODEL_FILE = "model_000650.pt"
    SOURCE_META_FILE = "meta_000650.json"

    app = modal.App(APP_NAME)

    cache_volume = modal.Volume.from_name(
        CACHE_VOLUME_NAME,
        create_if_missing=True,
    )

    # GitHub Actions exposes HF_TOKEN to `modal run`; this copies it into the
    # remote function without creating a separately named Modal Secret.
    hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])

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


    def _run_streamed(cmd: list[str]) -> None:
        print("[initialize] running:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"command failed with exit code {return_code}: {' '.join(cmd)}"
            )


    def _download_source(filename: str, *, token: str | None = None) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=SOURCE_REPO,
                filename=filename,
                repo_type="model",
                revision=SOURCE_REVISION,
                token=token,
                cache_dir=f"{CACHE_DIR}/huggingface",
            )
        )


    def _target_paths(run_id: str) -> tuple[str, str, str]:
        base = f"attention/{run_id}"
        checkpoint = f"{base}/checkpoints/step-000000"
        return (
            f"{base}/run.json",
            f"{checkpoint}/model.safetensors",
            f"{checkpoint}/checkpoint.json",
        )


    @app.function(
        image=image,
        cpu=16,
        memory=65536,
        ephemeral_disk=65536,
        timeout=6 * 60 * 60,
        retries=0,
        volumes={CACHE_DIR: cache_volume},
        secrets=[hf_secret],
    )
    def initialize_d32(
        hf_repo: str,
        run_id: str = "run-0001",
        force: bool = False,
        git_commit: str = "",
    ) -> dict[str, Any]:
        """Convert and upload Karpathy d32 as attention/<run>/step-000000."""
        if not hf_repo.strip():
            raise ValueError("hf_repo must be non-empty")

        token = os.environ.get("HF_TOKEN") or None
        if token is None:
            raise RuntimeError(
                "HF_TOKEN is required; the target Hugging Face repo needs "
                "write access"
            )

        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # Fail early if HF_REPO is wrong rather than accidentally creating a
        # public repository or uploading somewhere unintended.
        api.repo_info(
            repo_id=hf_repo,
            repo_type="model",
            token=token,
        )

        existing = set(
            api.list_repo_files(
                repo_id=hf_repo,
                repo_type="model",
                token=token,
            )
        )
        target_paths = _target_paths(run_id)
        collisions = sorted(set(target_paths) & existing)
        if collisions and not force:
            raise RuntimeError(
                "canonical bootstrap target already exists; refusing to "
                "overwrite without --force:\n  "
                + "\n  ".join(collisions)
            )

        cache_volume.reload()

        source_model = _download_source(SOURCE_MODEL_FILE, token=None)
        source_meta = _download_source(SOURCE_META_FILE, token=None)

        job_root = Path(
            tempfile.mkdtemp(prefix="nanochat-initialize-", dir="/tmp")
        )
        output_root = job_root / "output"
        output_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            f"{VENV}/bin/python",
            f"{REPO_DIR}/modal/initialize.py",
            "--runtime",
            "--source-model",
            str(source_model),
            "--source-meta",
            str(source_meta),
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--source-repo",
            SOURCE_REPO,
            "--source-revision",
            SOURCE_REVISION,
            "--source-model-file",
            SOURCE_MODEL_FILE,
            "--source-meta-file",
            SOURCE_META_FILE,
            "--git-commit",
            git_commit,
        ]
        _run_streamed(cmd)

        produced = [
            output_root / Path(path)
            for path in target_paths
        ]
        missing = [path for path in produced if not path.is_file()]
        if missing:
            raise RuntimeError(
                "conversion finished without expected output files:\n  "
                + "\n  ".join(str(path) for path in missing)
            )

        # Upload the run folder in its canonical layout. Hugging Face's
        # upload_folder handles large model files and resumes already-uploaded
        # chunks when rerun.
        run_folder = output_root / "attention" / run_id
        api.upload_folder(
            repo_id=hf_repo,
            repo_type="model",
            token=token,
            folder_path=str(run_folder),
            path_in_repo=f"attention/{run_id}",
            commit_message=(
                "bootstrap Karpathy nanochat-d32 as "
                f"attention/{run_id}/step-000000"
            ),
        )

        cache_volume.commit()

        checkpoint_json = json.loads(
            (
                output_root
                / "attention"
                / run_id
                / "checkpoints"
                / "step-000000"
                / "checkpoint.json"
            ).read_text()
        )

        return {
            "hf_repo": hf_repo,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_step": 650,
            "target": f"attention/{run_id}/checkpoints/step-000000",
            "model_sha256": checkpoint_json["model_sha256"],
            "uploaded_files": list(target_paths),
            "force": force,
        }


    @app.local_entrypoint()
    def main(
        hf_repo: str,
        run_id: str = "run-0001",
        force: bool = False,
    ) -> None:
        """Local/GitHub Actions entrypoint."""
        result = initialize_d32.remote(
            hf_repo=hf_repo,
            run_id=run_id,
            force=force,
            git_commit=os.environ.get("GITHUB_SHA", ""),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
