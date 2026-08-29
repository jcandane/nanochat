"""modal/induction.py
Modal bridge for the induction behavior + DAC experiment.

This file is infrastructure only. The scientific experiment lives in:

    experiments/induction/run.py
    experiments/induction/behavior.py
    experiments/induction/dac.py

Responsibilities here are intentionally narrow:

1. build the repo's GPU runtime on Modal,
2. stage one canonical checkpoint from the shared Hugging Face model repo,
3. install the exact nanochat d32 tokenizer,
4. invoke ``python -m experiments.induction.run``,
5. upload the emitted ``metrics/induction/...`` JSON artifacts back to the
   same Hugging Face repo.

There are deliberately no arm-coordinate tables, model-construction rules,
induction metric formulas, or DAC formulas in this module.

Typical use:

    modal run modal/induction.py \
        --hf-repo your-name/attention-operator-square \
        --checkpoint attention/run-0001/checkpoints/step-010000 \
        --evaluation-arm all

GitHub Actions can invoke the same command. Set ``HF_TOKEN`` in the caller's
environment; Modal forwards it to the remote Function as a Secret.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any

import modal


APP_NAME = "nanochat-induction"
REPO_DIR = "/root/nanochat"
VENV = f"{REPO_DIR}/.venv"

CACHE_DIR = "/root/.cache/nanochat-research"
CACHE_VOLUME_NAME = "nanochat-research-cache"

DEFAULT_TOKENIZER_REPO = "karpathy/nanochat-d32"
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")

# Modal's current GPU name is "A10" (not the older "A10G" spelling).
# CI may override this before `modal run`, e.g. NANOCHAT_MODAL_GPU=H100.
MODAL_GPU = os.environ.get("NANOCHAT_MODAL_GPU", "A10")


app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name(
    CACHE_VOLUME_NAME,
    create_if_missing=True,
)

# This mirrors the user's existing CI pattern: GitHub Actions exports HF_TOKEN
# locally, then Modal injects it into the remote Function. `from_local_environ`
# avoids hard-coding or logging the token.
if modal.is_local():
    hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])
else:
    # Global definitions are also evaluated in the remote runtime.
    hf_secret = modal.Secret.from_dict({})


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install("huggingface_hub")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
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
    # Bake the checked-out fork into the Image because the next build step runs
    # `uv sync` against its pyproject/lockfile.
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
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)


def _run_streamed(cmd: list[str]) -> None:
    """Run a subprocess in the repo venv while streaming logs to Modal."""
    print("[modal/induction] running:", " ".join(cmd), flush=True)

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


def _canonical_checkpoint_dir(checkpoint: str) -> PurePosixPath:
    """Normalize the structural checkpoint path without duplicating arm logic.

    Full semantic validation still belongs to
    ``experiments.common.checkpoints.CheckpointRef`` inside the repo venv.
    """
    raw = checkpoint.strip().replace("\\", "/")
    if not raw:
        raise ValueError("checkpoint must be a non-empty repo-relative path")

    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("checkpoint must be a safe repo-relative path")

    if path.name in {"model.safetensors", "checkpoint.json"}:
        path = path.parent

    parts = path.parts
    if len(parts) != 4 or parts[2] != "checkpoints":
        raise ValueError(
            "checkpoint must have shape "
            "'<ARM>/run-XXXX/checkpoints/step-XXXXXX'"
        )
    return path


def _checkpoint_files(checkpoint: str) -> tuple[PurePosixPath, ...]:
    """Files required by experiments.common.model.load_model()."""
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
    """Download through the persistent HF cache, then copy into job staging."""
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


def _stage_checkpoint(
    *,
    hf_repo: str,
    checkpoint: str,
    staging_root: Path,
    token: str | None,
    revision: str,
) -> list[str]:
    staged: list[str] = []

    for relative in _checkpoint_files(checkpoint):
        relative_text = relative.as_posix()
        destination = staging_root / Path(*relative.parts)
        _hf_download_into(
            repo_id=hf_repo,
            filename=relative_text,
            destination=destination,
            token=token,
            revision=revision,
        )
        staged.append(relative_text)
        print(
            f"[modal/induction] staged {hf_repo}/{relative_text}",
            flush=True,
        )

    return staged


def _ensure_tokenizer(
    *,
    tokenizer_repo: str,
    token: str | None = None,
) -> list[str]:
    """Install the exact tokenizer where nanochat.get_tokenizer() expects it."""
    tokenizer_dir = Path(CACHE_DIR) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    for filename in TOKENIZER_FILES:
        destination = tokenizer_dir / filename
        _hf_download_into(
            repo_id=tokenizer_repo,
            filename=filename,
            destination=destination,
            token=token,
            revision="main",
        )
        installed.append(str(destination))

    print(
        f"[modal/induction] tokenizer ready from {tokenizer_repo} -> "
        f"{tokenizer_dir}",
        flush=True,
    )
    return installed


def _result_files(output_root: Path) -> list[Path]:
    metrics_root = output_root / "metrics" / "induction"
    if not metrics_root.is_dir():
        raise RuntimeError(
            "induction runner completed but produced no metrics/induction directory"
        )

    files = sorted(path for path in metrics_root.rglob("*.json") if path.is_file())
    if not files:
        raise RuntimeError(
            "induction runner completed but produced no induction result JSON files"
        )
    return files


def _upload_results(
    *,
    api: Any,
    hf_repo: str,
    output_root: Path,
    token: str,
    revision: str,
    checkpoint: str,
) -> list[str]:
    files = _result_files(output_root)
    relative_paths = [
        path.relative_to(output_root).as_posix()
        for path in files
    ]

    # output_root is job-specific and contains only artifacts from this
    # invocation. Uploading the folder preserves the canonical metrics/... paths.
    api.upload_folder(
        repo_id=hf_repo,
        repo_type="model",
        revision=revision,
        token=token,
        folder_path=str(output_root),
        path_in_repo=None,
        allow_patterns=["metrics/induction/**/*.json"],
        commit_message=f"induction metrics: {checkpoint}",
    )

    for relative in relative_paths:
        print(
            f"[modal/induction] uploaded {hf_repo}/{relative}",
            flush=True,
        )
    return relative_paths


@app.function(
    image=image,
    gpu=MODAL_GPU,
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    retries=0,
    volumes={CACHE_DIR: cache_volume},
    secrets=[hf_secret],
)
def run_induction(
    hf_repo: str,
    checkpoint: str,
    evaluation_arm: str = "all",
    lengths: str = "32,64,128,256,512",
    offsets: str = "-2,-1,0,1,2",
    logical_batch_size: int = 16,
    micro_batch_size: int = 0,
    max_forward_tokens: int = 8192,
    num_batches: int = 4,
    seed: int = 1234,
    top_k: int = 8,
    behavior_only: bool = False,
    tokenizer_repo: str = DEFAULT_TOKENIZER_REPO,
    revision: str = "main",
    upload: bool = True,
) -> dict[str, Any]:
    """Run induction/DAC on Modal and optionally push metrics to the same HF repo."""
    if not hf_repo.strip():
        raise ValueError("hf_repo must be non-empty")
    if micro_batch_size < 0:
        raise ValueError("micro_batch_size must be >= 0")

    # Structural normalization here; the scientific runner performs canonical
    # CheckpointRef and arm validation inside the repo's GPU environment.
    checkpoint_dir = _canonical_checkpoint_dir(checkpoint)
    checkpoint_text = checkpoint_dir.as_posix()

    token = os.environ.get("HF_TOKEN") or None
    if upload and token is None:
        raise RuntimeError(
            "HF_TOKEN is required when upload=True; provide a write-capable "
            "Hugging Face token to the process running `modal run`"
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Pick up cache writes from previous containers before touching the Volume.
    cache_volume.reload()

    job_root = Path(tempfile.mkdtemp(prefix="nanochat-induction-", dir="/tmp"))
    staging_root = job_root / "hf_repo"
    output_root = job_root / "results"
    staging_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    staged_files = _stage_checkpoint(
        hf_repo=hf_repo,
        checkpoint=checkpoint_text,
        staging_root=staging_root,
        token=token,
        revision=revision,
    )

    # Karpathy's tokenizer repo is public by default, so do not accidentally
    # couple tokenizer access to the permissions of the experiment repo token.
    _ensure_tokenizer(
        tokenizer_repo=tokenizer_repo,
        token=None,
    )

    cmd = [
        f"{VENV}/bin/python",
        "-m",
        "experiments.induction.run",
        "--repo-root",
        str(staging_root),
        "--output-root",
        str(output_root),
        "--checkpoint",
        checkpoint_text,
        "--evaluation-arm",
        evaluation_arm,
        "--lengths",
        lengths,
        f"--offsets={offsets}",
        "--logical-batch-size",
        str(logical_batch_size),
        "--micro-batch-size",
        str(micro_batch_size),
        "--max-forward-tokens",
        str(max_forward_tokens),
        "--num-batches",
        str(num_batches),
        "--seed",
        str(seed),
        "--top-k",
        str(top_k),
        "--device",
        "cuda",
        "--weights-device",
        "cpu",
    ]
    if behavior_only:
        cmd.append("--behavior-only")

    _run_streamed(cmd)

    local_results = [
        path.relative_to(output_root).as_posix()
        for path in _result_files(output_root)
    ]

    uploaded: list[str] = []
    if upload:
        assert token is not None
        uploaded = _upload_results(
            api=api,
            hf_repo=hf_repo,
            output_root=output_root,
            token=token,
            revision=revision,
            checkpoint=checkpoint_text,
        )

    # Persist HF downloads and tokenizer files for future jobs.
    cache_volume.commit()

    return {
        "hf_repo": hf_repo,
        "revision": revision,
        "checkpoint": checkpoint_text,
        "evaluation_arm": evaluation_arm,
        "gpu": MODAL_GPU,
        "staged_files": staged_files,
        "result_files": local_results,
        "uploaded_files": uploaded,
        "upload": upload,
    }


@app.local_entrypoint()
def main(
    hf_repo: str,
    checkpoint: str,
    evaluation_arm: str = "all",
    lengths: str = "32,64,128,256,512",
    offsets: str = "-2,-1,0,1,2",
    logical_batch_size: int = 16,
    micro_batch_size: int = 0,
    max_forward_tokens: int = 8192,
    num_batches: int = 4,
    seed: int = 1234,
    top_k: int = 8,
    behavior_only: bool = False,
    tokenizer_repo: str = DEFAULT_TOKENIZER_REPO,
    revision: str = "main",
    upload: bool = True,
) -> None:
    """CLI entrypoint used locally or from GitHub Actions."""
    result = run_induction.remote(
        hf_repo=hf_repo,
        checkpoint=checkpoint,
        evaluation_arm=evaluation_arm,
        lengths=lengths,
        offsets=offsets,
        logical_batch_size=logical_batch_size,
        micro_batch_size=micro_batch_size,
        max_forward_tokens=max_forward_tokens,
        num_batches=num_batches,
        seed=seed,
        top_k=top_k,
        behavior_only=behavior_only,
        tokenizer_repo=tokenizer_repo,
        revision=revision,
        upload=upload,
    )

    print(json.dumps(result, indent=2, sort_keys=True))
