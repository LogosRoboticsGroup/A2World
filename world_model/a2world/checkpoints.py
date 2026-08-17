import argparse
import hashlib
import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

EXPECTED_ACTION_DIM = 280
REQUIRED_TENSORS = {
    "net.action_embedder_B_D_agi.fc1.weight": (8192, EXPECTED_ACTION_DIM),
    "net.action_embedder_B_D_agi.fc2.weight": (2048, 8192),
    "net.view_embeddings.weight": (4, 7),
}

SENSITIVE_PICKLE_PATTERNS = {
    "absolute filesystem path": re.compile(rb"(?:/home/|/root/|/inspire/|/workspace/|[A-Za-z]:\\\\)"),
    "dated run identifier": re.compile(rb"20\d{2}[-_]\d{2}[-_]\d{2}[_T-]\d{2}[-:]\d{2}[-:]\d{2}"),
    "iteration-labelled checkpoint": re.compile(rb"iter(?:ation)?[_-]?\d{3,}", re.IGNORECASE),
    "training metadata": re.compile(
        rb"(?:accum_iteration|accum_train_in_hours|global_step|source_checkpoint|source_run|wandb)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    size_bytes: int
    tensor_count: int
    key_count: int
    dtypes: tuple[str, ...]
    variant: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().absolute()
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)

    checkpoint_dir = checkpoint_path / "checkpoints"
    if not checkpoint_dir.is_dir():
        checkpoint_dir = checkpoint_path
    latest_path = checkpoint_dir / "latest_checkpoint.txt"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint pointer: {latest_path}")
    checkpoint_name = latest_path.read_text().strip()
    if not checkpoint_name:
        raise ValueError(f"Empty checkpoint pointer: {latest_path}")
    model_path = checkpoint_dir / "model" / checkpoint_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    return model_path.absolute()


def load_state_dict(path: str | Path) -> Mapping[str, Any]:
    checkpoint_path = resolve_checkpoint_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected a mapping checkpoint, got {type(state_dict).__name__}")
    return state_dict


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    checkpoint_path = resolve_checkpoint_path(path)
    state_dict = load_state_dict(checkpoint_path)
    for key, expected_shape in REQUIRED_TENSORS.items():
        if key not in state_dict:
            raise KeyError(f"Missing A2World tensor: {key}")
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint value is not a tensor: {key}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: {tuple(value.shape)} != {expected_shape}")

    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    variant = "libero" if any(key.startswith("net.history_tokenizer.") for key in state_dict) else "pretrained"
    return CheckpointInfo(
        path=str(checkpoint_path),
        size_bytes=checkpoint_path.stat().st_size,
        tensor_count=len(tensors),
        key_count=len(state_dict),
        dtypes=tuple(sorted({str(value.dtype) for value in tensors})),
        variant=variant,
    )


def audit_checkpoint_privacy(path: str | Path) -> None:
    checkpoint_path = resolve_checkpoint_path(path)
    state_dict = load_state_dict(checkpoint_path)
    unsafe_entries = []
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith("net."):
            unsafe_entries.append(str(key))
        elif key.endswith("._extra_state"):
            unsafe_entries.append(key)
        if not isinstance(value, torch.Tensor):
            unsafe_entries.append(key)
    if unsafe_entries:
        preview = ", ".join(unsafe_entries[:5])
        raise ValueError(f"Checkpoint contains non-release entries: {preview}")

    try:
        with zipfile.ZipFile(checkpoint_path) as archive:
            pickle_members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if len(pickle_members) != 1:
                raise ValueError(f"Expected one data.pkl member, found {len(pickle_members)}")
            pickle_payload = archive.read(pickle_members[0])
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Checkpoint is not a PyTorch ZIP archive: {checkpoint_path}") from exc

    findings = [label for label, pattern in SENSITIVE_PICKLE_PATTERNS.items() if pattern.search(pickle_payload)]
    if findings:
        raise ValueError(f"Checkpoint privacy audit failed: {', '.join(findings)}")


def sha256sum(path: str | Path) -> str:
    checkpoint_path = resolve_checkpoint_path(path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(checkpoint_infos: list[CheckpointInfo], manifest_path: str | Path, verify_sha256: bool) -> None:
    manifest_file = Path(manifest_path).expanduser().absolute()
    with manifest_file.open() as file:
        manifest = json.load(file)
    artifacts = {artifact["filename"]: artifact for artifact in manifest["artifacts"]}

    for info in checkpoint_infos:
        checkpoint_path = Path(info.path)
        artifact = artifacts.get(checkpoint_path.name)
        if artifact is None:
            raise KeyError(f"Checkpoint is not listed in {manifest_file}: {checkpoint_path.name}")
        if artifact.get("size_bytes") is not None and info.size_bytes != artifact["size_bytes"]:
            raise ValueError(
                f"Unexpected size for {checkpoint_path.name}: {info.size_bytes} != {artifact['size_bytes']}"
            )
        if verify_sha256 and artifact.get("sha256"):
            actual_sha256 = sha256sum(checkpoint_path)
            if actual_sha256 != artifact["sha256"]:
                raise ValueError(
                    f"Unexpected SHA-256 for {checkpoint_path.name}: {actual_sha256} != {artifact['sha256']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate A2World checkpoint structure")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, help="Validate artifact names and sizes against weights.json")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Also hash each checkpoint; requires --manifest and reads every checkpoint in full",
    )
    parser.add_argument(
        "--audit-privacy",
        action="store_true",
        help="Require tensor-only net.* entries and reject embedded paths, timestamps, run IDs, and iteration labels",
    )
    args = parser.parse_args()
    if args.verify_sha256 and args.manifest is None:
        parser.error("--verify-sha256 requires --manifest")
    checkpoint_infos = [inspect_checkpoint(path) for path in args.checkpoints]
    if args.manifest is not None:
        validate_manifest(checkpoint_infos, args.manifest, args.verify_sha256)
    results = [info.to_dict() for info in checkpoint_infos]
    if args.audit_privacy:
        for info, result in zip(checkpoint_infos, results, strict=True):
            audit_checkpoint_privacy(info.path)
            result["privacy_audit"] = "passed"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
