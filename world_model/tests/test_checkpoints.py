from pathlib import Path

import pytest
import torch

from a2world.checkpoints import audit_checkpoint_privacy


def save_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    torch.save(state_dict, path)


def test_privacy_audit_accepts_tensor_only_release_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "release.pt"
    save_checkpoint(checkpoint_path, {"net.weight": torch.ones(2)})

    audit_checkpoint_privacy(checkpoint_path)


@pytest.mark.parametrize(
    "state_dict",
    [
        {"net.weight": torch.ones(2), "net.norm._extra_state": torch.empty(0)},
        {"optimizer.state": torch.ones(2)},
        {"net./inspire/private/weight": torch.ones(2)},
    ],
)
def test_privacy_audit_rejects_release_metadata(
    tmp_path: Path,
    state_dict: dict[str, torch.Tensor],
) -> None:
    checkpoint_path = tmp_path / "unsafe.pt"
    save_checkpoint(checkpoint_path, state_dict)

    with pytest.raises(ValueError):
        audit_checkpoint_privacy(checkpoint_path)
