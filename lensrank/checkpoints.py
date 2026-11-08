"""Atomic checkpoint and JSON persistence."""

import json
import os
from pathlib import Path
import torch


def _remove_empty_directory_artifact(path: Path) -> None:
    if path.is_dir():
        try:
            path.rmdir()
        except OSError as exc:
            raise IsADirectoryError(f'{path} is a non-empty directory, but this path must be a file. Move or remove it before resuming.') from exc


def atomic_torch_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    _remove_empty_directory_artifact(path)
    _remove_empty_directory_artifact(tmp)
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    _remove_empty_directory_artifact(path)
    _remove_empty_directory_artifact(tmp)
    tmp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(tmp, path)
