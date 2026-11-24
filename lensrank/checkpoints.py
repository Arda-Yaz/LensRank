"""Atomic checkpoint and JSON persistence."""

import json
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch


@contextmanager
def run_lock(folder):
    """Keep simultaneous writers out and release the lock on process exit."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.run.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt

            lock = partial(msvcrt.locking, stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = partial(msvcrt.locking, stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            lock = partial(fcntl.flock, stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = partial(fcntl.flock, stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError(f'Another writer holds the run lock for {folder}') from exc
        try:
            yield
        finally:
            unlock()


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
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv_save(table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    table.to_csv(tmp, index=False)
    os.replace(tmp, path)
