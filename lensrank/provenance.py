"""Stable data, model and run identities."""

import hashlib
import json
import platform
import re
import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

import open_clip
import torch
from huggingface_hub import try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[1]
TABLES = ('train', 'val', 'test', 'image_pool', 'caption_pool')


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def data_identity(directory, directories, splits, names):
    manifest_path = directory / 'manifest.json'
    manifest = {}
    if manifest_path.is_file():
        for item in json.loads(manifest_path.read_text(encoding='utf-8')):
            path = item['path'].removeprefix('data/')
            manifest[path] = item
    tables = {name: file_sha256(directory / f'{name}.csv') for name in TABLES}
    identities = {}
    for split, (images, captions) in splits.items():
        files = {}
        manifest_count = 0
        for image in sorted(images):
            path = directories[split] / Path(names[image]).name
            relative = path.relative_to(directory).as_posix()
            stat = path.stat()
            if relative in manifest:
                record = manifest[relative]
                if record.get('bytes', stat.st_size) != stat.st_size:
                    raise ValueError(f'Image size differs from the data manifest: {path}')
                files[relative] = record['sha256']
                manifest_count += 1
            else:
                files[relative] = [stat.st_size, stat.st_mtime_ns]
        identities[split] = {
            'table_sha256': tables[split],
            'image_ids_sha256': digest(sorted(images)),
            'caption_ids_sha256': digest(sorted(captions)),
            'image_files_sha256': digest(files),
            'manifest_image_hashes': manifest_count,
            'image_count': len(images),
        }
    return {'tables_sha256': tables, 'splits': identities}


def resolve_pretrained(cfg, download=False):
    tag = str(cfg.pretrained) if cfg.pretrained else ''
    source = {'model': cfg.model_name, 'tag': tag, 'sha256': None, 'revision': None}
    if not tag:
        return {**source, 'verification': 'random initialization controlled by seed'}, None
    if Path(tag).is_file():
        path = Path(tag).resolve()
        return {**source, 'sha256': file_sha256(path), 'verification': 'local file SHA-256'}, path
    metadata = open_clip.get_pretrained_cfg(cfg.model_name, tag) or {}
    path = None
    hub = metadata.get('hf_hub')
    url = metadata.get('url', '')
    if hub:
        repo, _, filename = hub.rpartition('/')
        filename = filename or 'open_clip_pytorch_model.bin'
        if filename == 'open_clip_pytorch_model.bin':
            safe = 'open_clip_model.safetensors'
        else:
            safe = re.sub(r'\.(bin|pth)$', '.safetensors', filename)
        for candidate in dict.fromkeys((safe, filename)):
            cached = try_to_load_from_cache(repo, candidate)
            if isinstance(cached, str):
                path = Path(cached)
                break
    elif url:
        cached = Path.home() / '.cache' / 'clip' / Path(urlsplit(url).path).name
        if cached.is_file():
            path = cached
    if path is None and download and metadata:
        path = Path(open_clip.download_pretrained(metadata))
    if path is not None and path.is_file():
        revision = path.parent.name if path.parent.parent.name == 'snapshots' else None
        return {**source, 'sha256': file_sha256(path), 'revision': revision,
                'verification': 'resolved OpenCLIP weight file SHA-256'}, path
    match = re.search(r'/([a-f0-9]{64})/[^/]+$', url)
    if match and 'openaipublic.azureedge.net' in url and not hub:
        return {**source, 'sha256': match[1], 'verification': 'OpenCLIP download checksum'}, None
    return {**source, 'verification': 'weight revision and checksum unavailable'}, None


def _git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def run_identity(cfg, name, spec, dataset, initial_weights):
    lock = ROOT / 'requirements-lock.txt'
    device = torch.device(cfg.device)
    config = {**asdict(cfg), 'data_dir': str(Path(cfg.data_dir).resolve()),
              'run_dir': str(Path(cfg.run_dir).resolve()), 'device': str(device),
              'pretrained': str(cfg.pretrained) if cfg.pretrained else ''}
    details = {
        'schema_version': 1,
        'experiment': name,
        'config': config,
        'spec': spec,
        'initial_weights': initial_weights,
        'data': dataset,
        'code': {
            'git_revision': _git_revision(),
            'sources_sha256': digest({path.name: file_sha256(path)
                                      for path in (ROOT / 'lensrank').glob('*.py')}),
        },
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'packages': {name: _package_version(name) for name in
                         ('torch', 'torchvision', 'open_clip_torch', 'numpy', 'pandas', 'pillow')},
            'requirements_lock_sha256': file_sha256(lock) if lock.is_file() else None,
            'cuda': torch.version.cuda if device.type == 'cuda' else None,
            'accelerator': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        },
    }
    return {'id': digest(details), **details}
