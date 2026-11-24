"""LensRank encoding, evaluation and resumable experiment suite."""

from __future__ import annotations

import json
import os
import random
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .adapters import LoRALinear, inject_lora
from .checkpoints import (
    _remove_empty_directory_artifact,
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    run_lock,
)
from .config import EXPERIMENTS, SuiteConfig
from .data import (
    ImageBatchSampler,
    PairDataset,
    collate_pairs,
    read_table,
    validate_splits,
)
from .provenance import (
    data_identity,
    execution_fingerprint,
    file_sha256,
    resolve_pretrained,
    run_identity,
)
from .retrieval import retrieval_recall

__all__ = [
    'EXPERIMENTS', 'LoRALinear', 'PairDataset', 'RetrievalSuite', 'SuiteConfig',
    '_remove_empty_directory_artifact', 'atomic_json_save', 'atomic_torch_save',
    'inject_lora', 'seed_all',
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RetrievalSuite:

    def __init__(self, cfg: SuiteConfig, *, verify_images=False):
        self.cfg = cfg
        self.data = Path(cfg.data_dir).resolve()
        self.runs = Path(cfg.run_dir).resolve()
        for field in ('batch_size', 'eval_batch_size', 'checkpoint_every_batches'):
            if getattr(cfg, field) < 1:
                raise ValueError(f'{field} must be positive')
        self.train = read_table(self.data, 'train', ('caption_id', 'image_id'))
        self.val = read_table(self.data, 'val', ('caption_id', 'image_id'))
        self.test = read_table(self.data, 'test', ('id', 'query'), ('query_type',))
        self.captions_df = read_table(
            self.data, 'caption_pool', ('caption_id', 'image_id'), ('caption',), empty_caption=True,
        )
        self.images_df = read_table(self.data, 'image_pool', ('image_id',), ('image_name',))
        self.captions = self.captions_df.set_index('caption_id').caption.to_dict()
        self.image_names = self.images_df.set_index('image_id').image_name.to_dict()
        self.directories = {split: self.data / f'{split}_images' / f'{split}_images'
                            for split in ('train', 'val', 'test')}
        self.train_dir = self.directories['train']
        self.val_dir = self.directories['val']
        self._validate_data()
        self.data_identity = data_identity(
            self.data, self.directories, self.splits, self.image_names, verify_images=verify_images,
        )

    def _validate_data(self):
        self.splits, self.caption_owners = validate_splits(
            self.train, self.val, self.test, self.images_df, self.captions_df, self.directories,
        )

    def _build_model(self, spec, *, weights_path=None, checkpoint_only=False):
        pretrained = self.cfg.pretrained
        options = {}
        if weights_path is not None or checkpoint_only:
            metadata = open_clip.get_pretrained_cfg(self.cfg.model_name, str(pretrained or '')) or {}
            options = {f'image_{key}': metadata[key] for key in ('mean', 'std', 'interpolation', 'resize_mode')
                       if key in metadata}
            pretrained = str(weights_path) if weights_path is not None else None
            options['pretrained_hf'] = False
        model, _, preprocess = open_clip.create_model_and_transforms(self.cfg.model_name, pretrained=pretrained, **options)
        tokenizer = open_clip.get_tokenizer(self.cfg.model_name)
        strategy = spec['strategy']
        if strategy in {'zero_shot', 'projection', 'lora', 'gradual'}:
            for p in model.parameters():
                p.requires_grad = False
        if strategy in {'projection', 'gradual'}:
            for n, p in model.named_parameters():
                if any(x in n for x in ('visual.proj', 'text_projection', 'ln_post', 'ln_final', 'logit_scale')):
                    p.requires_grad = True
        elif strategy == 'lora':
            count = inject_lora(model, spec.get('lora_rank', 8), spec.get('lora_alpha', 16))
            model.logit_scale.requires_grad = True
            if count == 0:
                raise RuntimeError('No attention projections received LoRA adapters')
        elif strategy == 'full':
            for p in model.parameters():
                p.requires_grad = True
        return (model.to(self.cfg.device), preprocess, tokenizer)

    def _loader(self, dataset, tokenizer, epoch, start_batch=0):
        generator = torch.Generator().manual_seed(self.cfg.seed + epoch)
        return DataLoader(
            dataset,
            batch_sampler=ImageBatchSampler(
                dataset.pairs.image_id, self.cfg.batch_size, self.cfg.seed, epoch, start_batch,
            ),
            generator=generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=partial(collate_pairs, tokenizer=tokenizer),
        )

    @torch.no_grad()
    def _encode_images(self, model, preprocess, ids, directory):
        model.eval()
        output = []
        for start in range(0, len(ids), self.cfg.eval_batch_size):
            batch = []
            for i in ids[start:start + self.cfg.eval_batch_size]:
                with Image.open(Path(directory) / Path(self.image_names[i]).name) as image:
                    batch.append(preprocess(image.convert('RGB')))
            z = model.encode_image(torch.stack(batch).to(self.cfg.device))
            output.append(F.normalize(z, dim=-1).cpu())
        return torch.cat(output)

    @torch.no_grad()
    def _encode_text(self, model, tokenizer, ids):
        model.eval()
        output = []
        for start in range(0, len(ids), self.cfg.eval_batch_size):
            tok = tokenizer([self.captions[i] for i in ids[start:start + self.cfg.eval_batch_size]]).to(self.cfg.device)
            output.append(F.normalize(model.encode_text(tok), dim=-1).cpu())
        return torch.cat(output)

    @torch.no_grad()
    def evaluate(self, model, preprocess, tokenizer):
        return self._evaluate_split(model, preprocess, tokenizer, 'val')

    @torch.no_grad()
    def _evaluate_split(self, model, preprocess, tokenizer, split, image_queries=None, caption_queries=None):
        image_ids, caption_ids = map(sorted, self.splits[split])
        ie = self._encode_images(model, preprocess, image_ids, self.directories[split])
        te = self._encode_text(model, tokenizer, caption_ids)
        return retrieval_recall(ie, te, image_ids, caption_ids, self.caption_owners,
                                image_queries, caption_queries)

    def run_one(self, name: str, spec: dict):
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError(f'Invalid experiment name: {name!r}')
        spec = dict(spec)
        strategy = spec['strategy']
        spec['epochs'] = 0 if strategy == 'zero_shot' else spec.get('epochs', self.cfg.epochs)
        if strategy not in {item['strategy'] for item in EXPERIMENTS.values()}:
            raise ValueError(f'Unknown strategy: {strategy}')
        if strategy != 'zero_shot' and spec['epochs'] < 1:
            raise ValueError('Training requires at least one epoch')
        if strategy == 'gradual':
            spec['unfreeze_epoch'] = max(2, spec['epochs'] // 2)
        if strategy == 'lora':
            spec.setdefault('lora_rank', 8)
            spec.setdefault('lora_alpha', 16)

        folder = self.runs / name
        self._read_run_marker(folder)
        with run_lock(folder):
            recorded = self._read_run_marker(folder)
            source, weights_path = resolve_pretrained(self.cfg, download=recorded is None)
            identity = run_identity(self.cfg, name, spec, self.data_identity, source)
            marker = folder / 'run.json'
            if recorded is None:
                atomic_json_save(identity, marker)
            else:
                if execution_fingerprint(recorded) != execution_fingerprint(identity):
                    raise ValueError(f'Run identity mismatch at {marker}. Use a separate run_dir.')
                identity = recorded
            try:
                return self._run_one(name, spec, folder, identity, weights_path)
            except Exception as exc:
                atomic_json_save({'experiment': name, 'status': 'failed', 'error': repr(exc)},
                                 folder / 'failure.json')
                raise

    @staticmethod
    def _read_run_marker(folder):
        marker = folder / 'run.json'
        if marker.is_file():
            return json.loads(marker.read_text(encoding='utf-8'))
        if folder.exists() and any(path.name not in {'.run.lock', 'run.json.tmp'} for path in folder.iterdir()):
            raise ValueError(f'Legacy or unidentified run at {folder}. Use a separate run_dir.')

    @staticmethod
    def _verify_identity(recorded, expected, path):
        if recorded != expected:
            raise ValueError(f'Run identity mismatch at {path}. Use a separate run_dir.')

    @staticmethod
    def _restore_best(folder, selection):
        if selection is None:
            return
        best = folder / 'best.pt'
        staged = folder / 'best.pending.pt'
        for path in (best, staged):
            if path.is_file() and file_sha256(path) == selection['sha256']:
                if path == staged:
                    os.replace(staged, best)
                return
        raise RuntimeError(f'Best checkpoint for epoch {selection["epoch"]} is missing or changed in {folder}')

    def _run_one(self, name, spec, folder, identity, weights_path):
        done = folder / 'completed.json'
        last = folder / 'last.pt'
        if done.exists():
            result = json.loads(done.read_text(encoding='utf-8'))
            self._verify_identity(result.get('run_identity'), identity, done)
            if result['checkpoint']['kind'] == 'trained':
                self._restore_best(folder, result['checkpoint'])
            print(f'[skip] {name} is complete')
            return result

        state = None
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            self._verify_identity(state.get('run_identity'), identity, last)
            self._restore_best(folder, state['best_checkpoint'])
        seed_all(self.cfg.seed)
        model, preprocess, tokenizer = self._build_model(spec, weights_path=weights_path)
        epochs = spec['epochs']
        unfreeze_epoch = spec.get('unfreeze_epoch')
        history = state['history'] if state else []
        best = state['best'] if state else -1.0
        best_checkpoint = state['best_checkpoint'] if state else None
        start_epoch = state['epoch'] if state else 1
        start_batch = state['next_batch'] if state else 0
        stages = state['optimizer_stages'] if state else []
        if state:
            model.load_state_dict(state['model'])
            print(f'[resume] {name}: epoch {start_epoch}, batch {start_batch}')
        optimizer = None

        def make_optimizer(first_epoch, unfreeze=False):
            if unfreeze:
                model.requires_grad_(True)
            parameters = {key: value for key, value in model.named_parameters() if value.requires_grad}
            if not parameters:
                raise RuntimeError(f'Experiment {name!r} selected no trainable parameters')
            if not stages or stages[-1]['first_epoch'] != first_epoch:
                stages.append({'first_epoch': first_epoch, 'names': list(parameters),
                               'parameter_count': sum(p.numel() for p in parameters.values())})
            return torch.optim.AdamW(parameters.values(), lr=spec['lr'], weight_decay=0.0001)

        if epochs:
            if unfreeze_epoch is not None and start_epoch >= unfreeze_epoch:
                optimizer = make_optimizer(unfreeze_epoch, unfreeze=True)
            else:
                optimizer = make_optimizer(1)
            if state and not (start_epoch == unfreeze_epoch and start_batch == 0):
                optimizer.load_state_dict(state['optimizer'])
        cuda = torch.device(self.cfg.device).type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=cuda and optimizer is not None)
        if state:
            scaler.load_state_dict(state['scaler'])
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if cuda:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.cfg.device)
        started = perf_counter()
        prior_resources = state['resources'] if state else None

        def resources():
            peak = None
            elapsed = perf_counter() - started
            if prior_resources:
                elapsed += prior_resources['elapsed_seconds']
            if cuda:
                peak = torch.cuda.max_memory_allocated(self.cfg.device)
                if prior_resources:
                    peak = max(peak, prior_resources['cuda_peak_allocated_bytes'])
            return {
                'elapsed_seconds': elapsed,
                'scope': 'training and validation after setup, including data loading and in-loop persistence',
                'cuda_peak_allocated_bytes': peak,
                'cuda_peak_api': 'torch.cuda.max_memory_allocated' if cuda else None,
                'cuda_status': 'measured' if cuda else 'not applicable on CPU',
                'resume_rule': 'sum elapsed time and take maximum CUDA peak across checkpointed segments',
            }

        def save(epoch, next_batch, loss_sum, loss_count):
            atomic_torch_save({
                'run_identity': identity, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'next_batch': next_batch, 'history': history, 'best': best,
                'best_checkpoint': best_checkpoint, 'optimizer_stages': stages,
                'loss_sum': loss_sum, 'loss_count': loss_count, 'resources': resources(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if cuda else None,
            }, last)

        if epochs == 0:
            metrics = self.evaluate(model, preprocess, tokenizer)
            history = [{'epoch': 0, 'train_loss': None, **metrics}]
            best_checkpoint = {'kind': 'pretrained', 'epoch': 0, **identity['initial_weights']}
        else:
            pairs = self.train.head(512) if self.cfg.fast_dev_run else self.train
            dataset = PairDataset(pairs, self.train_dir, self.image_names, self.captions, preprocess)
            for epoch in range(start_epoch, epochs + 1):
                if epoch == unfreeze_epoch and epoch != start_epoch:
                    optimizer = make_optimizer(epoch, unfreeze=True)
                resume_batch = start_batch if epoch == start_epoch else 0
                loss_sum = state['loss_sum'] if state and epoch == start_epoch else 0.0
                loss_count = state['loss_count'] if state and epoch == start_epoch else 0
                loader = self._loader(dataset, tokenizer, epoch, resume_batch)
                model.train()
                for bi, (images, tokens) in enumerate(tqdm(loader, desc=f'{name} e{epoch}'), resume_batch):
                    if len(images) > 1:
                        images, tokens = images.to(self.cfg.device), tokens.to(self.cfg.device)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', enabled=cuda):
                            ie = F.normalize(model.encode_image(images), dim=-1)
                            te = F.normalize(model.encode_text(tokens), dim=-1)
                            logits = model.logit_scale.exp().clamp(max=100) * (ie @ te.T)
                            labels = torch.arange(len(images), device=self.cfg.device)
                            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        loss_sum += loss.item()
                        loss_count += 1
                    if (bi + 1) % self.cfg.checkpoint_every_batches == 0:
                        save(epoch, bi + 1, loss_sum, loss_count)
                metrics = self.evaluate(model, preprocess, tokenizer)
                record = {'epoch': epoch, 'train_loss': loss_sum / loss_count if loss_count else None, **metrics}
                history.append(record)
                improved = metrics['mean_score'] > best
                if improved:
                    best = metrics['mean_score']
                    staged = folder / 'best.pending.pt'
                    atomic_torch_save({'model': model.state_dict(), 'metrics': metrics, 'epoch': epoch,
                                       'run_id': identity['id']}, staged)
                    best_checkpoint = {'kind': 'trained', 'epoch': epoch, 'file': 'best.pt',
                                       'sha256': file_sha256(staged)}
                save(epoch + 1, 0, 0.0, 0)
                if improved:
                    os.replace(staged, folder / 'best.pt')
                atomic_csv_save(pd.DataFrame(history), folder / 'history.csv')
        if start_epoch > epochs:
            atomic_csv_save(pd.DataFrame(history), folder / 'history.csv')
        best_record = max(history, key=lambda x: x['mean_score'])
        if best_checkpoint['epoch'] != best_record['epoch']:
            raise RuntimeError('Best checkpoint and validation history select different epochs')
        result = {
            'experiment': name,
            'strategy': spec['strategy'],
            'status': 'completed',
            'best_epoch': best_record['epoch'],
            'trainable_parameters': stages[-1]['parameter_count'] if stages else 0,
            'run_identity': identity,
            'checkpoint': best_checkpoint,
            'optimizer_stages': stages,
            'resources': resources(),
            **best_record,
        }
        atomic_json_save(result, done)
        return result

    def run_all(self, experiments=None):
        experiments = experiments or EXPERIMENTS
        results = []
        comparison = self.runs / 'fresh_comparison.csv'

        def verify_comparison():
            if comparison.exists() and 'run_id' not in pd.read_csv(comparison, nrows=0):
                raise ValueError(f'Legacy comparison at {comparison}. Use a separate run_dir.')

        verify_comparison()
        with run_lock(self.runs):
            verify_comparison()
            for name, spec in experiments.items():
                try:
                    results.append(self.run_one(name, spec))
                finally:
                    if torch.device(self.cfg.device).type == 'cuda':
                        torch.cuda.empty_cache()
            details = {'run_identity', 'optimizer_stages', 'checkpoint', 'resources'}
            rows = [
                {**{key: value for key, value in result.items() if key not in details},
                 'run_id': result['run_identity']['id'],
                 'elapsed_seconds': result['resources']['elapsed_seconds'],
                 'cuda_peak_allocated_bytes': result['resources']['cuda_peak_allocated_bytes']}
                for result in results
            ]
            table = pd.DataFrame(rows).sort_values('mean_score', ascending=False, kind='stable')
            atomic_csv_save(table, comparison)
            return table
