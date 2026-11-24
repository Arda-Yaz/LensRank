"""Freeze a validation winner, then evaluate the frozen selection on local test data."""

import argparse
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch

from .config import EXPERIMENTS, SuiteConfig
from .experiments import RetrievalSuite, seed_all
from .provenance import file_sha256, resolve_pretrained


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')


def freeze_selection(run_dir, *, experiment=None):
    """Lock a validation checkpoint for the winner or a preselected strategy."""
    run_dir = Path(run_dir).resolve()
    comparison = run_dir / 'fresh_comparison.csv'
    comparison_bytes = comparison.read_bytes()
    table = pd.read_csv(BytesIO(comparison_bytes))
    if table.empty or not {'experiment', 'mean_score', 'best_epoch'} <= set(table.columns):
        raise ValueError(f'No usable validation comparison at {comparison}')
    if not table.mean_score.map(math.isfinite).all() or table.experiment.duplicated().any():
        raise ValueError(f'Ambiguous validation comparison at {comparison}')
    if experiment is not None:
        table = table.loc[table.experiment.eq(experiment)]
        if table.empty:
            raise ValueError(f'No validation result for {experiment!r} at {comparison}')
    winner = table.sort_values('mean_score', ascending=False, kind='stable').iloc[0]
    name = str(winner['experiment'])
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError(f'Invalid experiment name: {name!r}')
    folder = run_dir / name
    completed = folder / 'completed.json'
    result = _read_json(completed)
    epoch = int(winner['best_epoch'])
    score = float(winner['mean_score'])
    if (result.get('status') != 'completed' or result.get('experiment') != name
            or result.get('best_epoch') != epoch or result.get('epoch') != epoch
            or not math.isclose(result['mean_score'], score, abs_tol=1e-12)):
        raise ValueError(f'Validation selection disagrees with {completed}')

    identity = result.get('run_identity')
    if identity:
        if winner.get('run_id') != identity['id'] or _read_json(folder / 'run.json') != identity:
            raise ValueError(f'Run identity disagrees with validation selection at {completed}')
        cfg = SuiteConfig(**identity['config'])
        spec = identity['spec']
        checkpoint = result['checkpoint']
        if checkpoint['epoch'] != epoch:
            raise ValueError(f'Checkpoint epoch disagrees with validation selection at {completed}')
    else:
        if name not in EXPERIMENTS or 'run_id' in table:
            raise ValueError(f'Unknown legacy validation experiment: {name!r}')
        cfg = SuiteConfig()
        spec = EXPERIMENTS[name]
        checkpoint = None
    if result['strategy'] != spec['strategy']:
        raise ValueError(f'Strategy disagrees with validation selection at {completed}')

    if spec['strategy'] == 'zero_shot':
        if epoch != 0 or not cfg.pretrained:
            raise ValueError('Zero-shot final evaluation requires pretrained weights and epoch 0')
        if checkpoint is None:
            source, _ = resolve_pretrained(cfg)
            checkpoint = {'kind': 'pretrained', 'epoch': 0, **source}
        if checkpoint['kind'] != 'pretrained':
            raise ValueError('Zero-shot checkpoint must identify pretrained weights')
    else:
        path = folder / 'best.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Trained validation winner requires its checkpoint: {path}')
        sha256 = file_sha256(path)
        if checkpoint and (checkpoint['kind'] != 'trained' or checkpoint['sha256'] != sha256):
            raise ValueError(f'Checkpoint identity disagrees with validation selection: {path}')
        checkpoint = {'kind': 'trained', 'epoch': epoch, 'path': str(path), 'sha256': sha256}

    return {
        'schema_version': 1, 'selected_on': 'validation', 'experiment': name,
        'epoch': epoch, 'validation_mean_score': score, 'spec': spec,
        'model': cfg.model_name, 'pretrained': cfg.pretrained, 'seed': cfg.seed,
        'run_id': identity['id'] if identity else None,
        'data_identity': identity['data'] if identity else None,
        'checkpoint': checkpoint,
        'validation_comparison_sha256': hashlib.sha256(comparison_bytes).hexdigest(),
    }


def evaluate_frozen(selection, data_dir, device, eval_batch_size):
    if selection.get('schema_version') != 1 or selection.get('selected_on') != 'validation':
        raise ValueError('Expected a frozen validation selection created by the select command')
    spec = selection['spec']
    checkpoint = selection['checkpoint']
    if checkpoint['epoch'] != selection['epoch']:
        raise ValueError('Frozen checkpoint epoch disagrees with the selected epoch')
    zero_shot = spec['strategy'] == 'zero_shot'
    if checkpoint['kind'] != ('pretrained' if zero_shot else 'trained'):
        raise ValueError('Frozen checkpoint kind disagrees with the selected strategy')
    cfg = SuiteConfig(data_dir=str(data_dir), model_name=selection['model'], pretrained=selection['pretrained'],
                      device=device, seed=selection['seed'], eval_batch_size=eval_batch_size)
    suite = RetrievalSuite(cfg, verify_images=True)
    if selection['data_identity'] and suite.data_identity != selection['data_identity']:
        raise ValueError('Dataset identity differs from the frozen validation run')
    weights = None
    weights_path = None
    if zero_shot:
        source, weights_path = resolve_pretrained(cfg, download=True)
        for key in ('model', 'tag', 'sha256', 'revision'):
            if checkpoint[key] is not None and checkpoint[key] != source[key]:
                raise ValueError('Pretrained weight identity differs from the frozen selection')
        checkpoint = {**checkpoint, 'evaluated_weights': source}
    else:
        path = Path(checkpoint['path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != checkpoint['sha256']:
                raise ValueError(f'Checkpoint changed after validation selection: {path}')
            stream.seek(0)
            weights = torch.load(stream, map_location='cpu', weights_only=False)
        if (not math.isclose(weights['metrics']['mean_score'], selection['validation_mean_score'], abs_tol=1e-12)
                or weights.get('epoch', selection['epoch']) != selection['epoch']
                or weights.get('run_id') != selection['run_id']):
            raise ValueError(f'Checkpoint metadata disagrees with frozen validation selection: {path}')
        if selection['run_id'] and 'epoch' not in weights:
            raise ValueError(f'Checkpoint is missing its selected epoch: {path}')
    seed_all(cfg.seed)
    model, preprocess, tokenizer = suite._build_model(spec, weights_path=weights_path, checkpoint_only=not zero_shot)
    if weights is not None:
        model.load_state_dict(weights['model'])
        del weights

    image_queries = suite.test.loc[suite.test.query_type.eq('image'), 'query'].tolist()
    caption_queries = suite.test.loc[suite.test.query_type.eq('caption'), 'query'].tolist()
    image_ids, caption_ids = suite.splits['test']
    scores = suite._evaluate_split(model, preprocess, tokenizer, 'test', image_queries, caption_queries)
    return {
        'split': 'test', 'experiment': selection['experiment'], 'epoch': selection['epoch'],
        'run_id': selection['run_id'], 'checkpoint': checkpoint,
        'ground_truth_source': 'local image_pool filenames and caption_pool image ownership; test.csv selects queries',
        'query_counts': {'image': len(image_queries), 'caption': len(caption_queries)},
        'candidate_counts': {'image': len(image_ids), 'caption': len(caption_ids)},
        'pool_identity': suite.data_identity['splits']['test'],
        'pool_tables_sha256': {name: suite.data_identity['tables_sha256'][name]
                               for name in ('image_pool', 'caption_pool')},
        **scores,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    select = commands.add_parser('select', help='freeze the winner from validation records')
    select.add_argument('--run-dir', type=Path, required=True)
    select.add_argument('--experiment', help='freeze a strategy chosen by a separate validation comparison')
    select.add_argument('--selection', type=Path, required=True)
    evaluate = commands.add_parser('evaluate', help='evaluate a frozen selection on the local test split')
    evaluate.add_argument('--selection', type=Path, required=True)
    evaluate.add_argument('--data-dir', type=Path, default=Path('data'))
    evaluate.add_argument('--device', default=SuiteConfig.device)
    evaluate.add_argument('--eval-batch-size', type=int, default=SuiteConfig.eval_batch_size)
    evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'select':
        _write_new(args.selection, freeze_selection(args.run_dir, experiment=args.experiment))
    else:
        if args.output.exists():
            raise FileExistsError(f'Final report already exists: {args.output}')
        result = evaluate_frozen(_read_json(args.selection), args.data_dir, args.device, args.eval_batch_size)
        _write_new(args.output, result)


if __name__ == '__main__':
    main()
