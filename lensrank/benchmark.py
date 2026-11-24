"""Run the documented, paired LoRA and full fine-tuning benchmark."""

import argparse
import statistics
from pathlib import Path

import torch

from .checkpoints import atomic_json_save, run_lock
from .config import EXPERIMENTS, SuiteConfig
from .experiments import RetrievalSuite
from .final_test import freeze_selection
from .provenance import resolve_pretrained

SEEDS = (42, 43, 44)
STRATEGIES = {name: EXPERIMENTS[name] for name in ('openclip_full_finetune', 'openclip_lora')}


class BenchmarkSuite(RetrievalSuite):
    def evaluate(self, model, preprocess, tokenizer):
        torch.cuda.empty_cache()
        return super().evaluate(model, preprocess, tokenizer)


def run_benchmark(data_dir, run_dir):
    """Train and select on validation. Final test is a separate command."""
    rows = []
    expected_data = None
    expected_weights = None
    for seed in SEEDS:
        cfg = SuiteConfig(data_dir=data_dir, run_dir=run_dir / f'seed_{seed}', device='cuda',
                          epochs=5, seed=seed, batch_size=32, eval_batch_size=32)
        suite = BenchmarkSuite(cfg, verify_images=expected_data is None)
        weights, path = resolve_pretrained(cfg)
        if path is None or not weights['sha256']:
            raise ValueError('The benchmark requires cached, verifiable OpenCLIP weights')
        if len(suite.train) != 40_000:
            raise ValueError('Expected the documented 40,000 training pairs')
        if expected_data is None:
            expected_data, expected_weights = suite.data_identity, weights
        elif suite.data_identity != expected_data or weights != expected_weights:
            raise ValueError('Dataset or initial weights changed between seeds')
        rows.extend({'seed': seed, **row} for row in suite.run_all(STRATEGIES).to_dict('records'))

    scores = {
        name: statistics.mean(row['mean_score'] for row in rows if row['experiment'] == name)
        for name in STRATEGIES
    }
    winner = max(scores, key=scores.__getitem__)
    best = max((row for row in rows if row['experiment'] == winner), key=lambda row: row['mean_score'])
    selection = freeze_selection(run_dir / f'seed_{best["seed"]}', experiment=winner)
    atomic_json_save({
        'split': 'validation', 'mean_score_by_strategy': scores,
        'validation_winner': winner, 'runs': rows,
    }, run_dir / 'validation_report.json')
    atomic_json_save(selection, run_dir / 'final_selection.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('The documented benchmark requires CUDA')
    torch.set_num_threads(2)
    with run_lock(args.run_dir):
        run_benchmark(args.data_dir, args.run_dir)


if __name__ == '__main__':
    main()
