import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from lensrank import experiments
from lensrank.data import ImageBatchSampler, PairDataset
from lensrank.final_test import freeze_selection, main
from lensrank.provenance import resolve_pretrained
from lensrank.retrieval import retrieval_recall
from resumable_experiments import RetrievalSuite, SuiteConfig


def make_data(root):
    def table(name, columns, rows):
        pd.DataFrame(rows, columns=columns).to_csv(root / f'{name}.csv', index=False)

    images = {split: ids for split, ids in [('train', range(1, 5)), ('val', [5, 6]), ('test', [7, 8])]}
    for split, ids in images.items():
        directory = root / f'{split}_images' / f'{split}_images'
        directory.mkdir(parents=True)
        for image in ids:
            Image.new('RGB', (2, 2), (image * 24, image * 5, 50)).save(directory / f'{image}.png')
    pool = list(range(1, 9))
    table('image_pool', ['image_id', 'image_name'], [(i, f'{i}.png') for i in pool])
    table('caption_pool', ['caption_id', 'image_id', 'caption'],
          [(10 * i + j, i, str(10 * i + j)) for i in pool for j in (0, 1)])
    for split in ('train', 'val'):
        table(split, ['image_id', 'caption_id'], [(i, 10 * i + j) for i in images[split] for j in (0, 1)])
    table('test', ['id', 'query', 'query_type'], [(1, 7, 'image'), (2, 81, 'caption')])


def pixels(image):
    return torch.tensor(image.getpixel((0, 0)), dtype=torch.float32) / 255


def tokenize(captions):
    return torch.tensor([[int(c) / 100, int(c) % 2, 1] for c in captions], dtype=torch.float32)


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 4)
        self.text_projection = nn.Linear(3, 4)
        self.logit_scale = nn.Parameter(torch.zeros(()))

    def encode_image(self, images):
        hidden = self.visual(images)
        if self.training:
            hidden = nn.functional.dropout(hidden, p=0.2)
            hidden = hidden + (random.random() + np.random.random()) * 0.01
        return hidden

    def encode_text(self, tokens):
        return self.text_projection(tokens)


def tiny_transforms(*args, **kwargs):
    return TinyCLIP(), None, pixels


class RetrievalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'data'
        make_data(self.data)
        for name, mock in [('create_model_and_transforms', tiny_transforms), ('get_tokenizer', lambda _: tokenize)]:
            patcher = patch.object(experiments.open_clip, name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)

    def suite(self, name, **options):
        return RetrievalSuite(SuiteConfig(
            data_dir=self.data, run_dir=self.root / name, device='cpu', pretrained='',
            batch_size=3, epochs=2, checkpoint_every_batches=1, num_workers=0, **options,
        ))

    def test_multicaption_recall_with_query_subsets_and_small_pools(self):
        images = torch.tensor([[3., 0.], [0., 3.], [-3., 0.]])
        captions = torch.tensor([[1., 3.], [3., 0.], [0., 3.], [-3., 0.]])
        ids = ([1, 2, 3], [10, 11, 20, 30])
        owners = {10: 1, 11: 1, 20: 2, 30: 3}
        scores = retrieval_recall(images, captions, *ids, owners, [1], [10])
        self.assertEqual(scores['i2t_R@1'], 1)
        self.assertEqual(scores['t2i_R@1'], 0)
        for direction in ('i2t', 't2i'):
            for k in (5, 10):
                self.assertEqual(scores[f'{direction}_R@{k}'], 1)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            retrieval_recall(images, captions, *ids, owners, [], [10])
        with self.assertRaisesRegex(ValueError, 'align'):
            retrieval_recall(images, captions, *ids, {**owners, 10: 4})

    def test_split_rejects_same_image_through_different_caption(self):
        pd.DataFrame([(5, 50), (6, 60), (1, 11)], columns=['image_id', 'caption_id']).to_csv(
            self.data / 'val.csv', index=False,
        )
        train = pd.read_csv(self.data / 'train.csv')
        train.loc[train.caption_id.ne(11)].to_csv(self.data / 'train.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'train/val split overlap in image ID: 1'):
            self.suite('overlap')

    def test_test_query_must_resolve_inside_test_split(self):
        pd.DataFrame([(1, 5, 'image'), (2, 81, 'caption')], columns=['id', 'query', 'query_type']).to_csv(
            self.data / 'test.csv', index=False,
        )
        with self.assertRaisesRegex(ValueError, 'inside the test candidate pool'):
            self.suite('invalid_query')

    def test_url_weights_keep_the_same_identity_after_cache_creation(self):
        cfg = SuiteConfig(model_name='mock', pretrained='mock_tag', device='cpu')
        cached = self.root / '.cache' / 'clip' / 'weights.pt'
        metadata = {'url': 'https://example.test/weights.pt'}

        def create_cache(_):
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b'mock OpenCLIP weights')
            return str(cached)

        with (patch.object(Path, 'home', return_value=self.root),
              patch.object(experiments.open_clip, 'get_pretrained_cfg', return_value=metadata),
              patch.object(experiments.open_clip, 'download_pretrained', side_effect=create_cache) as download):
            initial = resolve_pretrained(cfg, download=True)
            self.assertEqual(initial, resolve_pretrained(cfg))
            download.assert_called_once()

    def test_sampler_coverage_repeatability_and_resume(self):
        ids = [4] * 7 + [8] * 2 + [9] * 3 + [10] + [11] * 2
        batches = list(ImageBatchSampler(ids, 3, seed=42, epoch=1))
        self.assertEqual(batches, list(ImageBatchSampler(ids, 3, 42, 1)))
        self.assertEqual(batches[2:], list(ImageBatchSampler(ids, 3, 42, 1, 2)))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(ids))))
        self.assertTrue(all(1 <= len(batch) <= 3 for batch in batches))
        self.assertTrue(all(len(batch) == len({ids[i] for i in batch}) for batch in batches))
        self.assertEqual(list(ImageBatchSampler([1], 128, 42, 1)), [[0]])

    def test_worker_collate_and_batch_cursor(self):
        suite = self.suite('worker')
        dataset = PairDataset(suite.train, suite.train_dir, suite.image_names, suite.captions, pixels)
        expected = list(suite._loader(dataset, tokenize, epoch=1))[1:]
        suite.cfg.num_workers = 1
        resumed = list(suite._loader(dataset, tokenize, epoch=1, start_batch=1))
        torch.testing.assert_close(expected, resumed, rtol=0, atol=0)

    def test_resume_within_and_after_epoch_including_gradual_boundary(self):
        spec = {'strategy': 'gradual', 'lr': 0.005}
        baseline = self.suite('uninterrupted')
        result = baseline.run_one('gradual', spec)
        expected = torch.load(baseline.runs / 'gradual' / 'last.pt', weights_only=False)
        self.assertEqual([item['first_epoch'] for item in result['optimizer_stages']], [1, 2])
        self.assertIsNone(result['resources']['cuda_peak_allocated_bytes'])
        save = experiments.atomic_torch_save
        for epoch, batch in [(1, 1), (1, 3), (2, 0), (2, 1), (3, 0)]:
            with self.subTest(epoch=epoch, batch=batch):
                suite = self.suite(f'interrupted_{epoch}_{batch}')

                def interrupt_after_save(value, path, cursor=(epoch, batch)):
                    save(value, path)
                    if path.name == 'last.pt' and (value['epoch'], value['next_batch']) == cursor:
                        raise KeyboardInterrupt

                with (patch.object(experiments, 'atomic_torch_save', interrupt_after_save),
                      self.assertRaises(KeyboardInterrupt)):
                    suite.run_one('gradual', spec)
                random.random()
                np.random.random()
                torch.rand(5)
                resumed = suite.run_one('gradual', spec)
                actual = torch.load(suite.runs / 'gradual' / 'last.pt', weights_only=False)
                for key in ('model', 'optimizer'):
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                self.assertEqual(actual['history'], expected['history'])
                self.assertEqual(resumed['optimizer_stages'], result['optimizer_stages'])
                self.assertEqual(resumed['best_epoch'], result['best_epoch'])
                best = torch.load(suite.runs / 'gradual' / 'best.pt', weights_only=False)
                self.assertEqual(best['epoch'], resumed['best_epoch'])
                self.assertEqual(best['metrics']['mean_score'], resumed['mean_score'])

    def test_seed_independent_of_suite_order_and_completed_skips(self):
        spec = {'strategy': 'full', 'lr': 0.005}
        single = self.suite('single')
        single.run_one('full', spec)
        for skip in (False, True):
            suite = self.suite(f'suite_{skip}')
            if skip:
                suite.run_one('prior', spec)
            suite.run_all({'prior': spec, 'full': spec})
            expected = torch.load(single.runs / 'full' / 'last.pt', weights_only=False)
            actual = torch.load(suite.runs / 'full' / 'last.pt', weights_only=False)
            torch.testing.assert_close(actual['model'], expected['model'], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'identity mismatch.*separate run_dir'):
            single.run_one('full', {**spec, 'lr': 0.02})
        legacy = self.suite('legacy')
        folder = legacy.runs / 'full'
        folder.mkdir(parents=True)
        (folder / 'completed.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Legacy.*separate run_dir'):
            legacy.run_one('full', spec)
        self.assertEqual(list(folder.iterdir()), [folder / 'completed.json'])
        recovered = self.suite('marker_recovery')
        folder = recovered.runs / 'full'
        folder.mkdir(parents=True)
        (folder / 'run.json.tmp').write_text('{')
        self.assertEqual(recovered.run_one('full', spec)['status'], 'completed')

    def test_final_commands_use_frozen_checkpoint_and_full_candidate_pools(self):
        for strategy in ('full', 'zero_shot'):
            with self.subTest(strategy=strategy):
                suite = self.suite(strategy)
                if strategy == 'zero_shot':
                    weights = self.root / 'mock_pretrained.pt'
                    weights.write_bytes(b'synthetic pretrained fixture')
                    suite.cfg.pretrained = str(weights)
                suite.run_all({strategy: {'strategy': strategy, 'lr': 0.005}})
                selection = self.root / f'{strategy}_selection.json'
                report = self.root / f'{strategy}_report.json'
                main(['select', '--run-dir', str(suite.runs), '--selection', str(selection)])
                comparison = suite.runs / 'fresh_comparison.csv'
                before = comparison.read_bytes()
                cmd = ['evaluate', '--selection', str(selection), '--data-dir', str(self.data),
                       '--device', 'cpu', '--output', str(report)]
                main(cmd)
                result = json.loads(report.read_text())
                self.assertEqual(result['split'], 'test')
                self.assertEqual(result['query_counts'], {'image': 1, 'caption': 1})
                self.assertEqual(result['candidate_counts'], {'image': 2, 'caption': 4})
                self.assertEqual(result['checkpoint']['kind'], 'pretrained' if strategy == 'zero_shot' else 'trained')
                self.assertEqual(result['i2t_R@10'], 1)
                self.assertEqual(result['t2i_R@10'], 1)
                self.assertEqual(comparison.read_bytes(), before)
                if strategy == 'full':
                    checkpoint = suite.runs / strategy / 'best.pt'
                    with checkpoint.open('ab') as stream:
                        stream.write(b'changed')
                    report.unlink()
                    with self.assertRaisesRegex(ValueError, 'changed after validation selection'):
                        main(cmd)
                    self.assertFalse(report.exists())
                    checkpoint.unlink()
                    with self.assertRaisesRegex(FileNotFoundError, 'requires its checkpoint'):
                        freeze_selection(suite.runs)


if __name__ == '__main__':
    unittest.main()
