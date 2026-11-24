"""Split validation and image-aware training batches."""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_table(directory, name, ids, strings=(), empty_caption=False):
    try:
        table = pd.read_csv(directory / f'{name}.csv', dtype='string', keep_default_na=empty_caption)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f'{name}.csv is empty') from exc
    missing = {*ids, *strings} - set(table.columns)
    if missing or table.empty:
        raise ValueError(f'{name}.csv needs rows and required columns; missing: {sorted(missing)}')
    for column in ids:
        if not table[column].str.fullmatch(r'[0-9]+', na=False).all():
            raise ValueError(f'{name}.csv has an invalid integer ID in {column}')
        try:
            table[column] = table[column].astype('int64')
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'{name}.csv has an out-of-range ID in {column}') from exc
    for column in strings:
        if column == 'caption' and empty_caption:
            # Match the historical caption pool's read_csv().astype(str) behavior.
            table[column] = table[column].fillna('nan')
            continue
        if not table[column].str.strip().ne('').all():
            raise ValueError(f'{name}.csv has a blank {column}')
    if table[ids[0]].duplicated().any():
        raise ValueError(f'{name}.csv has duplicate {ids[0]} values')
    return table


def validate_splits(train, val, test, images, captions, directories):
    names = images.set_index('image_id').image_name.map(lambda name: Path(name).name)
    if names.eq('').any() or names.str.casefold().duplicated().any():
        raise ValueError('image_pool.csv has an empty or ambiguous image filename')
    owners = captions.set_index('caption_id').image_id
    if not owners.isin(names.index).all():
        raise ValueError('caption_pool.csv has an owner missing from image_pool.csv')
    splits = {}
    for split, pairs in [('train', train), ('val', val)]:
        if not pairs.image_id.isin(names.index).all() or not pairs.caption_id.isin(owners.index).all():
            raise ValueError(f'{split}.csv contains IDs missing from the image or caption pool')
        if not pairs.caption_id.map(owners).eq(pairs.image_id).all():
            raise ValueError(f'{split}.csv pairs a caption with the wrong image owner')
        splits[split] = (set(pairs.image_id), set(pairs.caption_id))

    files = {}
    for split, directory in directories.items():
        entries = [path.name for path in directory.iterdir() if path.is_file()]
        files[split] = {name.casefold(): name for name in entries}
        if len(files[split]) != len(entries) or not entries:
            raise ValueError(f'{directory} has no files or has ambiguous image filenames')

    by_name = {name: image for image, name in names.items()}
    unknown = set(files['test'].values()) - by_name.keys()
    if unknown:
        raise ValueError(f'Test image file missing from image_pool.csv: {min(unknown)}')
    test_images = {by_name[name] for name in files['test'].values()}
    test_captions = set(owners.index[owners.isin(test_images)])
    if test_images - set(owners.loc[sorted(test_captions)]):
        raise ValueError('Test image has no captions in caption_pool.csv')
    splits['test'] = (test_images, test_captions)

    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for kind, left, right in zip(('image ID', 'caption ID'), splits[first], splits[second]):
            if overlap := left & right:
                raise ValueError(f'{first}/{second} split overlap in {kind}: {min(overlap)}')
        if overlap := files[first].keys() & files[second].keys():
            raise ValueError(f'{first}/{second} split reuses image filename: {min(overlap)}')

    for split, (image_ids, _) in splits.items():
        missing = set(names.loc[sorted(image_ids)]) - set(files[split].values())
        if missing:
            raise FileNotFoundError(f'{split} image file missing from {directories[split]}: {min(missing)}')
    if not test.query_type.isin(['image', 'caption']).all():
        raise ValueError('test.csv query_type must be image or caption')
    for kind, candidates in zip(('image', 'caption'), splits['test']):
        queries = test.loc[test.query_type.eq(kind), 'query']
        if queries.empty or not queries.isin(candidates).all():
            raise ValueError(f'test.csv needs {kind} queries that resolve inside the test candidate pool')
        if queries.duplicated().any():
            raise ValueError(f'test.csv has duplicate {kind} query IDs')
    return splits, owners.to_dict()


class ImageBatchSampler:
    """Distribute each image's captions across distinct, balanced batches."""

    def __init__(self, image_ids, batch_size, seed, epoch, start_batch=0):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        groups = defaultdict(list)
        for index, image in enumerate(image_ids):
            groups[image].append(index)
        if not groups:
            raise ValueError('Training pairs are empty')
        count = max((sum(map(len, groups.values())) + batch_size - 1) // batch_size,
                    max(map(len, groups.values())))
        rng = random.Random(seed + epoch)
        self.batches = [[] for _ in range(count)]
        order = list(groups.values())
        rng.shuffle(order)
        slot = 0
        for indices in order:
            rng.shuffle(indices)
            for index in indices:
                self.batches[slot % count].append(index)
                slot += 1
        for batch in self.batches:
            rng.shuffle(batch)
        rng.shuffle(self.batches)
        if not 0 <= start_batch <= count:
            raise ValueError(f'Invalid resume batch {start_batch} for {count} batches')
        self.start_batch = start_batch

    def __iter__(self):
        return iter(self.batches[self.start_batch:])

    def __len__(self):
        return len(self.batches) - self.start_batch


def collate_pairs(items, tokenizer):
    """Stack image tensors and tokenize captions for a batch."""
    images, texts = zip(*items)
    return torch.stack(images), tokenizer(list(texts))


class PairDataset(Dataset):

    def __init__(self, pairs, image_dir, image_names, captions, preprocess):
        self.pairs = pairs.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_names = image_names
        self.captions = captions
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        iid, cid = (int(row.image_id), int(row.caption_id))
        path = self.image_dir / Path(self.image_names[iid]).name
        with Image.open(path) as image:
            return self.preprocess(image.convert('RGB')), self.captions[cid]
