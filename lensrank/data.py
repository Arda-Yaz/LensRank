"""Image and caption pair dataset."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


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
        return (self.preprocess(Image.open(path).convert('RGB')), self.captions[cid])
