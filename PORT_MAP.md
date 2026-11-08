# Module layout

| Python entry point | Module |
| --- | --- |
| `SuiteConfig`, `EXPERIMENTS` | `lensrank/config.py` |
| `LoRALinear`, `inject_lora` | `lensrank/adapters.py` |
| `PairDataset`, `collate_pairs` | `lensrank/data.py` |
| `atomic_torch_save`, `atomic_json_save`, `_remove_empty_directory_artifact` | `lensrank/checkpoints.py` |
| `seed_all`, `RetrievalSuite` | `lensrank/experiments.py` |
| `load_winner`, `build_or_load_index`, `search_images`, `search_captions` | `app.py` |

`resumable_experiments.py` exports the suite entry points for notebook and Python callers.

## Model and retrieval

LoRA targets the `c_fc`, `c_proj`, `fc1`, and `fc2` layers with rank 8 and alpha 16.
The adapter state dictionary contains the `base`, `a`, and `b` parameters.

Validation image and caption IDs are sorted before encoding.
Embeddings are normalized and ranked by their dot product.
The cache contains `checkpoint_stamp`, `image_ids`, `caption_ids`, `image_embeddings`, and `text_embeddings`.

The experiment suite supports zero-shot, projection tuning, full fine-tuning, gradual unfreezing, and LoRA.

## Windows data loading

`collate_pairs` is defined at module scope so it can be serialized for worker processes.
The DataLoader binds the tokenizer with `functools.partial`.
Epoch seeding, shuffling, batch size, and tokenization order are consistent across worker configurations.
