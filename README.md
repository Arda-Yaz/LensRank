# LensRank

LensRank is an OpenCLIP retrieval benchmark and Gradio demo for text-to-image and image-to-caption search. It compares zero-shot inference, projection tuning, gradual unfreezing, full fine-tuning, and LoRA on a shared ViT-B-16 backbone.

- **Training** uses deterministic batches with unique image IDs to avoid treating captions of the same image as negatives.
- **Evaluation** measures Recall@1/5/10 in both directions and evaluates a frozen validation checkpoint on held-out queries.
- **Reproducibility** includes checkpointed training state, data and weight fingerprints, and offline regression tests for resume and retrieval.

## Benchmark results

On an RTX 5060, Full fine-tuning reached **92.17% mean validation Recall** across three seeds. LoRA reached **91.96%** and completed training and validation in **23.14 minutes on average**, compared with **38.28 minutes** for Full. The selected Full checkpoint reached **89.93%** on the held-out test split. The test score applies to that checkpoint. [Protocol, per-seed results, and reproduction commands](BENCHMARK.md)

The code and benchmark summaries are included in the repository. The dataset, model checkpoints, and embedding cache are local assets. The offline tests use synthetic fixtures and run without those assets.

## Installation

Create a virtual environment with Python 3.12 from the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` contains the validated package versions for Windows, Python 3.12, and CUDA 12.8.
`requirements.txt` lists the application's dependency requirements.

## Usage

```powershell
.\.venv\Scripts\python.exe app.py
```

Enter a natural-language description in **Text → Images**, or upload an image in **Image → Captions**.
Both search modes support 1–10 results.

The demo uses the historical LoRA checkpoint and searches the validation collection. For the local asset layout and checksums, see [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

The application builds validation embeddings in `artifacts/fresh_runs/demo_validation_index.pt` on first launch.
The cache is reused while the checkpoint modification timestamp matches.
The demo uses `eval_batch_size=256` and `num_workers=0`.
Importing `app` also initializes the model and cache.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio interface and search callbacks |
| `lensrank/` | Model configuration, LoRA adapters, data loading, encoding, and evaluation |
| `resumable_experiments.py` | Public Python imports for the experiment notebook |
| `train_all_models_resumable.ipynb` | Experiment configuration, training, and evaluation |
| `data/` | Dataset tables and training, validation, and test images |
| `artifacts/fresh_runs/` | Checkpoints, experiment records, comparison results, and embedding cache |
| `SOURCE_MANIFEST.md` | Asset inventory, checksums, and verification results |
| `PORT_MAP.md` | Module responsibilities and Python entry points |

The training notebook can start long-running GPU jobs when its training cells are executed.
The interface runs independently of the notebook.

## Training and final test

Run the training notebook with its new `artifacts/retrieval_runs_v1` output directory. To continue a run, rerun the same configuration. Use a separate `run_dir` for legacy records, changed settings, or a `fast_dev_run`. The comparison CSV ranks validation scores. The demo continues to read the historical validation comparison and cache under `artifacts/fresh_runs`.

Each epoch visits every selected training pair in deterministic batches with unique image IDs. A singleton batch advances the checkpoint cursor without an AdamW update because it has no contrastive negatives. Epoch loss averages the batches with at least two images. Its value is empty when the epoch has no such batch. Resume relies on the deterministic OpenCLIP validation transform used for pair loading and deterministic tokenization in the workers. The CPU regression covers training with identical software, data and configuration. Gradual unfreezing starts a new AdamW when all parameters become trainable and carries forward the AMP scaler.

`run.json` records the original run ID and its resolved training inputs. Resume accepts a moved run directory or a documentation-only commit when the execution fingerprint matches. For assets outside the manifest, keep file sizes and modification times intact when moving the dataset. Resume also checks that the checkpoint and run metadata belong to the same original run.

Routine training startup checks the sizes of manifest-listed images. The paired benchmark verifies their SHA-256 hashes before training, and final evaluation verifies them before inference. Images outside the manifest use a size and modification-time fingerprint. Refresh the manifest after intentional dataset changes.

`completed.json` records each AdamW parameter stage and the selected checkpoint. Elapsed time covers training and validation after setup, including data loading and in-loop persistence. Resume sums elapsed seconds from checkpointed segments and takes the maximum CUDA peak across them. CUDA peak comes from `torch.cuda.max_memory_allocated` and is inapplicable on CPU.

After validation, freeze the winning experiment, epoch and checkpoint, then evaluate that selection separately.

```powershell
.\.venv\Scripts\python.exe -m lensrank.final_test select --run-dir artifacts/retrieval_runs_v1 --selection artifacts/retrieval_runs_v1/final_selection.json
.\.venv\Scripts\python.exe -m lensrank.final_test evaluate --selection artifacts/retrieval_runs_v1/final_selection.json --output artifacts/retrieval_runs_v1/final_test.json
```

The final report uses the queries from `test.csv` and the full candidate pools derived from test image filenames and caption ownership. It reports local ground-truth Recall in both directions. For historical validation records, use `artifacts/fresh_runs` as the selection source and provide a new path for both JSON outputs. Historical best checkpoints identify their selection through the recorded validation score and the checkpoint hash. New checkpoints also store the selected epoch and run ID. Keep final test scores out of model and hyperparameter selection.

If you move a run directory after freezing its selection, run `select` again with the new run directory and a fresh JSON output path.

## Verification

Run the CPU regression tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Model loading, bidirectional retrieval, cache reuse, and Gradio HTTP/API behavior have been verified on an RTX 5060.
The dataset contains 8,000 training images, 1,000 validation images, and 1,000 test images.
Windows DataLoader batches have been checked with four worker processes for consistent ordering and tensor contents.
Asset checksums, retrieval metrics, and environment details are listed in `SOURCE_MANIFEST.md`.
