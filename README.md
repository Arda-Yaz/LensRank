# LensRank

LensRank is a multimodal retrieval application with a Gradio interface for text-to-image and image-to-caption search.
It uses OpenCLIP ViT-B-16/OpenAI with LoRA adapters and searches the validation collection.
The active experiment is selected by sorting `artifacts/fresh_runs/fresh_comparison.csv` by `mean_score`.

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

`run.json` records the resolved training identity. Image identity uses the manifest checksums and checks file sizes. Refresh the manifest after replacing images. `completed.json` records each AdamW parameter stage and the selected checkpoint. Elapsed seconds cover training and validation after setup, including data loading and in-loop persistence. Resume sums elapsed seconds from checkpointed segments and takes the maximum CUDA peak across them. CUDA peak is allocated tensor memory in bytes from `torch.cuda.max_memory_allocated` and is inapplicable on CPU.

After validation, freeze the winning experiment, epoch and checkpoint, then evaluate that selection separately.

```powershell
.\.venv\Scripts\python.exe -m lensrank.final_test select --run-dir artifacts/retrieval_runs_v1 --selection artifacts/retrieval_runs_v1/final_selection.json
.\.venv\Scripts\python.exe -m lensrank.final_test evaluate --selection artifacts/retrieval_runs_v1/final_selection.json --output artifacts/retrieval_runs_v1/final_test.json
```

The final report uses the queries from `test.csv` and the full candidate pools derived from test image filenames and caption ownership. It reports local ground-truth Recall in both directions. For historical validation records, use `artifacts/fresh_runs` as the selection source and provide a new path for both JSON outputs. Historical best checkpoints identify their selection through the recorded validation score and the checkpoint hash. New checkpoints also store the selected epoch and run ID. Keep final test scores out of model and hyperparameter selection.

## Verification

Run the CPU regression tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Model loading, bidirectional retrieval, cache reuse, and Gradio HTTP/API behavior have been verified on an RTX 5060.
The dataset contains 8,000 training images, 1,000 validation images, and 1,000 test images.
Windows DataLoader batches have been checked with four worker processes for consistent ordering and tensor contents.
Asset checksums, retrieval metrics, and environment details are listed in `SOURCE_MANIFEST.md`.
