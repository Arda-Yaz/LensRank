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

## Verification

Model loading, bidirectional retrieval, cache reuse, and Gradio HTTP/API behavior have been verified on an RTX 5060.
The dataset contains 8,000 training images, 1,000 validation images, and 1,000 test images.
Windows DataLoader batches have been checked with four worker processes for consistent ordering and tensor contents.
Asset checksums, retrieval metrics, and environment details are listed in `SOURCE_MANIFEST.md`.
