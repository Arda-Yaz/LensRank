# Asset manifest

## Model

- Configuration uses OpenCLIP ViT-B-16/OpenAI.
- LoRA targets `c_fc`, `c_proj`, `fc1`, and `fc2` with rank 8 and alpha 16.
- The checkpoint is stored at `artifacts/fresh_runs/openclip_lora/best.pt`.
- The state dictionary contains 398 keys and passes strict loading.
- Validation embeddings have shapes `1000 × 512` for images and `5000 × 512` for captions.

## Dataset

| Split | CSV rows | Images |
| --- | --- | --- |
| Training | 40,000 | 8,000 |
| Validation | 5,000 | 1,000 |
| Test | 1,000 | 1,000 |

The training and validation tables contain `image_id` and `caption_id` mappings.
The test table uses `id`, `query`, and `query_type` fields for 500 image queries and 500 caption queries.
All 10,000 images pass image-file verification, and query IDs resolve to their corresponding pool entries.
`data/manifest.json` contains per-file SHA-256 checksums for dataset files and experiment records.
Images and model weights are stored locally.

## Environment

The application is verified on Windows with the following environment.

- Python 3.12.14
- NVIDIA GeForce RTX 5060 with 8,151 MiB of VRAM
- PyTorch 2.7.1+cu128
- Torchvision 0.22.1+cu128
- OpenCLIP 2.32.0
- Gradio 5.49.1
- Uvicorn 0.35.0

`requirements-lock.txt` contains the complete package versions.
Uvicorn is pinned to 0.35.0 to support clean Gradio server shutdown on Windows.

## Verification

- Checkpoint size and SHA-256 match the expected values.
- Model loading and GPU inference complete successfully.
- Both retrieval directions return the requested 1, 5, and 10 results.
- Empty text and missing-image inputs produce the expected responses.
- Embeddings contain finite values and have unit norms.
- Cache reuse preserves the cache file's modification timestamp and embedding contents.
- Real-data batches loaded with four workers match single-process batches exactly.
- The local HTTP endpoint returns 200, and both Gradio search endpoints return five results.
- Server startup and shutdown complete successfully.
- `pip check` reports no dependency conflicts.
- Python modules and notebook code cells pass syntax checks.

OpenCLIP emits a QuickGELU activation warning for the ViT-B-16/OpenAI combination.
The model name and activation selection follow the application configuration.
All six Recall metrics computed from the validation cache match the comparison table.

| Direction | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| Image → Text | 0.886 | 0.988 | 0.999 |
| Text → Image | 0.7592 | 0.9452 | 0.976 |

The Recall check uses the cached validation embeddings.

## SHA-256 checksums

| File | Bytes | SHA-256 |
| --- | --- | --- |
| artifacts/fresh_runs/openclip_lora/best.pt | 603549475 | b9c2631b47db0830f3e745186c8f983ea9bb6fb011bec446fa692ee1994005c7 |
| artifacts/fresh_runs/fresh_comparison.csv | 923 | 0be9bf1619c201e383174c98ca8ce12362a6c790c3fe127758c41073c6f13335 |
| artifacts/fresh_runs/demo_validation_index.pt | 12308117 | b412d9471ea66d2ab18244ef6a87cf4349df6d685e08b730b418aa6e33518473 |
| data/caption_pool.csv | 3754148 | 8e95d5e81ac1dadffcc4b1d2cdcfb3d26accea1f90a954da1889ee68fc5729a9 |
| data/image_pool.csv | 206121 | dbda7b0e652159ba7af8479355786166cda91031cac8803968314cb9f5828f20 |
| data/sample_submission.csv | 24948 | b2837dcb4b7041028d3e024dcd273bf96d6ea258e0e924e1893feac6cda03126 |
| data/test.csv | 17415 | dd58eb3fd8957281d93e161da7ad3e9b3c25ed7f97c99b44c9c73bdebaa78143 |
| data/train.csv | 463380 | 7a8a29a40d010b9918d239d08556283b143803812499657fa7363dd92d3809b4 |
| data/val.csv | 60021 | b618080737f2fb12aa536df38a05c8004cede78f51822bb8c630361808d907c7 |
| data/manifest.json | 1761854 | 6bf0a4ef46e0eec30c3f9c31f218cf52716abd81d7435c61f08072528b94fb73 |
| requirements-lock.txt | 1307 | f422252bb18e44e2274cd721198c181dba4e4815cbab15623562c8d3fc563b05 |
