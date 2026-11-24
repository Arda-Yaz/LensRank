# LoRA vs. full fine-tuning

The paired benchmark measures multimodal retrieval quality and training cost under a shared five-epoch budget. Full scored slightly higher on validation across all three seeds. LoRA used 1.23M trainable parameters versus 149.62M for Full and completed training and validation about 40% faster on average.

## Protocol

| Setting | Value |
| --- | --- |
| Initialization | OpenCLIP ViT-B-16, OpenAI weights |
| Train | 8,000 images, 40,000 image–caption pairs per epoch |
| Validation | 1,000 images and 5,000 captions, both full pools |
| Schedule | 5 epochs per run, seeds 42/43/44, train and eval batch size 32 |
| Optimizer | AdamW, LR 1e-6 for Full and 1e-4 for LoRA |
| LoRA | MLP linear projections, rank 8, alpha 16, plus trainable logit scale |
| Hardware | NVIDIA GeForce RTX 5060 |
| Epoch selection | Highest validation mean Recall@1/5/10 across both directions, earliest epoch on ties |
| Final selection | Highest mean validation score across seeds selects the strategy. Its best validation checkpoint across seeds is frozen before test. |

## Results

| Strategy | Best validation score, mean ± sample SD | Trainable parameters | Mean elapsed time | Mean CUDA peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Full | 92.1656% ± 0.0512 pp | 149,620,737 | 38.28 min | 6.17 GiB |
| LoRA | 91.9644% ± 0.1805 pp | 1,228,801 | 23.14 min | 4.82 GiB |

The mean paired validation difference was **0.2011 percentage points** in favor of Full, with a sample SD of **0.1393 pp** over three seeds. The per-seed scores, six directional Recall metrics, elapsed times, and CUDA allocation peaks are in [validation_runs.csv](benchmarks/validation_runs.csv). Elapsed time includes training, validation, data loading, and in-loop persistence after setup. CUDA peaks are allocated tensor memory. Full had the lower allocation peak in seed 42, and LoRA had the lower peak in seeds 43 and 44.

### Held-out test

Full, seed 43, epoch 5 was frozen at **92.20% validation**. The final test used 500 image queries and 500 caption queries against 1,000 image and 5,000 caption candidates. Ground truth comes from the local image and caption ownership tables.

| Direction | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Image → text | 82.2% | 95.8% | 97.6% |
| Text → image | 73.2% | 93.2% | 97.6% |

The six metrics average to **89.93%**. The test measured the selected Full checkpoint, while the LoRA comparison used validation. Full reached its best validation score at the final epoch for two seeds, so the benchmark leaves longer training budgets open.

## Reproduce

Use the [Windows/CUDA environment](README.md#installation), arrange local dataset files as described in [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md), and cache the ViT-B-16/OpenAI initialization. The tracked runner trains the two strategies, writes the validation records, and freezes the selection. The separate final command verifies the dataset and checkpoint before scoring.

If the OpenCLIP cache is empty, prepare it once with this command. It downloads the pretrained weights.

```powershell
.\.venv\Scripts\python.exe -c "from lensrank.config import SuiteConfig; from lensrank.provenance import resolve_pretrained; resolve_pretrained(SuiteConfig(), download=True)"
```

```powershell
$run = '.local/benchmarks/reproduction'
.\.venv\Scripts\python.exe -m lensrank.benchmark --data-dir data --run-dir $run
.\.venv\Scripts\python.exe -m lensrank.final_test evaluate --selection "$run/final_selection.json" --data-dir data --device cuda --eval-batch-size 32 --output "$run/final_test.json"
```

The reported measurements were collected at code revision `86caef0f331550945b3e01c505c8ca7782d17b62`. The tracked runner implements the same benchmark protocol for new runs. Use a fresh run directory when the training source changes.

| Recorded object | SHA-256 |
| --- | --- |
| OpenCLIP initialization | `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f` |
| Frozen Full checkpoint | `cdca41fffc057e1020a52fec649a89f1b3bf38542f994e98ee2120953182c0c3` |
| Original local benchmark driver | `1e22011d80c9644a243b8e1158944c869dc216e9309827fab3e8068819a9d98d` |
