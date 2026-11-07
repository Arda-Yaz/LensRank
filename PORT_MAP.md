# Modül eşlemeleri

| Python girişi | Modül |
| --- | --- |
| resumable_experiments.SuiteConfig, EXPERIMENTS | lensrank/config.py |
| resumable_experiments.LoRALinear, inject_lora | lensrank/adapters.py |
| resumable_experiments.PairDataset | lensrank/data.py |
| resumable_experiments.atomic_torch_save, atomic_json_save, _remove_empty_directory_artifact | lensrank/checkpoints.py |
| resumable_experiments.seed_all, RetrievalSuite | lensrank/experiments.py |
| app.py | app.py |
| train_all_models_resumable.ipynb | Aynı adlı notebook |

resumable_experiments.py, notebook ve Python çağrıları için girişleri yeniden dışa aktarır.
Deney altyapısı beş model adaptasyon stratejisini çalıştırır. Gradio uygulaması validation koleksiyonunda arama yapar.
