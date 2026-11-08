# Modül yapısı

| Python girişi | Modül |
| --- | --- |
| SuiteConfig, EXPERIMENTS | lensrank/config.py |
| LoRALinear, inject_lora | lensrank/adapters.py |
| PairDataset | lensrank/data.py |
| atomic_torch_save, atomic_json_save, _remove_empty_directory_artifact | lensrank/checkpoints.py |
| seed_all, RetrievalSuite | lensrank/experiments.py |
| load_winner, build_or_load_index, search_images, search_captions | app.py |

resumable_experiments.py notebook ve Python çağrıları için suite girişlerini dışa aktarır.

LoRA hedefleri c_fc, c_proj, fc1 ve fc2 katmanlarıdır.
Rank 8 ve alpha 16 kullanılır.
State dictionary base, a ve b parametrelerini içerir.

Validation görsel ve caption ID'leri sıralanarak encode edilir.
Embedding'ler normalize edilir ve dot product ile sıralanır.
Cache checkpoint_stamp, image_ids, caption_ids, image_embeddings ve text_embeddings alanlarını içerir.

Deney stratejileri zero-shot, projection tuning, full fine-tuning, gradual unfreezing ve LoRA'dır.

Windows worker süreçleri için collate_pairs modül düzeyinde tanımlanır.
DataLoader tokenizer'ı functools.partial ile bu fonksiyona bağlar.
Epoch seed'i, shuffle, batch boyutu ve tokenization sırası aynı kalır.
