# Dosya envanteri

## Model

- Yapılandırma OpenCLIP ViT-B-16/OpenAI
- LoRA hedefleri c_fc, c_proj, fc1, fc2
- LoRA rank 8, alpha 16
- Checkpoint artifacts/fresh_runs/openclip_lora/best.pt
- Boyut 603549475 bayt
- SHA-256 b9c2631b47db0830f3e745186c8f983ea9bb6fb011bec446fa692ee1994005c7

## Veri ve çıktılar

- data/train.csv, data/val.csv, data/test.csv
- data/image_pool.csv, data/caption_pool.csv
- data/train_images/train_images/
- data/val_images/val_images/
- data/test_images/test_images/
- artifacts/fresh_runs/fresh_comparison.csv
- artifacts/fresh_runs/demo_validation_index.pt

## Ortam

Python 3.12 ve CUDA 12.8 PyTorch ortamı kuruluyor.
Veri eşlemeleri ve gerçek model yüklemesi kontrol ediliyor.
Doğrulama tamamlandığında ölçülen sonuçlar bu dosyaya kaydedilir.
