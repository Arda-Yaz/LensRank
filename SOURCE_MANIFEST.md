# LensRank varlık kaydı

Yerel varlıklar, checkpoint doğrulaması ve ortam durumu.

## Yerel checkpoint

- Konum `artifacts/fresh_runs/openclip_lora/best.pt`
- Boyut 603549475 bayt
- SHA-256 `b9c2631b47db0830f3e745186c8f983ea9bb6fb011bec446fa692ee1994005c7`
- Kullanıcının sağladığı dosya üzerinde boyut ve SHA-256 eşleşmesi doğrulandı.
- Tensor içeriği ve model yükleme uyumluluğu henüz kontrol edilmedi.
- Ağırlık dosyası Git geçmişine alınmaz.

## Eksik yerel varlıklar

- data/train.csv, data/val.csv, data/test.csv
- data/image_pool.csv, data/caption_pool.csv
- data/train_images/train_images/, data/val_images/val_images/, data/test_images/test_images/
- artifacts/fresh_runs/*/history.csv ve completed.json

Demo cache dosyası sabit commit'in dosya ağacında bulunamadı.
Uygulama gerektiğinde artifacts/fresh_runs/demo_validation_index.pt konumuna aynı model ve veriden üretir.

## Kontrol ortamı

Kontroller Codex ile gelen Python 3.12.14 kullanılarak yapıldı.
Torch, open_clip ve gradio bu ortamda bulunmuyor.
Tensor yükleme ve uygulamanın çalışması henüz doğrulanmadı.
Checkpoint ZIP dizininde 404 kayıt ve data.pkl bulundu.

## Yerel dosya SHA-256 kayıtları

Aşağıdaki değerler kaydedilen dosyaların baytlarından hesaplandı.

| Dosya | SHA-256 |
| --- | --- |
| requirements.txt | d3d460db706ac69f8105718daa9369e698c868a9f98a696fa1d8b1bbb7426dbf |
| app.py | 546c3040e66474f7770ea57251640dc238ced6ff0cc7aa07fda069c716b3bfae |
| resumable_experiments.py | be7e2a5067f8c76ffa4322f1f1201368ef1b3e65ecf830b51820cbade3f223bf |
| train_all_models_resumable.ipynb | db96c0b54c088a39518dd581aee052e1c089285cc5e5b4d7148442976f64d2a1 |
| artifacts/fresh_runs/fresh_comparison.csv | 0be9bf1619c201e383174c98ca8ce12362a6c790c3fe127758c41073c6f13335 |
| lensrank/__init__.py | da64b23cee438084f6d2c48423d5ad61337b52b3b6a87eae0fd4fb49535085af |
| lensrank/adapters.py | 36354cf91e2630fb0d550719f50508c45d8e5b2f2cbbebedc4ac5c9f31d98fd7 |
| lensrank/checkpoints.py | b8a0ac5e6b79d9e7523a42b60e354613c3ceed1251476eeaf8be1a2ac5975015 |
| lensrank/config.py | b38c05da802ff3ea6c1e66f5e9ea1b2049c7bae9f813f3fbb914cea0abf7b6a7 |
| lensrank/data.py | 93613d7a8cc9de69ccabdf83a7800b06793b451dda30aa21c70f305cff81a446 |
| lensrank/experiments.py | b274955587dc4b107d3c4df0a17ef39bf6c80a7e4fdf182631c2a191e4ec8b6a |
