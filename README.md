# LensRank

LensRank, doğal dil sorgusuyla görsel ve yüklenen görselle açıklama arayan bir Gradio uygulamasıdır.
Arama koleksiyonu validation kümesidir. OpenCLIP ViT-B-16/OpenAI modeli ve yerel LoRA checkpoint'i kullanılır.

Yerel dosya ve checkpoint kayıtları SOURCE_MANIFEST.md içindedir.
Model seçimi CSV içindeki mean_score sıralamasından yapılır.

## Kurulum ve açılış

Python ortamında requirements.txt içindeki bağımlılıkları kurun.
Tam sürüm kilidi henüz doğrulanmadı.
PyTorch kurulumu kullanılan GPU ve CUDA ortamıyla birlikte kontrol edilmelidir.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Komutları LensRank klasöründe çalıştırın. Önce aşağıdaki yerel dosyaları tamamlayın.
Mevcut uygulama açılışta training görsellerinin varlığını da kontrol eder.

| Varlık | Yerel durum |
| --- | --- |
| artifacts/fresh_runs/openclip_lora/best.pt | Boyut ve SHA-256 doğrulandı |
| artifacts/fresh_runs/fresh_comparison.csv | Validation karşılaştırma kaydı |
| data/train.csv, val.csv, test.csv | Henüz alınmadı |
| data/image_pool.csv, caption_pool.csv | Henüz alınmadı |
| data/train_images/train_images/ | Henüz alınmadı |
| data/val_images/val_images/ | Henüz alınmadı |
| data/test_images/test_images/ | Henüz alınmadı |
| artifacts/fresh_runs/*/history.csv, completed.json | Henüz alınmadı |

İlk açılış modeli yükler ve artifacts/fresh_runs/demo_validation_index.pt dosyasını üretir.
Sonraki açılışlar checkpoint değişiklik zamanı aynıysa bu cache'i kullanır.
Demo ayarları eval_batch_size=256 ve num_workers=0 olarak korunmuştur.
app modülünü import etmek de model yükleme ve cache hazırlığı başlatır.

## Kod girişleri

- app.py Gradio uygulamasını açar.
- resumable_experiments.py mevcut Python import girişlerini sağlar.
- lensrank/ uygulama mantığını içerir. Modül eşlemeleri PORT_MAP.md içindedir.
- train_all_models_resumable.ipynb mevcut eğitim ve evaluation girişlerini korur.
  Çalıştırıldığında uzun eğitim işleri başlatabilir. Kurulum kontrolünde eğitim hücreleri çalıştırılmadı.
  Kaydedilmiş hücre çıktıları temizlendi. Parametreler ve eğitim hücreleri korundu.

## Hazırlık kontrolü

7 Kasım 2026 tarihinde Python dosyaları ve notebook kod hücreleri derleme kontrolünden geçti.
Temel lensrank paketi import edildi.
Checkpoint dosyasının SHA-256 ve boyutu eşleşti. ZIP dizininde data.pkl ve 404 kayıt görüldü.
Tensor yükleme, GPU inference ve arama sonuçları henüz doğrulanmadı.

Makinedeki python ve py komutları erişilemeyen Windows uygulama takma adlarına işaret ediyor.
Temel kontroller Codex ile gelen Python 3.12.14 kullanılarak yapıldı.
Bu ortamda torch, open_clip ve gradio kurulu değil.

Training DataLoader içindeki yerel collate fonksiyonu ve num_workers=4 varsayılanı korunmuştur.
Windows multiprocessing uyumluluğu henüz doğrulanmadı.
