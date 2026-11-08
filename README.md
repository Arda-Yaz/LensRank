# LensRank

LensRank, doğal dil sorgusuyla görsel ve yüklenen görselle açıklama arayan bir Gradio uygulamasıdır.
Arama koleksiyonu validation kümesidir.
OpenCLIP ViT-B-16/OpenAI yapılandırması ve LoRA katmanları kullanılır.
Model seçimi karşılaştırma CSV'sindeki mean_score sıralamasından yapılır.

## Kurulum

Python 3.12 ile proje klasöründe bir sanal ortam oluşturun.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
```

## Çalıştırma

```powershell
.\.venv\Scripts\python.exe app.py
```

Text → Images sekmesinde bir görseli doğal dille tarif edin.
Image → Captions sekmesinde bir görsel yükleyin.
Her iki aramada 1–10 sonuç seçilebilir.

İlk açılış validation embedding'lerini artifacts/fresh_runs/demo_validation_index.pt dosyasına yazar.
Checkpoint değişiklik zamanı aynıysa sonraki açılışlarda bu cache kullanılır.
Demo eval_batch_size=256 ve num_workers=0 ayarlarıyla çalışır.
app modülünün import edilmesi de model ve cache yüklemesini başlatır.

## Dosyalar

- app.py Gradio arayüzünü ve arama callback'lerini içerir.
- lensrank/ model, veri, encoding ve evaluation modüllerini içerir.
- resumable_experiments.py notebook için Python girişlerini sağlar.
- train_all_models_resumable.ipynb deney yapılandırması ve eğitim hücrelerini içerir.
- data/ veri tablolarını ve train/validation/test görsellerini içerir.
- artifacts/fresh_runs/ checkpoint, karşılaştırma ve cache dosyalarını içerir.
- SOURCE_MANIFEST.md dosya envanterini ve doğrulama durumunu içerir.
- PORT_MAP.md modül sorumluluklarını gösterir.

Eğitim notebook'u çalıştırıldığında uzun süren GPU işleri başlatabilir.
Arayüzü açmak için notebook'un çalıştırılması gerekmez.
