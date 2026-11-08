# Dosya envanteri

## Model

- Yapılandırma OpenCLIP ViT-B-16/OpenAI
- LoRA hedefleri c_fc, c_proj, fc1, fc2
- LoRA rank 8, alpha 16
- Checkpoint artifacts/fresh_runs/openclip_lora/best.pt
- State dictionary 398 anahtar içerir ve strict yükleme tamamlandı.
- Validation embedding boyutları görseller için 1000 × 512, açıklamalar için 5000 × 512.

## Veri

| Bölüm | CSV satırı | Görsel |
| --- | --- | --- |
| Train | 40000 | 8000 |
| Validation | 5000 | 1000 |
| Test | 1000 | 1000 |

Train ve validation tabloları image_id/caption_id eşlemelerini içerir.
Test tablosu id/query/query_type alanlarıyla 500 image ve 500 caption sorgusu içerir.
10.000 görselin açılabilirliği ve sorgu ID eşlemeleri kontrol edildi.
data/manifest.json veri dosyaları ve deney kayıtlarının dosya bazında SHA-256 envanterini içerir.
Görseller ve ağırlık dosyaları yerel diskte tutulur.

## Ortam

8 Kasım 2026 tarihinde Windows üzerinde doğrulandı.

- Python 3.12.14
- GPU NVIDIA GeForce RTX 5060, 8151 MiB
- PyTorch 2.7.1+cu128
- Torchvision 0.22.1+cu128
- OpenCLIP 2.32.0
- Gradio 5.49.1
- Uvicorn 0.35.0

Tam paket sürümleri requirements-lock.txt dosyasındadır.
Uvicorn sürümü, Gradio sunucusunun Windows'ta hatasız kapanması için 0.35.0 olarak sabitlendi.

## Doğrulama

- Checkpoint boyutu ve SHA-256 eşleşti.
- Modelin GPU üzerinde yüklenmesi ve inference tamamlandı.
- Metin aramasında ve görselden açıklama aramasında 1, 5 ve 10 sonuç kontrol edildi.
- Boş metin ve eksik görsel davranışları kontrol edildi.
- Embedding değerlerinin sonlu ve normalize olduğu doğrulandı.
- Cache yeniden kullanıldığında dosyanın değişiklik zamanı ve embedding içeriği aynı kaldı.
- Dört worker ile yüklenen gerçek veri batch'leri tek işlemli yüklemeyle birebir eşleşti.
- Yerel HTTP yanıtı 200. İki Gradio arama endpoint'i beşer sonuç döndürdü.
- Sunucu açılışı ve kapanışı tamamlandı.
- pip check bağımlılık uyuşmazlığı bildirmedi.
- Python dosyaları ve notebook kod hücreleri sözdizimi kontrolünden geçti.

OpenCLIP ViT-B-16/OpenAI eşleşmesinde QuickGELU aktivasyonu hakkında uyarı verir.
Model adı ve aktivasyon seçimi yapılandırmadaki haliyle kullanılır.
Validation cache'i üzerinden altı Recall metriği hesaplandı ve karşılaştırma tablosuyla eşleşti.

| Yön | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| Image → Text | 0.886 | 0.988 | 0.999 |
| Text → Image | 0.7592 | 0.9452 | 0.976 |

Bu kontrolde cache embedding'leri kullanıldı.

## SHA-256 kayıtları

| Dosya | Bayt | SHA-256 |
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
