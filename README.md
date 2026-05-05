# Acibadem Chatbot

Docker Compose ile ayağa kalkan Django tabanlı bir üniversite chatbot uygulaması. Uygulama PostgreSQL + pgvector, Redis, Docker Model Runner ve `sentence-transformers` kullanır. Varsayılan akışta Django servisleri Docker içinde çalışır, embedding API ise host terminalinden açılır.

## Gereksinimler

- Git
- Docker Desktop
- Docker Desktop içinde `Docker Model Runner` özelliği açık olmalı

## Quickstart

Komutlar macOS terminalinde veya PowerShell içinde çalıştırılabilir.

### 1. Repoyu klonla

```powershell
git clone https://github.com/Saewt/acibadem-chatbot.git
cd acibadem-chatbot
```

### 2. Ortam dosyasını oluştur

```powershell
Copy-Item .env.example .env
```

Varsayılan `.env` değerleri yerel deneme için yeterlidir. Farklı bir port veya veritabanı adı kullanacaksan `.env` dosyasını güncelle.

Repo, ilk deneme için temizlenmiş bir bilgi tabanı snapshot'ı içerir. Varsayılan dataset yolu `./data/acibadem-dataset` olduğu için Windows'ta ekstra path ayarı yapmadan bootstrap çalışır. Farklı bir dataset kullanmak istersen `.env` içinde `ACIBADEM_DATASET_HOST_ROOT` değerini örneğin `C:/Users/<kullanici>/Desktop/acibadem-dataset` olarak değiştir.

### 3. Model Runner modelini indir

Docker Desktop ayarlarında `Settings > Features in development > Docker Model Runner` etkin olmalı.

```powershell
docker model pull docker.io/qwen3:4B-UD-Q4_K_XL
docker model list
```

Listede `docker.io/qwen3:4B-UD-Q4_K_XL` görünmelidir. Bu 4B model kullanılacağı için varsayılan `.env` tek eşzamanlı LLM isteği, kısa cevap limiti ve küçük RAG context ile gelir.

### 4. Host embedding API'yi başlat

Varsayılan `.env` ayarı `EMBEDDING_BACKEND=api` ve `EMBEDDING_API_URL=http://host.docker.internal:8001` kullanır. Bu yüzden embedding API'yi host terminalinde aç:

```powershell
cd embedding_api
uvicorn main:app --host 0.0.0.0 --port 8001
```

Bu süreç açık kalmalıdır. API sağlık kontrolü için:

```powershell
curl http://localhost:8001/health
```

Embedding API parent `.env` dosyasını okur. Varsayılan `EMBEDDING_DEVICE=auto` olduğu için Apple Silicon üzerinde MPS/Metal kullanır; donma yaşarsan `EMBEDDING_DEVICE=cpu` yapıp servisi yeniden başlat.

### 5. Uygulamayı başlat

Geliştirme için önerilen akış:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

Bu modda `./webapp` container'a bind mount edilir ve Django `runserver` autoreload kullanır. Python, template ve static değişikliklerinde image rebuild gerekmez; dependency, Dockerfile veya compose değişirse tekrar build gerekir.

Production-benzeri yerel akış:

```powershell
docker compose up --build -d
```

Bu komut:

- PostgreSQL + pgvector container'ını başlatır
- Redis container'ını başlatır
- Django image'ını build eder
- migrate ve collectstatic çalıştırır
- Qwen warmup, veri bootstrap ve canlı sync başlatmaz

İlk build Playwright Chromium indirdiği için birkaç dakika sürebilir. Web image artık embedding modelini veya PyTorch'u indirmez; embedding modeli hosttaki `embedding_api` tarafından yönetilir.

### 6. Veri bootstrap davranışı

Varsayılan compose akışı ilk açılışta veri bootstrap yapmaz. Repo içinde gelen `./data/acibadem-dataset` snapshot'ı şu dosyaları içerir ve container içinde `/data/acibadem-dataset` olarak mount edilir:

- `acibadem_output/sources_clean.jsonl`
- `acibadem_output/chunks_clean.jsonl`
- `acibadem_output/records_clean.jsonl`
- `bologna_courses/sources.jsonl`
- `bologna_courses/records.jsonl`
- `bologna_courses/summary.json`

İlk veri yükleme için önerilen komut:

```powershell
docker compose --profile bootstrap run --rm bootstrap
```

Veritabanında aktif kayıt varsa bootstrap no-op olur ve tekrar import çalışmaz. Snapshot'ı force refresh ile yeniden içeri almak istersen:

```powershell
docker compose --profile bootstrap run --rm bootstrap --force-refresh --rebuild-embeddings
```

Bu snapshot ayrı bir scraper çıktısından üretilmiş clean JSONL verisidir. Proje bu snapshot'ı import eder ve sonrasında kendi canlı scraper/sync komutlarıyla veriyi güncelleyebilir; dış scraper'ın tüm üretim pipeline'ı bu repoya dahil değildir.

Canlı siteden sınırlı sayfa scrape etmek istersen:

```powershell
docker compose exec web python manage.py scrape_main_site --max-pages 50
docker compose exec web python manage.py generate_embeddings
```

Daha geniş veri istersen:

```powershell
docker compose exec web python manage.py scrape_main_site
docker compose exec web python manage.py generate_embeddings --rebuild
```

Canlı kaynağı kontrol edip sadece değişiklik varsa sistemi güncellemek için:

```powershell
docker compose exec web python manage.py sync_acibadem_knowledge --check-only
docker compose exec web python manage.py sync_acibadem_knowledge
```

`scheduler` servisi default stack'te açılmaz. Periyodik canlı kontrolü istiyorsan ayrıca başlat:

```powershell
docker compose --profile scheduler up -d scheduler
```

Varsayılan yerel ayarda `KNOWLEDGE_SYNC_ENABLED=False` ve `KNOWLEDGE_SYNC_RUN_ON_START=False` olduğu için scheduler ilk açılışta canlı crawl başlatmaz. Açmak istersen `.env` içinden `KNOWLEDGE_SYNC_ENABLED=True` yap.

### 7. Uygulamayı test et

Tarayıcıda aç:

```text
http://localhost:8000
```

Frontend smoke test için şu soruları deneyebilirsin:

- `Acıbadem Üniversitesi'nde Bilgisayar Mühendisliği var mı?`
- `Tıp Fakültesi yıllık ücreti nedir?`
- `Yurt olanakları var mı?`

Beklenen sonuç:

- Sayfa açılır
- Chat isteği hata vermeden döner
- Cevapla birlikte kaynak kartları görünür

## Yararlı Komutlar

### Logları izle

```powershell
docker compose logs -f web
docker compose logs -f db
docker compose logs -f redis
```

### Çalışan servisleri kontrol et

```powershell
docker compose ps
```

### Testleri çalıştır

```powershell
docker compose exec web python manage.py test chat scraper
```

### Servisleri durdur

```powershell
docker compose down
```

Volume'leri de silmek istersen:

```powershell
docker compose down -v
```

## Sorun Giderme

### Qwen yavaş veya sistem zorlanıyor

Varsayılan akış Qwen'i startup'ta ısıtmaz ve aynı anda yalnızca bir LLM isteğine izin verir. Modeli manuel kontrol etmek istersen:

```powershell
docker model list
```

Gerekirse modeli tekrar indir:

```powershell
docker model pull docker.io/qwen3:4B-UD-Q4_K_XL
```

Manuel warmup gerektiğinde açıkça çalıştır:

```powershell
docker compose exec web python manage.py warm_models --llm
```

Yanıt sırasında sistem hâlâ zorlanıyorsa `.env` içinde `LLM_MAX_TOKENS`, `RAG_RETRIEVE_LIMIT` ve `RAG_MAX_CONTEXT_CHARS` değerlerini daha da düşür. Varsayılanlar MacBook Air için düşük tutulur: tek LLM isteği, kısa cevap limiti ve küçük RAG context.

Host embedding API gerçekten ayakta mı kontrol et:

```powershell
curl http://localhost:8001/health
```

`EMBEDDING_BACKEND=api` iken `warm_models` local embedding warmup yapmaz; bu modda asıl bağımlılık host embedding API'nin erişilebilir olmasıdır. Varsayılan `EMBEDDING_DEVICE=auto` Metal/MPS kullanır. Takılma yaşarsan `EMBEDDING_DEVICE=cpu` yap.

### `localhost:8000` açılmıyor

Web container durumunu ve loglarını kontrol et:

```powershell
docker compose ps
docker compose logs web
```

### Chat cevap veriyor ama içerik boş veya alakasız

Muhtemel neden bootstrap'ın henüz çalıştırılmamış olması, veri dosyalarını bulamaması veya verinin henüz yüklenmemiş olmasıdır. Önce bootstrap ve mounted dataset yolunu kontrol et. Gerekirse şunları tekrar çalıştır:

```powershell
docker compose --profile bootstrap run --rm bootstrap
```

Canlı scrape ile veri tazelemek istersen şunları tekrar çalıştır:

```powershell
docker compose exec web python manage.py scrape_main_site --max-pages 50
docker compose exec web python manage.py generate_embeddings
```

Eğer embedding API hostta çalışıyorsa ayrıca şu kontrolü yap:

```powershell
curl http://localhost:8001/health
docker compose logs web
```

### Baştan temiz kurulum yapmak istiyorum

```powershell
docker compose down -v
docker compose up --build -d
docker compose --profile bootstrap run --rm bootstrap
```
