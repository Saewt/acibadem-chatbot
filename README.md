# Acibadem Chatbot

Docker Compose ile ayağa kalkan Django tabanlı bir üniversite chatbot uygulaması. Uygulama PostgreSQL + pgvector, Redis, Docker Model Runner ve `sentence-transformers` kullanır.

## Gereksinimler

- Git
- Docker Desktop
- Docker Desktop içinde `Docker Model Runner` özelliği açık olmalı

## Windows Quickstart

Komutlar PowerShell içinde çalıştırılabilir.

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

### 3. Model Runner modelini indir

Docker Desktop ayarlarında `Settings > Features in development > Docker Model Runner` etkin olmalı.

```powershell
docker model pull ai/qwen3:4B-UD-Q4_K_XL
docker model list
```

Listede `ai/qwen3:4B-UD-Q4_K_XL` görünmelidir.

### 4. Uygulamayı başlat

```powershell
docker compose up --build -d
```

Bu komut:

- PostgreSQL + pgvector container'ını başlatır
- Redis container'ını başlatır
- Django image'ını build eder
- migrate, collectstatic ve `warm_models` komutlarını çalıştırır

İlk build uzun sürebilir. Image içinde Playwright Chromium ve embedding modeli bulunduğu için birkaç dakika beklemek normaldir.

### 5. Hızlı veri bootstrap yap

`docker compose up` tek başına içerik verisini yüklemez. Frontend'de anlamlı chat sonucu görmek için önce veri çekip embedding üretmek gerekir.

Hızlı başlangıç için sınırlı sayfa scrape et:

```powershell
docker compose exec web python manage.py scrape_main_site --max-pages 50
docker compose exec web python manage.py generate_embeddings
```

Daha geniş veri istersen:

```powershell
docker compose exec web python manage.py scrape_main_site
docker compose exec web python manage.py generate_embeddings --rebuild
```

### 6. Uygulamayı test et

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

### `warm_models` uyarı veriyor

Önce model runner ve modeli kontrol et:

```powershell
docker model list
```

Gerekirse modeli tekrar indir:

```powershell
docker model pull ai/qwen3:4B-UD-Q4_K_XL
```

Ardından web loglarını incele:

```powershell
docker compose logs web
```

### `localhost:8000` açılmıyor

Web container durumunu ve loglarını kontrol et:

```powershell
docker compose ps
docker compose logs web
```

### Chat cevap veriyor ama içerik boş veya alakasız

Muhtemel neden verinin henüz yüklenmemiş olmasıdır. Şunları tekrar çalıştır:

```powershell
docker compose exec web python manage.py scrape_main_site --max-pages 50
docker compose exec web python manage.py generate_embeddings
```

### Baştan temiz kurulum yapmak istiyorum

```powershell
docker compose down -v
docker compose up --build -d
docker compose exec web python manage.py scrape_main_site --max-pages 50
docker compose exec web python manage.py generate_embeddings
```
