from io import StringIO
from unittest.mock import Mock, patch

import requests
from django.conf import settings
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from .embeddings import DEFAULT_EMBEDDING_BATCH_SIZE, iter_text_embedding_batches
from .models import ContentChunk, WebPage
from .services import (
    DEFAULT_MAIN_SITE_SEEDS,
    build_chunk_embedding_text,
    canonicalize_main_site_url,
    crawl_candidate_data,
    crawl_bologna,
    crawl_main_site,
    extract_candidate_fee_records,
    extract_candidate_score_records,
    extract_candidate_topic_pages,
    extract_bologna_page,
    extract_bologna_program_menu,
    extract_bologna_programs,
    ExtractedPage,
    extract_main_site_page,
    fetch_html,
    upsert_page_content,
)


def build_vector(seed: float) -> list[float]:
    return [seed] * 384


def normalize_vector(value) -> list[float]:
    if hasattr(value, 'tolist'):
        return value.tolist()
    return list(value)


class ScraperConfigTests(SimpleTestCase):
    def test_whitenoise_middleware_follows_security_middleware(self):
        security_index = settings.MIDDLEWARE.index('django.middleware.security.SecurityMiddleware')
        self.assertEqual(
            settings.MIDDLEWARE[security_index + 1],
            'whitenoise.middleware.WhiteNoiseMiddleware',
        )

    def test_contentchunk_uses_cosine_hnsw_index(self):
        index = next(
            index
            for index in ContentChunk._meta.indexes
            if index.name == 'contentchunk_embedding_hnsw_idx'
        )

        self.assertEqual(index.__class__.__name__, 'HnswIndex')
        self.assertEqual(index.fields, ['embedding'])
        self.assertEqual(index.opclasses, ['vector_cosine_ops'])


class ScraperServiceTests(TestCase):
    def test_build_chunk_embedding_text_prefixes_metadata(self):
        embedding_text = build_chunk_embedding_text(
            'Başkanlık bilgisi burada yer alır.',
            {
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'section_title': 'Bölüm Başkanı',
                'page_title': 'Bölüm Başkanı',
            },
        )

        self.assertIn('Program: Bilgisayar Mühendisliği (İngilizce)', embedding_text)
        self.assertIn('Fakulte: Mühendislik ve Doğa Bilimleri Fakültesi', embedding_text)
        self.assertIn('Bolum: Bölüm Başkanı', embedding_text)
        self.assertIn('Baslik: Bölüm Başkanı', embedding_text)
        self.assertTrue(embedding_text.endswith('Icerik: Başkanlık bilgisi burada yer alır.'))

    def test_canonicalize_main_site_url_filters_unsupported_targets(self):
        self.assertEqual(
            canonicalize_main_site_url('https://www.acibadem.edu.tr/tip-fakultesi/'),
            'https://www.acibadem.edu.tr/tip-fakultesi',
        )
        self.assertIsNone(canonicalize_main_site_url('https://www.acibadem.edu.tr/en'))
        self.assertIsNone(
            canonicalize_main_site_url('https://www.acibadem.edu.tr/duyurular/example')
        )
        self.assertIsNone(
            canonicalize_main_site_url('https://www.acibadem.edu.tr/files/katalog.pdf')
        )
        self.assertIsNone(
            canonicalize_main_site_url(
                'https://www.acibadem.edu.tr/sites/default/files/2025-12/lisans_4-1_0.csv'
            )
        )
        self.assertIsNone(
            canonicalize_main_site_url(
                "https://www.acibadem.edu.tr/akademik/ortak-dersler-bolumleri/{{ '/etkinlikler/arsiv' ~ raw_arguments.field_fakulte_secimi_target_id }}"
            )
        )
        self.assertEqual(
            canonicalize_main_site_url('https://www.acibadem.edu.tr/arastirma'),
            'https://www.acibadem.edu.tr/arastirma',
        )
        self.assertEqual(
            canonicalize_main_site_url('https://www.acibadem.edu.tr/aday/ogrenci'),
            'https://www.acibadem.edu.tr/aday/ogrenci',
        )
        self.assertEqual(
            canonicalize_main_site_url('https://www.acibadem.edu.tr/surdurulebilir-kampus'),
            'https://www.acibadem.edu.tr/surdurulebilir-kampus',
        )

    def test_extract_main_site_page_removes_sidebar_and_footer_noise(self):
        html = """
        <html>
          <body>
            <h1 class="page-title">Tıp Fakültesi</h1>
            <main>
              <div class="sidebar-custom-video-block">Video</div>
                <div class="sidebar-page-content">
                <div id="block-acu-content">
                  <p>Öğrenci işleri bilgisi ve başvuru süreçlerine dair ayrıntılı açıklama</p>
                  <p>Akademik takvim detayları ile kayıt yenileme adımları</p>
                </div>
              </div>
              <footer>Footer linkleri</footer>
            </main>
          </body>
        </html>
        """
        extracted = extract_main_site_page('https://www.acibadem.edu.tr/ogrenci', html)

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.title, 'Tıp Fakültesi')
        self.assertIn('Öğrenci işleri bilgisi', extracted.text)
        self.assertNotIn('Footer linkleri', extracted.text)
        self.assertNotIn('Video', extracted.text)

    def test_extract_main_site_page_filters_short_text(self):
        html = """
        <html>
          <body>
            <main>
              <p>Kısa metin</p>
            </main>
          </body>
        </html>
        """

        self.assertIsNone(extract_main_site_page('https://www.acibadem.edu.tr/kisa', html))

    def test_extract_main_site_page_uses_document_title_when_h1_missing(self):
        html = """
        <html>
          <head>
            <title>Acıbadem Üniversitesi Ana Sayfa</title>
          </head>
          <body>
            <main>
              <p>Acıbadem Üniversitesi hakkında yeterince uzun bir açıklama burada yer alıyor.</p>
            </main>
          </body>
        </html>
        """

        extracted = extract_main_site_page('https://www.acibadem.edu.tr/', html)

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.title, 'Acıbadem Üniversitesi Ana Sayfa')

    def test_extract_main_site_page_removes_program_news_tabs(self):
        html = """
        <html>
          <body>
            <h1 class="page-title">Biyomedikal Mühendisliği</h1>
            <main>
              <div id="block-acu-content">
                <div>
                  <h2>Bölüm Hakkında</h2>
                  <p>Biyomedikal mühendisliği programı hakkında yeterince uzun açıklama burada yer alıyor.</p>
                </div>
                <div class="right-cover-bg">
                  <button>Haberler</button>
                  <a href="/haberler/ornek">Program haberleri</a>
                </div>
              </div>
            </main>
          </body>
        </html>
        """

        extracted = extract_main_site_page(
            'https://www.acibadem.edu.tr/akademik/lisans/muhendislik-ve-doga-bilimleri-fakultesi/bolumler/biyomedikal-muhendisligi',
            html,
        )

        self.assertIsNotNone(extracted)
        self.assertIn('Bölüm Hakkında', extracted.text)
        self.assertNotIn('Program haberleri', extracted.text)
        self.assertNotIn('Haberler', extracted.text)

    def test_extract_main_site_page_marks_candidate_topic_pages(self):
        html = """
        <html>
          <body>
            <h1 class="page-title">Burs Olanakları</h1>
            <main>
              <div id="block-acu-content">
                <p>Burs olanakları ile ilgili yeterince uzun resmi açıklama burada yer almaktadır.</p>
              </div>
            </main>
          </body>
        </html>
        """

        extracted = extract_main_site_page(
            'https://www.acibadem.edu.tr/aday/ogrenci/egitim/burs/burs-olanaklari',
            html,
        )

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.metadata['kind'], 'candidate_topic_page')
        self.assertEqual(extracted.metadata['topic'], 'scholarships')
        self.assertEqual(extracted.metadata['topic_label'], 'Burs Olanakları')
        self.assertEqual(extracted.metadata['section_title'], 'Burs Olanakları')

    def test_extract_bologna_page_filters_short_text(self):
        html = """
        <html>
          <body>
            <div id="UpdatePanel1">Kısa içerik</div>
          </body>
        </html>
        """

        self.assertIsNone(
            extract_bologna_page(
                'https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=2&lang=tr',
                html,
                metadata={'kind': 'bologna_program_page'},
            )
        )

    def test_extract_bologna_programs_and_menu(self):
        unit_selection_html = """
        <div class="panel panel-default">
          <div class="panel-heading">
            <h5 class="panel-title">
              <a>Sağlık Bilimleri Fakültesi</a>
            </h5>
          </div>
          <ul class="list-group">
            <li class="list-group-item">
              <a href="index.aspx?lang=tr&curOp=showPac&curUnit=05&curSunit=2">
                Fizyoterapi ve Rehabilitasyon
              </a>
            </li>
          </ul>
        </div>
        """
        programs = extract_bologna_programs(unit_selection_html)

        self.assertEqual(len(programs), 1)
        self.assertEqual(programs[0]['faculty'], 'Sağlık Bilimleri Fakültesi')
        self.assertEqual(programs[0]['cur_sunit'], '2')

        program_menu_html = """
        <ul id="proMenu">
          <li><a class="nav-link" onclick="menu_close(this,'progAbout.aspx?lang=tr&curSunit=2');">Program Hakkında</a></li>
          <li><a class="nav-link" onclick="menu_close(this,'progCourses.aspx?lang=tr&curSunit=2');">Dersler</a></li>
          <li><a class="nav-link" onclick="menu_close(this,'dynConPage.aspx?curPageId=400&lang=tr');">Bologna</a></li>
        </ul>
        """
        sections = extract_bologna_program_menu(program_menu_html)

        self.assertEqual(
            [section['title'] for section in sections], ['Program Hakkında', 'Dersler']
        )
        self.assertTrue(sections[0]['url'].endswith('progAbout.aspx?curSunit=2&lang=tr'))

    def test_extract_bologna_program_menu_synthesizes_staff_pages(self):
        sections = extract_bologna_program_menu(
            """
            <ul id="proMenu">
              <li><a class="nav-link" onclick="menu_close(this,'progAbout.aspx?lang=tr&curSunit=6246');">Program Hakkında</a></li>
            </ul>
            """,
            program_id='6246',
        )

        self.assertEqual(
            [section['title'] for section in sections],
            ['Program Hakkında', 'Program Yetkilileri', 'Akademik Kadro'],
        )
        self.assertEqual(sections[1]['kind'], 'bologna_staff_page')
        self.assertEqual(sections[2]['staff_page_type'], 'academic_staff')

    def test_extract_main_site_page_structures_staff_entries(self):
        html = """
        <html>
          <body>
            <nav aria-label="breadcrumb">
              <ol class="breadcrumb">
                <li><a>Mühendislik ve Doğa Bilimleri Fakültesi</a></li>
                <li><a>Bilgisayar Mühendisliği</a></li>
                <li>Akademik Kadro</li>
              </ol>
            </nav>
            <main>
              <div id="block-acu-content">
                <div class="views-row">
                  <h3>Prof. Dr. Ayşe Yılmaz</h3>
                </div>
                <div class="views-row">
                  <h3>Dr. Öğr. Üyesi Mehmet Kaya</h3>
                </div>
              </div>
            </main>
          </body>
        </html>
        """

        extracted = extract_main_site_page(
            'https://www.acibadem.edu.tr/akademik/lisans/muhendislik-ve-doga-bilimleri-fakultesi/bolumler/bilgisayar-muhendisligi/akademik-kadro',
            html,
        )

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.metadata['kind'], 'main_site_staff_page')
        self.assertEqual(extracted.metadata['staff_count'], 2)
        self.assertEqual(extracted.metadata['program_title'], 'Bilgisayar Mühendisliği')
        self.assertEqual(
            extracted.metadata['faculty'],
            'Mühendislik ve Doğa Bilimleri Fakültesi',
        )
        self.assertIn('Toplam hoca sayisi: 2', extracted.text)
        self.assertIn('Ayşe Yılmaz', extracted.text)
        self.assertIn('Mehmet Kaya', extracted.text)

    def test_extract_bologna_page_structures_staff_entries(self):
        html = """
        <html>
          <body>
            <div class="panel-heading">Akademik Kadro</div>
            <div id="UpdatePanel1">
              <table>
                <thead>
                  <tr><th>Ad Soyad</th><th>Unvan</th></tr>
                </thead>
                <tbody>
                  <tr><td>Ayşe Yılmaz</td><td>Prof. Dr.</td></tr>
                  <tr><td>Mehmet Kaya</td><td>Dr. Öğr. Üyesi</td></tr>
                </tbody>
              </table>
            </div>
          </body>
        </html>
        """

        extracted = extract_bologna_page(
            'https://obs.acibadem.edu.tr/oibs/bologna/progAcademicStaff.aspx?curSunit=6246&lang=tr',
            html,
            metadata={
                'program_title': 'Bilgisayar Mühendisliği',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
            },
        )

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.metadata['kind'], 'bologna_staff_page')
        self.assertEqual(extracted.metadata['staff_page_type'], 'academic_staff')
        self.assertEqual(extracted.metadata['staff_count'], 2)
        self.assertIn('Toplam hoca sayisi: 2', extracted.text)
        self.assertIn('- Ayşe Yılmaz | Prof. Dr.', extracted.text)
        self.assertIn('- Mehmet Kaya | Dr. Öğr. Üyesi', extracted.text)

    def test_default_main_site_seeds_include_staff_pages(self):
        self.assertIn(
            'https://www.acibadem.edu.tr/akademik/lisans/muhendislik-ve-doga-bilimleri-fakultesi/akademik-kadro',
            DEFAULT_MAIN_SITE_SEEDS,
        )
        self.assertIn('https://www.acibadem.edu.tr/akademik/onlisans', DEFAULT_MAIN_SITE_SEEDS)

    def test_extract_candidate_topic_pages_returns_whitelist_order(self):
        html = """
        <div>
          <a href="/aday/ogrenci/egitim/burs/burs-olanaklari">Burslar</a>
          <a href="/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu">Puanlar</a>
        </div>
        """

        pages = extract_candidate_topic_pages(html)

        self.assertEqual(
            [page['topic'] for page in pages],
            [
                'admissions_scores',
                'tuition',
                'scholarships',
                'dormitory',
                'international',
                'double_major_minor',
            ],
        )
        self.assertEqual(pages[0]['url'], 'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu')
        self.assertEqual(pages[2]['url'], 'https://www.acibadem.edu.tr/aday/ogrenci/egitim/burs/burs-olanaklari')

    def test_extract_candidate_score_records_parses_program_rows(self):
        client = Mock()
        response = Mock(status_code=200)
        response.raise_for_status = Mock()
        response.json.return_value = {
            'status': True,
            'columns': [
                'Fakülte/Bölüm Adı',
                'Puan Türü',
                'Kontenjan',
                'Tavan Puan',
                'Başarı Sırası',
                'Taban Puan',
                'Taban Başarı Sırası',
            ],
            'data': [
                ['MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ', '', '', '', '', '', ''],
                [
                    'Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
                    'SAY',
                    '40',
                    '500',
                    '1000',
                    '450',
                    '30000',
                ],
            ],
        }
        client.get.return_value = response
        html = """
        <div class="datatable-wrapper">
          <h3 class="datatable-title">LİSANS 2025 KONTENJAN, PUAN ve BAŞARI SIRASI TABLOSU</h3>
          <table class="datatable-item" data-fid="12884"></table>
          <a href="/files/program.csv">İndir</a>
        </div>
        """

        records = extract_candidate_score_records(
            client,
            html,
            'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['metadata']['kind'], 'structured_admissions_score')
        self.assertEqual(record['metadata']['program_title'], 'Bilgisayar Mühendisliği (İngilizce)')
        self.assertEqual(
            record['metadata']['placement_label'],
            'Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
        )
        self.assertEqual(record['metadata']['placement_type'], '%50 İndirimli')
        self.assertEqual(record['metadata']['score_type'], 'SAY')
        self.assertEqual(record['metadata']['faculty'], 'MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ')
        self.assertEqual(
            record['metadata']['source_url'],
            'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
        )
        self.assertIn('Taban Puan: 450', record['text'])

    def test_extract_candidate_score_records_supports_onlisans_short_rows(self):
        client = Mock()
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            'status': True,
            'columns': [
                'Fakülte/Bölüm Adı',
                'Puan Türü',
                'Kontenjan',
                'Taban Puan',
                'Taban Başarı Sırası',
            ],
            'data': [
                ['SAĞLIK HİZMETLERİ MESLEK YÜKSEKOKULU', '', '', '', ''],
                [
                    'Anestezi (%50 İndirimli)',
                    'TYT',
                    '42',
                    '311,05705',
                    '725.743',
                ],
            ],
        }
        client.get.return_value = response
        html = """
        <div class="datatable-wrapper">
          <h3 class="datatable-title">ÖNLİSANS 2025 KONTENJAN, PUAN ve BAŞARI SIRASI TABLOSU</h3>
          <table class="datatable-item" data-fid="11917"></table>
        </div>
        """

        records = extract_candidate_score_records(
            client,
            html,
            'https://www.acibadem.edu.tr/aday/ogrenci/egitim/onlisans/onlisans-kontenjan-ve-puan-tablosu',
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['metadata']['admission_level'], 'onlisans')
        self.assertEqual(record['metadata']['score_type'], 'TYT')
        self.assertEqual(record['metadata']['quota'], '42')
        self.assertEqual(record['metadata']['top_score'], '')
        self.assertEqual(record['metadata']['top_rank'], '')
        self.assertEqual(record['metadata']['base_score'], '311,05705')
        self.assertEqual(record['metadata']['base_rank'], '725.743')

    def test_extract_candidate_fee_records_parses_program_notes(self):
        html = """
        <table>
          <tr><th colspan="5">ACIBADEM 2025-2026 LİSANS PROGRAMLARI ÖĞRENİM ÜCRETLERİ</th></tr>
          <tr>
            <th>AKADEMİK PROGRAM</th>
            <th>Ücretli</th>
            <th>25% İndirimli Ücret</th>
            <th>50% İndirimli Ücret</th>
            <th>İlave %25 KAV Destek Burslu ücret</th>
          </tr>
          <tr><td colspan="5">MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ</td></tr>
          <tr>
            <td>Bilgisayar Mühendisliği (İngilizce)** Bilgisayar Mühendisliği (İngilizce) bölümü için ayrıca Acıbadem Sağlık Grubu Bursu bulunmaktadır. Burs hakkında bilgi almak için tıklayın. (Bilgisayar Mühendisliğinin %50 indirimli kontenjanına yerleşen öğrencilere tercih sırasına bakılmaksızın ek olarak Kerem Aydınlar Vakfı tarafından ödenecek ücret üzerinden %25 oranında Eğitim bursu desteği sağlanacaktır.)</td>
            <td></td>
            <td></td>
            <td>675.000₺</td>
            <td>506.250₺</td>
          </tr>
        </table>
        """

        records = extract_candidate_fee_records(
            html,
            'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-ogrenim-ucretleri-2025-2026',
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['metadata']['kind'], 'structured_admissions_fee')
        self.assertEqual(record['metadata']['program_title'], 'Bilgisayar Mühendisliği (İngilizce)')
        self.assertEqual(record['metadata']['fee_50'], '675.000₺')
        self.assertEqual(record['metadata']['fee_kav_support'], '506.250₺')
        self.assertIn('Acıbadem Sağlık Grubu Bursu', record['metadata']['notes'])
        self.assertIn('Öğrenim Ücreti', record['title'])

    def test_extract_bologna_programs_deduplicates_repeated_program_ids(self):
        unit_selection_html = """
        <div class="panel panel-default">
          <div class="panel-heading">
            <h5 class="panel-title">
              <a>Eczacılık Fakültesi</a>
            </h5>
          </div>
          <div class="panel-body">
            <div class="panel panel-default">
              <div class="panel-heading">
                <h5 class="panel-title">
                  <a>Fen Edebiyat Fakültesi</a>
                </h5>
              </div>
              <ul class="list-group">
                <li class="list-group-item">
                  <a href="index.aspx?lang=tr&curOp=showPac&curUnit=05&curSunit=12">Psikoloji</a>
                </li>
                <li class="list-group-item">
                  <a href="index.aspx?lang=tr&curOp=showPac&curUnit=05&curSunit=13">Sosyoloji</a>
                </li>
              </ul>
            </div>
            <div class="panel panel-default">
              <div class="panel-heading">
                <h5 class="panel-title">
                  <a>İnsan ve Toplum Bilimleri Fakültesi</a>
                </h5>
              </div>
              <ul class="list-group">
                <li class="list-group-item">
                  <a href="index.aspx?lang=tr&curOp=showPac&curUnit=07&curSunit=12">Psikoloji</a>
                </li>
              </ul>
            </div>
          </div>
        </div>
        """

        programs = extract_bologna_programs(unit_selection_html)

        self.assertEqual(len(programs), 2)
        self.assertEqual(
            {program['cur_sunit']: program['faculty'] for program in programs},
            {'12': 'İnsan ve Toplum Bilimleri Fakültesi', '13': 'Fen Edebiyat Fakültesi'},
        )

    def test_extract_bologna_programs_skips_wrapper_panel(self):
        unit_selection_html = """
        <div class="panel panel-default">
          <div class="panel-heading">
            <h5 class="panel-title">
              <a>Eczacılık Fakültesi</a>
            </h5>
          </div>
          <div class="panel-body">
            <div class="panel panel-default">
              <div class="panel-heading">
                <h5 class="panel-title">
                  <a>Mühendislik ve Doğa Bilimleri Fakültesi</a>
                </h5>
              </div>
              <ul class="list-group">
                <li class="list-group-item">
                  <a href="index.aspx?lang=tr&curOp=showPac&curUnit=07&curSunit=6246">
                    Bilgisayar Mühendisliği (İngilizce)
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
        """

        programs = extract_bologna_programs(unit_selection_html)

        self.assertEqual(len(programs), 1)
        self.assertEqual(programs[0]['cur_sunit'], '6246')
        self.assertEqual(programs[0]['faculty'], 'Mühendislik ve Doğa Bilimleri Fakültesi')

    @patch('scraper.services.iter_text_embedding_batches')
    def test_upsert_page_content_recreates_chunks_when_content_changes(
        self, iter_batches_mock
    ):
        def fake_iter(texts, batch_size=DEFAULT_EMBEDDING_BATCH_SIZE):
            self.assertEqual(batch_size, DEFAULT_EMBEDDING_BATCH_SIZE)
            yield 0, [build_vector(0.0) for _ in texts]

        iter_batches_mock.side_effect = fake_iter
        page, changed = upsert_page_content(
            source='main_site',
            url='https://www.acibadem.edu.tr/akademik',
            title='Akademik',
            text='Birinci paragraf.\n\nİkinci paragraf.',
            raw_html='<main>Birinci paragraf.</main>',
            metadata={'kind': 'main_site_page'},
        )

        self.assertTrue(changed)
        self.assertEqual(WebPage.objects.count(), 1)
        self.assertGreater(ContentChunk.objects.count(), 0)

        page, changed = upsert_page_content(
            source='main_site',
            url='https://www.acibadem.edu.tr/akademik',
            title='Akademik',
            text='Birinci paragraf güncellendi.\n\nİkinci paragraf.',
            raw_html='<main>Birinci paragraf güncellendi.</main>',
            metadata={'kind': 'main_site_page'},
        )

        self.assertTrue(changed)
        self.assertEqual(WebPage.objects.count(), 1)
        self.assertEqual(page.chunks.count(), ContentChunk.objects.count())

    @patch('scraper.services.iter_text_embedding_batches')
    def test_upsert_page_content_assigns_embeddings_across_batches(self, iter_batches_mock):
        def fake_iter(texts, batch_size=DEFAULT_EMBEDDING_BATCH_SIZE):
            self.assertEqual(batch_size, DEFAULT_EMBEDDING_BATCH_SIZE)
            self.assertEqual(len(texts), 3)
            self.assertTrue(all(text.startswith('Baslik: Bolum\nIcerik: ') for text in texts))
            yield 0, [build_vector(1.0), build_vector(2.0)]
            yield 2, [build_vector(3.0)]

        iter_batches_mock.side_effect = fake_iter
        text = '\n\n'.join(['A' * 600, 'B' * 600, 'C' * 600])

        page, changed = upsert_page_content(
            source='main_site',
            url='https://www.acibadem.edu.tr/bolum',
            title='Bolum',
            text=text,
            raw_html='<main>Bolum</main>',
            metadata={'kind': 'main_site_page'},
        )

        self.assertTrue(changed)
        chunks = list(page.chunks.order_by('chunk_index'))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [normalize_vector(chunk.embedding) for chunk in chunks],
            [build_vector(1.0), build_vector(2.0), build_vector(3.0)],
        )

    @patch('scraper.services.iter_text_embedding_batches')
    def test_upsert_page_content_truncates_titles_to_model_limit(self, iter_batches_mock):
        iter_batches_mock.return_value = [(0, [build_vector(1.0)])]
        long_title = 'T' * 520

        page, changed = upsert_page_content(
            source='bologna',
            url='https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=61&lang=tr',
            title=long_title,
            text='Bu içerik model sınırını aşan başlıkları test etmek için yeterince uzundur.',
            raw_html='<div>Bu içerik model sınırını aşan başlıkları test etmek için yeterince uzundur.</div>',
            metadata={'kind': 'bologna_program_page'},
        )

        self.assertTrue(changed)
        self.assertEqual(len(page.title), 500)
        self.assertEqual(page.title, long_title[:500])
        self.assertEqual(page.chunks.get().metadata['page_title'], long_title[:500])

    @patch('scraper.services.iter_text_embedding_batches')
    def test_upsert_page_content_syncs_chunk_metadata_without_rebuild(self, iter_batches_mock):
        iter_batches_mock.return_value = [(0, [build_vector(1.0)])]
        url = 'https://obs.acibadem.edu.tr/oibs/bologna/progOfficials.aspx?curSunit=6246&lang=tr'
        text = 'Yeterince uzun bir bologna içerik metni burada yer alıyor ve kaydedilmeli.'

        page, changed = upsert_page_content(
            source='bologna',
            url=url,
            title='Bölüm Başkanı',
            text=text,
            raw_html='<div>icerik</div>',
            metadata={
                'kind': 'bologna_program_page',
                'faculty': 'Eczacılık Fakültesi',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
            },
        )

        original_chunk = page.chunks.get()
        original_chunk_id = original_chunk.id
        original_embedding = normalize_vector(original_chunk.embedding)

        page, changed = upsert_page_content(
            source='bologna',
            url=url,
            title='Bölüm Başkanı',
            text=text,
            raw_html='<div>icerik</div>',
            metadata={
                'kind': 'bologna_program_page',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
            },
        )

        self.assertFalse(changed)
        updated_chunk = page.chunks.get()
        self.assertEqual(updated_chunk.id, original_chunk_id)
        self.assertEqual(normalize_vector(updated_chunk.embedding), original_embedding)
        self.assertEqual(
            updated_chunk.metadata['faculty'], 'Mühendislik ve Doğa Bilimleri Fakültesi'
        )
        self.assertEqual(iter_batches_mock.call_count, 1)

    @patch('scraper.services.mark_missing_pages_inactive', return_value=0)
    @patch('scraper.services.upsert_page_content', return_value=(Mock(), True))
    @patch('scraper.services.extract_main_site_links', return_value=[])
    @patch('scraper.services.extract_main_site_page')
    @patch('scraper.services.fetch_html')
    def test_crawl_main_site_continues_when_one_url_fails(
        self,
        fetch_html_mock,
        extract_main_site_page_mock,
        _extract_main_site_links_mock,
        upsert_page_content_mock,
        mark_missing_pages_inactive_mock,
    ):
        first_url = 'https://www.acibadem.edu.tr/ilk'
        second_url = 'https://www.acibadem.edu.tr/ikinci'
        fetch_html_mock.side_effect = [
            None,
            '<html><main>Yeterince uzun bir içerik metni burada yer alıyor.</main></html>',
        ]
        extract_main_site_page_mock.return_value = ExtractedPage(
            url=second_url,
            title='Ikinci',
            text='Yeterince uzun bir içerik metni burada yer aliyor ve kaydedilmeli.',
            raw_html='<main>icerik</main>',
            metadata={'kind': 'main_site_page'},
        )

        summary = crawl_main_site(
            session=Mock(),
            seeds=[first_url, second_url],
            max_pages=10,
            rate_limit_delay=0,
        )

        self.assertEqual(summary['seen'], 2)
        self.assertEqual(summary['saved'], 1)
        self.assertEqual(summary['updated'], 1)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['deactivated'], 0)
        self.assertEqual(fetch_html_mock.call_count, 2)
        upsert_page_content_mock.assert_called_once()
        mark_missing_pages_inactive_mock.assert_not_called()

    @patch('scraper.services.mark_missing_pages_inactive', return_value=0)
    @patch('scraper.services.upsert_page_content', return_value=(Mock(), True))
    @patch('scraper.services.extract_bologna_page')
    @patch('scraper.services.extract_bologna_program_menu')
    @patch('scraper.services.extract_bologna_programs')
    @patch('scraper.services.fetch_html')
    def test_crawl_bologna_skips_duplicate_program_ids(
        self,
        fetch_html_mock,
        extract_bologna_programs_mock,
        extract_bologna_program_menu_mock,
        extract_bologna_page_mock,
        upsert_page_content_mock,
        mark_missing_pages_inactive_mock,
    ):
        fetch_html_mock.side_effect = [
            '<html>unit selection</html>',
            '<html>program index</html>',
            '<html>section page</html>',
        ]
        extract_bologna_programs_mock.return_value = [
            {
                'title': 'Psikoloji',
                'faculty': 'Eczacılık Fakültesi',
                'cur_sunit': '12',
                'cur_unit': '05',
                'index_url': 'https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?curSunit=12&lang=tr',
            },
            {
                'title': 'Psikoloji',
                'faculty': 'Fen Edebiyat Fakültesi',
                'cur_sunit': '12',
                'cur_unit': '05',
                'index_url': 'https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?curSunit=12&lang=tr',
            },
        ]
        extract_bologna_program_menu_mock.return_value = [
            {
                'title': 'Program Hakkında',
                'url': 'https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=12&lang=tr',
            }
        ]
        extract_bologna_page_mock.return_value = ExtractedPage(
            url='https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=12&lang=tr',
            title='Psikoloji - Programı Bilgileri',
            text='Yeterince uzun bir bologna içerik metni burada yer alıyor ve kaydedilmeli.',
            raw_html='<div>icerik</div>',
            metadata={'kind': 'bologna_program_page'},
        )

        summary = crawl_bologna(
            session=Mock(),
            unit_types=['lis'],
            include_general_pages=False,
            rate_limit_delay=0,
        )

        self.assertEqual(summary['seen'], 1)
        self.assertEqual(summary['saved'], 1)
        self.assertEqual(summary['updated'], 1)
        self.assertEqual(summary['failed'], 0)
        self.assertEqual(summary['deactivated'], 0)
        self.assertEqual(fetch_html_mock.call_count, 3)
        extract_bologna_program_menu_mock.assert_called_once_with(
            '<html>program index</html>',
            program_id='12',
        )
        upsert_page_content_mock.assert_called_once()
        mark_missing_pages_inactive_mock.assert_not_called()

    @patch('scraper.services.mark_missing_pages_inactive', return_value=7)
    @patch('scraper.services.upsert_page_content', return_value=(Mock(), True))
    @patch('scraper.services.extract_bologna_page')
    @patch('scraper.services.extract_bologna_program_menu')
    @patch('scraper.services.extract_bologna_programs')
    @patch('scraper.services.fetch_html')
    @patch('scraper.services.extract_bologna_general_pages')
    def test_crawl_bologna_deactivates_only_for_full_scope(
        self,
        extract_bologna_general_pages_mock,
        fetch_html_mock,
        extract_bologna_programs_mock,
        extract_bologna_program_menu_mock,
        extract_bologna_page_mock,
        upsert_page_content_mock,
        mark_missing_pages_inactive_mock,
    ):
        fetch_html_mock.side_effect = [
            '<html>base index</html>',
            '<html>general page</html>',
            '<html>unit selection</html>',
            '<html>program index</html>',
            '<html>section page</html>',
            '<html>unit selection</html>',
            '<html>unit selection</html>',
            '<html>unit selection</html>',
        ]
        extract_bologna_general_pages_mock.return_value = [
            'https://obs.acibadem.edu.tr/oibs/bologna/dynConPage.aspx?curPageId=100&lang=tr'
        ]
        extract_bologna_programs_mock.side_effect = [
            [
                {
                    'title': 'Psikoloji',
                    'faculty': 'Fen Edebiyat Fakültesi',
                    'cur_sunit': '12',
                    'cur_unit': '05',
                    'index_url': 'https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?curSunit=12&lang=tr',
                }
            ],
            [],
            [],
            [],
        ]
        extract_bologna_program_menu_mock.return_value = [
            {
                'title': 'Program Hakkında',
                'url': 'https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=12&lang=tr',
            }
        ]
        extract_bologna_page_mock.return_value = ExtractedPage(
            url='https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=12&lang=tr',
            title='Psikoloji - Programı Bilgileri',
            text='Yeterince uzun bir bologna içerik metni burada yer alıyor ve kaydedilmeli.',
            raw_html='<div>icerik</div>',
            metadata={'kind': 'bologna_program_page'},
        )

        summary = crawl_bologna(
            session=Mock(),
            unit_types=['myo', 'lis', 'yls', 'dok'],
            include_general_pages=True,
            rate_limit_delay=0,
        )

        self.assertEqual(summary['deactivated'], 7)
        mark_missing_pages_inactive_mock.assert_called_once_with(
            'bologna',
            {
                'https://obs.acibadem.edu.tr/oibs/bologna/dynConPage.aspx?curPageId=100&lang=tr',
                'https://obs.acibadem.edu.tr/oibs/bologna/progAbout.aspx?curSunit=12&lang=tr',
            },
        )


class FetchHtmlTests(SimpleTestCase):
    def test_fetch_html_uses_drupal_ajax_for_staff_pages(self):
        session = Mock()
        page_response = Mock(
            status_code=200,
            text=(
                '<html><body>'
                '<script type="application/json" data-drupal-selector="drupal-settings-json">'
                '{"ajaxBlocks":{"akademik_kadro_v2_config_key_155":{"plugin_id":"acibadem_akademik_kadro_block_v2","block_id":"akademik_kadro_v2_config_key_155","settings":{"program":"155","hide_footer":0}}}}'
                '</script>'
                '<div data-block-ek-id="akademik_kadro_v2_config_key_155"></div>'
                '</body></html>'
            ),
        )
        page_response.raise_for_status = Mock()
        token_response = Mock(status_code=200, text='csrf-token')
        token_response.raise_for_status = Mock()
        ajax_response = Mock(status_code=200)
        ajax_response.raise_for_status = Mock()
        ajax_response.json.return_value = [
            {
                'selector': '[data-block-ek-id="akademik_kadro_v2_config_key_155"]',
                'data': '<div class="akademik-kadro-item"><span>Prof. Dr. Ayşe Yılmaz</span></div>',
            }
        ]
        session.get.side_effect = [page_response, token_response]
        session.post.return_value = ajax_response

        html = fetch_html(
            session,
            'https://www.acibadem.edu.tr/akademik/lisans/muhendislik-ve-doga-bilimleri-fakultesi/akademik-kadro',
            rate_limit_delay=0,
        )

        self.assertIn('Ayşe Yılmaz', html)
        self.assertEqual(session.get.call_count, 2)
        session.post.assert_called_once()

    @patch('scraper.services.time.sleep')
    def test_fetch_html_retries_then_succeeds(self, sleep_mock):
        session = Mock()
        response = Mock(status_code=200, text='<html>ok</html>')
        response.raise_for_status = Mock()
        session.get.side_effect = [
            requests.exceptions.ConnectionError('temporary failure'),
            response,
        ]

        html = fetch_html(
            session,
            'https://www.acibadem.edu.tr/akademik',
            rate_limit_delay=0,
        )

        self.assertEqual(html, '<html>ok</html>')
        self.assertEqual(session.get.call_count, 2)
        self.assertGreaterEqual(sleep_mock.call_count, 3)

    @patch('scraper.services.time.sleep')
    def test_fetch_html_returns_none_after_retry_exhaustion(self, sleep_mock):
        session = Mock()
        session.get.side_effect = requests.exceptions.ConnectionError('still failing')

        html = fetch_html(
            session,
            'https://www.acibadem.edu.tr/akademik',
            rate_limit_delay=0,
        )

        self.assertIsNone(html)
        self.assertEqual(session.get.call_count, 3)
        self.assertGreaterEqual(sleep_mock.call_count, 5)


class EmbeddingHelperTests(SimpleTestCase):
    @patch('scraper.embeddings.get_embedding_model')
    def test_iter_text_embedding_batches_splits_inputs_and_preserves_order(
        self, get_embedding_model_mock
    ):
        class FakeVector(list):
            def tolist(self):
                return list(self)

        class FakeModel:
            def __init__(self):
                self.calls = []
                self.seed = 0

            def encode(
                self,
                batch,
                *,
                batch_size,
                normalize_embeddings,
                show_progress_bar,
            ):
                self.calls.append(
                    {
                        'batch': list(batch),
                        'batch_size': batch_size,
                        'normalize_embeddings': normalize_embeddings,
                        'show_progress_bar': show_progress_bar,
                    }
                )
                vectors = []
                for _ in batch:
                    vectors.append(FakeVector(build_vector(float(self.seed))))
                    self.seed += 1
                return vectors

        fake_model = FakeModel()
        get_embedding_model_mock.return_value = fake_model

        batches = list(iter_text_embedding_batches(['a', 'b', 'c'], batch_size=2))

        self.assertEqual(
            batches,
            [
                (0, [build_vector(0.0), build_vector(1.0)]),
                (2, [build_vector(2.0)]),
            ],
        )
        self.assertEqual(
            fake_model.calls,
            [
                {
                    'batch': ['a', 'b'],
                    'batch_size': 2,
                    'normalize_embeddings': True,
                    'show_progress_bar': False,
                },
                {
                    'batch': ['c'],
                    'batch_size': 1,
                    'normalize_embeddings': True,
                    'show_progress_bar': False,
                },
            ],
        )


class GenerateEmbeddingsCommandTests(TestCase):
    def setUp(self):
        self.page = WebPage.objects.create(
            url='https://www.acibadem.edu.tr/test',
            source='main_site',
            title='Test',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )

    @patch('scraper.management.commands.generate_embeddings.iter_text_embedding_batches')
    def test_generate_embeddings_processes_only_missing_embeddings_in_windows(
        self, iter_batches_mock
    ):
        chunks = [
            ContentChunk.objects.create(page=self.page, chunk_index=index, text=f'chunk-{index}')
            for index in range(5)
        ]
        preserved_vector = build_vector(99.0)
        chunks[1].embedding = preserved_vector
        chunks[1].save(update_fields=['embedding'])

        calls = []

        def fake_iter(texts, batch_size):
            calls.append((list(texts), batch_size))
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                yield start, [build_vector(float(len(text))) for text in batch]

        iter_batches_mock.side_effect = fake_iter
        stdout = StringIO()

        call_command('generate_embeddings', batch_size=2, fetch_size=3, stdout=stdout)

        self.assertEqual(
            calls,
            [
                (
                    [
                        build_chunk_embedding_text('chunk-0', {}),
                        build_chunk_embedding_text('chunk-2', {}),
                        build_chunk_embedding_text('chunk-3', {}),
                    ],
                    2,
                ),
                ([build_chunk_embedding_text('chunk-4', {})], 2),
            ],
        )
        chunks[1].refresh_from_db()
        self.assertEqual(normalize_vector(chunks[1].embedding), preserved_vector)
        self.assertEqual(ContentChunk.objects.filter(embedding__isnull=True).count(), 0)
        self.assertIn('Embedded 4/4 chunks.', stdout.getvalue())

    @patch('scraper.management.commands.generate_embeddings.iter_text_embedding_batches')
    def test_generate_embeddings_rebuild_rewrites_existing_embeddings(
        self, iter_batches_mock
    ):
        chunks = [
            ContentChunk.objects.create(
                page=self.page,
                chunk_index=index,
                text=f'chunk-{index}',
                embedding=build_vector(float(index)),
            )
            for index in range(2)
        ]

        def fake_iter(texts, batch_size):
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                yield start, [build_vector(42.0) for _ in batch]

        iter_batches_mock.side_effect = fake_iter

        call_command('generate_embeddings', rebuild=True, batch_size=1, fetch_size=1)

        for chunk in chunks:
            chunk.refresh_from_db()
            self.assertEqual(normalize_vector(chunk.embedding), build_vector(42.0))


class ScrapeMainSiteCommandTests(SimpleTestCase):
    @patch('scraper.management.commands.scrape_main_site.crawl_candidate_data')
    @patch('scraper.management.commands.scrape_main_site.crawl_main_site')
    @patch('scraper.management.commands.scrape_main_site.build_session')
    def test_scrape_main_site_command_also_refreshes_candidate_data(
        self,
        build_session_mock,
        crawl_main_site_mock,
        crawl_candidate_data_mock,
    ):
        client = Mock()
        build_session_mock.return_value = client
        crawl_main_site_mock.return_value = {
            'seen': 12,
            'saved': 8,
            'updated': 5,
            'deactivated': 1,
            'failed': 0,
        }
        crawl_candidate_data_mock.return_value = {
            'seen': 6,
            'saved': 10,
            'updated': 7,
            'deactivated': 0,
            'failed': 0,
            'structured_saved': 4,
        }
        stdout = StringIO()

        call_command('scrape_main_site', max_pages=12, rate_limit_delay=0.25, stdout=stdout)

        build_session_mock.assert_called_once()
        crawl_main_site_mock.assert_called_once_with(
            client=client,
            seeds=list(DEFAULT_MAIN_SITE_SEEDS),
            max_pages=12,
            force_refresh=False,
            rate_limit_delay=0.25,
        )
        crawl_candidate_data_mock.assert_called_once_with(
            client=client,
            force_refresh=False,
            rate_limit_delay=0.25,
        )
        client.close.assert_called_once()
        output = stdout.getvalue()
        self.assertIn('candidate_seen=6', output)
        self.assertIn('candidate_structured_saved=4', output)
