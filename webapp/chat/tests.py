from io import StringIO
from unittest.mock import patch

from django.core.cache import cache
from django.core.management import call_command
from django.test import TestCase
from django.test.utils import override_settings
from django.urls import reverse

from scraper.models import ContentChunk, WebPage

from .models import Conversation, Message
from .services import (
    NO_CONTEXT_ANSWER,
    _extract_question_scope_hints,
    _retrieve_candidates,
    build_prompt,
    build_sources,
    cache_key,
    chat,
    chat_stream,
    retrieve_keyword_context,
)


@override_settings(
    CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
)
class ChatServiceTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_chat_endpoint_rejects_empty_question(self):
        response = self.client.post(
            reverse('chat_api'),
            data='{"question": ""}',
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)

    @patch('chat.api.chat')
    def test_chat_endpoint_returns_service_payload(self, chat_mock):
        chat_mock.return_value = {
            'answer': 'Burs detaylari [1]',
            'conversation_id': 3,
            'sources': [{'title': 'Tip Fakultesi', 'url': 'https://example.com'}],
            'cached': False,
        }

        response = self.client.post(
            reverse('chat_api'),
            data='{"question": "Burs var mi?"}',
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['conversation_id'], 3)
        chat_mock.assert_called_once()

    @patch('chat.api.chat_stream')
    def test_chat_stream_endpoint_returns_streaming_response(self, chat_stream_mock):
        chat_stream_mock.return_value = iter(
            [
                'event: meta\ndata: {"conversation_id": 3}\n\n',
                'data: [DONE]\n\n',
            ]
        )

        response = self.client.post(
            reverse('chat_stream_api'),
            data='{"question": "Burs var mi?"}',
            content_type='application/json',
        )

        body = ''.join(
            chunk.decode() if isinstance(chunk, bytes) else chunk
            for chunk in response.streaming_content
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Cache-Control'], 'no-cache')
        self.assertEqual(response['X-Accel-Buffering'], 'no')
        self.assertEqual(response['Content-Type'], 'text/event-stream')
        self.assertIn('event: meta', body)
        self.assertIn('data: [DONE]', body)
        chat_stream_mock.assert_called_once()

    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context', return_value=[])
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_no_context_answer(self, *_mocks):
        payload = chat('Yurt kapasitesi nedir?')

        self.assertEqual(payload['answer'], NO_CONTEXT_ANSWER)
        self.assertEqual(payload['sources'], [])
        self.assertEqual(Conversation.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 2)

    def test_retrieve_keyword_context_returns_matching_chunks(self):
        page = WebPage.objects.create(
            url='https://example.com/akademik-takvim',
            source='main_site',
            title='Akademik Takvim',
            content_text='Akademik takvim detaylari',
            raw_html='<main>Akademik takvim detaylari</main>',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Akademik takvim bahar donemi baslangic tarihleri burada yer alir.',
        )

        results = retrieve_keyword_context('Akademik takvim nerede?')

        self.assertEqual(results, [chunk])

    def test_retrieve_keyword_context_matches_program_metadata(self):
        page = WebPage.objects.create(
            url='https://example.com/bolum-baskani',
            source='bologna',
            title='Bölüm Başkanı',
            content_text='Yönetim bilgileri',
            raw_html='<main>Yönetim bilgileri</main>',
            content_hash='hash',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'section_title': 'Bölüm Başkanı',
            },
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Bu sayfada bölüm yönetimine ilişkin bilgi yer alır.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'section_title': 'Bölüm Başkanı',
            },
        )

        results = retrieve_keyword_context('Bilgisayar mühendisliği bölümünün başkanı kimdir?')

        self.assertEqual(results, [chunk])

    def test_cache_key_normalizes_case_whitespace_and_punctuation(self):
        key_a = cache_key('Psikoloji bölümü 5. yarıyıl ders planı?')
        key_b = cache_key('  psikoloji   bölümü 5 yarıyıl ders planı  ')

        self.assertEqual(key_a, key_b)
        self.assertTrue(key_a.startswith('chat-answer:v3:'))

    @override_settings(RAG_MAX_CHUNK_CHARS=20, RAG_MAX_CONTEXT_CHARS=140)
    def test_build_prompt_trims_context_and_limits_used_chunks(self):
        page = WebPage.objects.create(
            url='https://example.com/program',
            source='main_site',
            title='Program',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )
        chunk_a = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='A' * 60,
        )
        chunk_b = ContentChunk.objects.create(
            page=page,
            chunk_index=1,
            text='B' * 60,
        )

        prompt, used_chunks = build_prompt('Soru nedir?', [chunk_a, chunk_b])

        self.assertIn('Icerik: ' + ('A' * 19) + '…', prompt)
        self.assertNotIn('Icerik: ' + ('B' * 19) + '…', prompt)
        self.assertEqual(used_chunks, [chunk_a])

    def test_build_prompt_includes_program_and_faculty_metadata(self):
        page = WebPage.objects.create(
            url='https://example.com/bolum-baskani',
            source='bologna',
            title='Bölüm Başkanı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Başkanlık bilgisi burada yer alır.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'section_title': 'Bölüm Başkanı',
            },
        )

        prompt, _used_chunks = build_prompt('Bilgisayar mühendisliği bölümünün başkanı kimdir?', [chunk])

        self.assertIn('Program: Bilgisayar Mühendisliği (İngilizce)', prompt)
        self.assertIn('Fakulte: Mühendislik ve Doğa Bilimleri Fakültesi', prompt)
        self.assertIn('Bolum/Sayfa: Bölüm Başkanı', prompt)

    def test_build_prompt_prefers_metadata_source_url(self):
        page = WebPage.objects.create(
            url='https://example.com/internal-structured-url',
            source='structured',
            title='Bilgisayar Mühendisliği (%50 İndirimli) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Taban puan bilgisi burada yer alır.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği',
                'placement_label': 'Bilgisayar Mühendisliği (%50 İndirimli)',
                'section_title': 'Kontenjan ve Puan',
                'source_url': 'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
            },
        )

        prompt, _used_chunks = build_prompt('Bilgisayar mühendisliği %50 indirimli taban puanı nedir?', [chunk])

        self.assertIn(
            'URL: https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
            prompt,
        )

    def test_build_prompt_includes_admissions_guidance_for_score_queries(self):
        page = WebPage.objects.create(
            url='https://example.com/internal-structured-url',
            source='structured',
            title='Bilgisayar Mühendisliği (%50 İndirimli) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Taban puan 450, taban başarı sırası 30000.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'placement_label': 'Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
                'section_title': 'Kontenjan ve Puan',
            },
        )

        prompt, _used_chunks = build_prompt('Bilgisayar mühendisliği sıralaması kaç?', [chunk])

        self.assertIn('Kullanıcı sadece "sıralama" dediyse bunu "taban başarı sırası" olarak yorumla.', prompt)
        self.assertIn('Aynı program için birden fazla yerleşim tipi varsa', prompt)

    def test_build_sources_prefers_contextual_labels(self):
        page = WebPage.objects.create(
            url='https://example.com/bolum-baskani',
            source='bologna',
            title='Bölüm Başkanı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Başkanlık bilgisi burada yer alır.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'section_title': 'Bölüm Başkanı',
            },
        )

        sources = build_sources([chunk])

        self.assertEqual(sources[0]['label'], 'Bilgisayar Mühendisliği (İngilizce) / Bölüm Başkanı')

    def test_build_sources_prefers_metadata_source_url(self):
        page = WebPage.objects.create(
            url='https://example.com/internal-structured-url',
            source='structured',
            title='Burslu Kayıt',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Resmi structured kayıt.',
            metadata={
                'program_title': 'Tıp Fakültesi (İngilizce)',
                'placement_label': 'Tıp Fakültesi (İngilizce) (Burslu)',
                'section_title': 'Kontenjan ve Puan',
                'source_url': 'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
            },
        )

        sources = build_sources([chunk])

        self.assertEqual(
            sources[0]['label'],
            'Tıp Fakültesi (İngilizce) (Burslu) / Kontenjan ve Puan',
        )
        self.assertEqual(
            sources[0]['url'],
            'https://www.acibadem.edu.tr/aday/ogrenci/egitim/lisans/lisans-kontenjan-ve-puan-tablosu',
        )

    def test_extract_question_scope_hints_matches_staff_queries_with_suffixes(self):
        hints = _extract_question_scope_hints('Bilgisayar mühendisliğinde kaç hoca var?')

        self.assertIn(
            ('program_title', {'bilgisayar mühendisliği', 'mühendisliği'}),
            hints,
        )

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context')
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_prioritizes_bologna_staff_pages_for_staff_queries(
        self,
        retrieve_context_mock,
        retrieve_keyword_context_mock,
    ):
        bologna_page = WebPage.objects.create(
            url='https://example.com/academic-staff',
            source='bologna',
            title='Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )
        main_site_page = WebPage.objects.create(
            url='https://example.com/main-site-staff',
            source='main_site',
            title='Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-main',
        )
        generic_page = WebPage.objects.create(
            url='https://example.com/program-about',
            source='bologna',
            title='Program Hakkında',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-about',
        )
        generic_chunk = ContentChunk.objects.create(
            page=generic_page,
            chunk_index=0,
            text='Program hakkında metni.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği',
                'kind': 'bologna_program_page',
            },
        )
        main_site_chunk = ContentChunk.objects.create(
            page=main_site_page,
            chunk_index=0,
            text='Toplam hoca sayisi: 5',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği',
                'kind': 'main_site_staff_page',
            },
        )
        bologna_chunk = ContentChunk.objects.create(
            page=bologna_page,
            chunk_index=0,
            text='Toplam hoca sayisi: 6',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği',
                'kind': 'bologna_staff_page',
            },
        )
        retrieve_context_mock.return_value = [generic_chunk, main_site_chunk, bologna_chunk]
        retrieve_keyword_context_mock.return_value = []

        results = _retrieve_candidates('Bilgisayar mühendisliğinde kaç hoca var?', [0.1, 0.2])

        self.assertEqual(
            [(chunk.page.source, chunk.metadata['kind']) for chunk in results],
            [
                ('bologna', 'bologna_staff_page'),
                ('main_site', 'main_site_staff_page'),
                ('bologna', 'bologna_program_page'),
            ],
        )

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_prioritizes_structured_score_pages_for_score_queries(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        structured_page = WebPage.objects.create(
            url='https://example.com/structured-score',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-structured',
        )
        topic_page = WebPage.objects.create(
            url='https://example.com/score-topic',
            source='main_site',
            title='Lisans/Ön Lisans Kontenjan ve Puan Tablosu',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-topic',
        )
        generic_page = WebPage.objects.create(
            url='https://example.com/program-about',
            source='bologna',
            title='Program Hakkında',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-about',
        )
        generic_chunk = ContentChunk.objects.create(
            page=generic_page,
            chunk_index=0,
            text='Program hakkında metni.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'kind': 'bologna_program_page',
            },
        )
        topic_chunk = ContentChunk.objects.create(
            page=topic_page,
            chunk_index=0,
            text='Kontenjan ve puan tablosu sayfası.',
            metadata={
                'kind': 'candidate_topic_page',
                'topic': 'admissions_scores',
            },
        )
        structured_chunk = ContentChunk.objects.create(
            page=structured_page,
            chunk_index=0,
            text='Taban puan: 450',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'placement_label': 'Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
                'program_alias_text': 'Bilgisayar Mühendisliği (İngilizce) | Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
                'topic': 'admissions_scores',
            },
        )
        retrieve_context_mock.return_value = [generic_chunk, topic_chunk, structured_chunk]

        results = _retrieve_candidates(
            'Taban puan ve kontenjan bilgisi nerede?',
            [0.1, 0.2],
        )

        self.assertEqual(results, [structured_chunk, topic_chunk])

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_discards_non_admissions_sources_for_score_queries(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        generic_page = WebPage.objects.create(
            url='https://example.com/program-about',
            source='bologna',
            title='Program Hakkında',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-about',
        )
        generic_chunk = ContentChunk.objects.create(
            page=generic_page,
            chunk_index=0,
            text='Program hakkında metni.',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'kind': 'bologna_program_page',
            },
        )
        retrieve_context_mock.return_value = [generic_chunk]

        results = _retrieve_candidates('Bilgisayar mühendisliği sıralaması kaç?', [0.1, 0.2])

        self.assertEqual(results, [])

    @override_settings(RAG_RETRIEVE_LIMIT=2, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_falls_back_when_scope_filter_eliminates_all_results(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/staff',
            source='bologna',
            title='Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-staff',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Toplam hoca sayisi: 4',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği',
                'kind': 'bologna_staff_page',
            },
        )
        retrieve_context_mock.return_value = [chunk]

        results = _retrieve_candidates('Psikoloji hocalarını listele', [0.1, 0.2])

        self.assertEqual(results, [chunk])

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_no_context_when_scoped_program_has_no_matching_source(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/psikoloji-baskani',
            source='bologna',
            title='Bölüm Başkanı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )
        wrong_chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Psikoloji bölümü yönetim bilgisi burada yer alır.',
            metadata={
                'program_title': 'Psikoloji',
                'faculty': 'Fen Edebiyat Fakültesi',
                'section_title': 'Bölüm Başkanı',
            },
        )
        retrieve_context_mock.return_value = [wrong_chunk]

        payload = chat('Bilgisayar mühendisliği bölümünün başkanı kimdir?')

        self.assertEqual(payload['answer'], NO_CONTEXT_ANSWER)
        self.assertEqual(payload['sources'], [])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_structured_score_answer_without_llm(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        burslu_page = WebPage.objects.create(
            url='https://example.com/structured-score-burslu',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) (Burslu) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-burslu',
        )
        indirimli_page = WebPage.objects.create(
            url='https://example.com/structured-score-indirimli',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-indirimli',
        )
        burslu_chunk = ContentChunk.objects.create(
            page=burslu_page,
            chunk_index=0,
            text='Taban puan: 490, taban başarı sırası: 1200',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'program_alias_text': 'Bilgisayar Mühendisliği (İngilizce) | Bilgisayar Mühendisliği',
                'placement_label': 'Bilgisayar Mühendisliği (İngilizce) (Burslu)',
                'placement_type': 'Burslu',
                'section_title': 'Kontenjan ve Puan',
                'base_score': '490',
                'base_rank': '1200',
            },
        )
        indirimli_chunk = ContentChunk.objects.create(
            page=indirimli_page,
            chunk_index=0,
            text='Taban puan: 450, taban başarı sırası: 30000',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'program_alias_text': 'Bilgisayar Mühendisliği (İngilizce) | Bilgisayar Mühendisliği',
                'placement_label': 'Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
                'placement_type': '%50 İndirimli',
                'section_title': 'Kontenjan ve Puan',
                'base_score': '450',
                'base_rank': '30000',
            },
        )
        retrieve_context_mock.return_value = [indirimli_chunk, burslu_chunk]

        payload = chat('Bilgisayar mühendisliği sıralaması kaç?')

        self.assertIn('Bilgisayar Mühendisliği (İngilizce) için resmi aday öğrenci kaynağındaki bilgiler:', payload['answer'])
        self.assertIn('Bilgisayar Mühendisliği (İngilizce) (Burslu): taban puan 490, taban başarı sırası 1200.', payload['answer'])
        self.assertIn('Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli): taban puan 450, taban başarı sırası 30000.', payload['answer'])
        self.assertEqual(len(payload['sources']), 2)
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query')
    def test_chat_uses_cached_answer(
        self, embed_query_mock, retrieve_context_mock, generate_answer_mock
    ):
        cache.set(
            cache_key('Akademik takvim nerede?'),
            {'answer': 'Cache cevabi', 'sources': []},
            timeout=300,
        )

        payload = chat('Akademik takvim nerede?')

        self.assertEqual(payload['answer'], 'Cache cevabi')
        self.assertTrue(payload['cached'])
        embed_query_mock.assert_not_called()
        retrieve_context_mock.assert_not_called()
        generate_answer_mock.assert_not_called()

    def test_chat_stream_uses_cached_answer(self):
        cache.set(
            cache_key('Akademik takvim nerede?'),
            {
                'answer': 'Cache cevabi',
                'sources': [{'title': 'Takvim', 'url': 'https://example.com/takvim'}],
            },
            timeout=300,
        )

        payload = ''.join(chat_stream('Akademik takvim nerede?'))

        self.assertTrue(payload.startswith('event: meta'))
        self.assertEqual(payload.count('event: meta'), 2)
        self.assertIn('"cached": true', payload)
        self.assertIn('event: meta', payload)
        self.assertIn('event: token', payload)
        self.assertIn('Cache cevabi', payload)
        self.assertIn('event: sources', payload)
        self.assertIn('data: [DONE]', payload)
        self.assertEqual(Conversation.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 2)

    @patch('chat.services.generate_answer_stream', return_value=iter(['Parca 1', ' Parca 2 [1]']))
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_stream_yields_tokens_and_caches_answer(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_stream_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/burs',
            source='main_site',
            title='Burslar',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-burs',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Burs kapsamı hakkında resmi bilgi.',
        )
        retrieve_context_mock.return_value = [chunk]

        payload = ''.join(chat_stream('Burs kapsamı nedir?'))

        self.assertTrue(payload.startswith('event: meta'))
        self.assertEqual(payload.count('event: meta'), 1)
        self.assertEqual(payload.count('event: token'), 2)
        self.assertIn('Parca 1', payload)
        self.assertIn('event: sources', payload)
        self.assertIn('data: [DONE]', payload)
        self.assertEqual(
            cache.get(cache_key('Burs kapsamı nedir?'))['answer'],
            'Parca 1 Parca 2 [1]',
        )
        self.assertEqual(Message.objects.filter(role='assistant').get().content, 'Parca 1 Parca 2 [1]')
        generate_answer_stream_mock.assert_called_once()


class WarmModelsCommandTests(TestCase):
    @patch('chat.management.commands.warm_models.warm_llm_model')
    @patch('chat.management.commands.warm_models.warm_embedding_model')
    def test_warm_models_command_logs_success(self, warm_embedding_model_mock, warm_llm_model_mock):
        stdout = StringIO()

        call_command('warm_models', stdout=stdout)

        warm_embedding_model_mock.assert_called_once()
        warm_llm_model_mock.assert_called_once()
        output = stdout.getvalue()
        self.assertIn('Warmup completed for embedding_model.', output)
        self.assertIn('Warmup completed for llm_model.', output)

    @patch('chat.management.commands.warm_models.warm_llm_model')
    @patch('chat.management.commands.warm_models.warm_embedding_model')
    def test_warm_models_command_warns_without_failing(
        self, warm_embedding_model_mock, warm_llm_model_mock
    ):
        warm_llm_model_mock.side_effect = RuntimeError('runner unavailable')
        stdout = StringIO()

        call_command('warm_models', stdout=stdout)

        warm_embedding_model_mock.assert_called_once()
        warm_llm_model_mock.assert_called_once()
        self.assertIn('Warmup failed for llm_model: runner unavailable', stdout.getvalue())
