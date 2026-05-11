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
    LLM_BUSY_ANSWER,
    LLMBusyError,
    NO_CONTEXT_ANSWER,
    _extract_question_scope_hints,
    _expanded_query_terms,
    _filter_candidates_for_query,
    _is_dentistry_query,
    _is_staff_count_query,
    _is_staff_list_query,
    _is_staff_query,
    _question_topics,
    _resolve_question_with_conversation,
    _retrieve_candidates,
    _retrieve_direct_facility_chunks,
    _retrieve_direct_staff_chunks,
    build_prompt,
    build_sources,
    cache_key,
    chat,
    chat_stream,
    generate_answer,
    get_llm_client,
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

    @override_settings(
        LLM_BASE_URL='http://host.docker.internal:11434/v1',
        LLM_API_KEY='ollama-test-key',
        LLM_TIMEOUT=12,
    )
    @patch('chat.services.OpenAI')
    def test_get_llm_client_uses_configured_base_url(self, openai_mock):
        get_llm_client()

        openai_mock.assert_called_once_with(
            base_url='http://host.docker.internal:11434/v1',
            api_key='ollama-test-key',
            timeout=12,
        )

    @override_settings(
        LLM_BACKEND='ollama',
        LLM_BASE_URL='http://host.docker.internal:11434/v1',
        LLM_MODEL='qwen3:8b',
        LLM_MAX_TOKENS=512,
        LLM_TIMEOUT=12,
        LLM_THINK=True,
    )
    @patch('chat.services.requests.post')
    def test_generate_answer_uses_ollama_native_chat_api(self, post_mock):
        response_mock = post_mock.return_value
        response_mock.json.return_value = {
            'message': {
                'content': 'Bilgisayar Mühendisliği programı vardır. [1]',
                'thinking': 'hidden reasoning',
            }
        }

        answer = generate_answer('Bilgisayar mühendisliği var mı?')

        self.assertEqual(answer, 'Bilgisayar Mühendisliği programı vardır. [1]')
        post_mock.assert_called_once()
        self.assertEqual(
            post_mock.call_args.args[0],
            'http://host.docker.internal:11434/api/chat',
        )
        body = post_mock.call_args.kwargs['json']
        self.assertEqual(body['model'], 'qwen3:8b')
        self.assertEqual(body['think'], True)
        self.assertEqual(body['options']['num_predict'], 512)
        response_mock.raise_for_status.assert_called_once()

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
        self.assertTrue(key_a.startswith('chat-answer:v15:'))

    @override_settings(RAG_MAX_CHUNK_CHARS=20, RAG_MAX_CONTEXT_CHARS=155)
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

    @override_settings(RAG_MAX_CHUNK_CHARS=30, RAG_MAX_CONTEXT_CHARS=190)
    def test_build_prompt_expands_context_for_facility_queries(self):
        page = WebPage.objects.create(
            url='https://example.com/kutuphane',
            source='main_site',
            title='Kütüphane',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-library',
        )
        chunk_a = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Kütüphane koleksiyonu ve çalışma alanları hakkında bilgi.',
        )
        chunk_b = ContentChunk.objects.create(
            page=page,
            chunk_index=1,
            text='Kütüphane veri tabanları ve danışma hizmetleri hakkında bilgi.',
        )

        _normal_prompt, normal_chunks = build_prompt('Genel soru nedir?', [chunk_a, chunk_b])
        _facility_prompt, facility_chunks = build_prompt('Kütüphane hakkında bilgi verir misin?', [chunk_a, chunk_b])

        self.assertEqual(normal_chunks, [chunk_a])
        self.assertEqual(facility_chunks, [chunk_a, chunk_b])

    def test_query_expansion_applies_only_to_topic_queries(self):
        self.assertIn('library', _expanded_query_terms('kütüphane hakkında bilgi'))
        self.assertIn('veritabanı', _expanded_query_terms('kütüphane hakkında bilgi'))
        self.assertIn('fitness', _expanded_query_terms('spor merkezi nerede?'))

        self.assertEqual(_expanded_query_terms('pc müh ücreti'), set())
        self.assertEqual(_expanded_query_terms('pc müh hocaları'), set())
        self.assertEqual(_expanded_query_terms('bilgisayar mühendisliği taban puan'), set())
        self.assertEqual(_expanded_query_terms('bilgisayar mühendisliği ders planı'), set())

    def test_direct_facility_retrieval_uses_expanded_library_terms(self):
        page = WebPage.objects.create(
            url='https://example.com/bilgi-merkezi',
            source='main_site',
            title='Bilgi Merkezi',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-expanded-library',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Veritabanı erişimi ve çalışma alanı hizmetleri resmi kaynakta anlatılır.',
            metadata={
                'kind': 'main_site_page',
                'topic': 'library',
                'topic_label': 'Kütüphane',
            },
        )

        results = _retrieve_direct_facility_chunks('kütüphane hakkında bilgi', limit=5)

        self.assertEqual(results, [chunk])

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

    def test_build_prompt_includes_curriculum_metadata(self):
        page = WebPage.objects.create(
            url='https://example.com/semester-plan',
            source='bologna',
            title='3. Yarıyıl Ders Planı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='CSE 201 Veri Yapıları | AKTS: 6',
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'curriculum_year': '2025',
                'period_label': '3. Yarıyıl Ders Planı',
                'section_title': '3. Yarıyıl Ders Planı',
            },
        )

        prompt, _used_chunks = build_prompt('Hemşirelik 3. yarıyıl dersler', [chunk])

        self.assertIn('Mufredat Yili: 2025', prompt)
        self.assertIn('Donem: 3. Yarıyıl Ders Planı', prompt)

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

    def test_pc_muh_staff_query_flags_and_direct_chunks(self):
        staff_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-muhendisligi-akademik-kadro',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-pc-muh-staff',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Ahmet Bulut | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'unit_name': 'Bilgisayar Mühendisliği',
                'entity_name': 'Ahmet Bulut',
                'staff_title': 'Prof. Dr.',
                'staff_count': 1,
                'section_title': 'Akademik Kadro',
            },
        )

        question = 'pc müh hocaları kaç tane ve isimleri'
        chunks = _retrieve_direct_staff_chunks(question, limit=5)

        self.assertTrue(_is_staff_query(question))
        self.assertTrue(_is_staff_count_query(question))
        self.assertTrue(_is_staff_list_query(question))
        self.assertEqual(chunks, [staff_chunk])

    def test_dentistry_query_matches_common_turkish_terms(self):
        self.assertTrue(_is_dentistry_query('Acıbadem Üniversitesinde dişçilik var mı?'))
        self.assertTrue(_is_dentistry_query('Diş hekimliği bölümü var mı?'))

    def test_followup_question_uses_previous_program_context(self):
        conversation = Conversation.objects.create(title='Bilgisayar Mühendisliği')
        Message.objects.create(
            conversation=conversation,
            role='user',
            content='Bilgisayar mühendisliği hocaları kimler?',
        )

        resolved = _resolve_question_with_conversation('kaç tane hocası var', conversation)

        self.assertEqual(resolved, 'bilgisayar mühendisliği kaç tane hocası var')

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
            ],
        )

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_prioritizes_main_site_staff_and_head_chunks_for_staff_queries(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        staff_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-muhendisligi-akademik-kadro',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-staff',
        )
        head_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-muhendisligi-bolum-baskani',
            source='main_site',
            title='Bilgisayar Mühendisliği - Bölüm Başkanının Mesajı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-head',
        )
        score_page = WebPage.objects.create(
            url='https://example.com/structured-score',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-score',
        )
        score_chunk = ContentChunk.objects.create(
            page=score_page,
            chunk_index=0,
            text='Taban puan 450, taban başarı sırası 30000.',
            metadata={
                'kind': 'structured_admissions_score',
                'record_type': 'quota_row',
                'program_title': 'Bilgisayar Mühendisliği',
                'placement_label': 'Bilgisayar Mühendisliği (İngilizce) (%50 İndirimli)',
            },
        )
        head_chunk = ContentChunk.objects.create(
            page=head_page,
            chunk_index=0,
            text='Bölüm Başkanının Mesajı / Bölüm Başkanı - Prof. Dr. Ahmet Bulut / Sevgili öğrenciler',
            metadata={
                'record_type': 'department_head_message',
                'program_title': 'Bilgisayar Mühendisliği',
                'entity_name': 'Bilgisayar Mühendisliği',
                'section_title': 'Bölüm Başkanının Mesajı',
            },
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Ahmet Bulut | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'unit_name': 'Bilgisayar Mühendisliği',
                'entity_name': 'Ahmet Bulut',
                'section_title': 'Akademik Kadro',
            },
        )
        retrieve_context_mock.return_value = [score_chunk, head_chunk, staff_chunk]

        results = _retrieve_candidates('Bilgisayar mühendisliği hocaları kimler?', [0.1, 0.2])

        self.assertEqual(results, [staff_chunk, head_chunk])

    @override_settings(RAG_RETRIEVE_LIMIT=2, RAG_PER_PAGE_LIMIT=2, RAG_RRF_K=60)
    @patch('chat.services.retrieve_keyword_context')
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_uses_rrf_across_vector_and_keyword_results(
        self,
        retrieve_context_mock,
        retrieve_keyword_context_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/kampus',
            source='main_site',
            title='Kampüs Olanakları',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-rrf-page',
        )
        vector_only_chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Vektör aramada ilk gelen genel kampüs bilgisi.',
        )
        shared_chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=1,
            text='Hem vektör hem keyword aramada üst sırada gelen kampüs bilgisi.',
        )
        retrieve_context_mock.return_value = [vector_only_chunk, shared_chunk]
        retrieve_keyword_context_mock.return_value = [shared_chunk]

        results = _retrieve_candidates('kampüs hakkında bilgi', [0.1, 0.2])

        self.assertEqual(results[:2], [shared_chunk, vector_only_chunk])

    @override_settings(RAG_RETRIEVE_LIMIT=2, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_keeps_protected_direct_staff_before_vector_noise(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        staff_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-akademik-kadro',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-protected-direct-staff',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Ahmet Bulut | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'unit_name': 'Bilgisayar Mühendisliği',
                'entity_name': 'Ahmet Bulut',
                'section_title': 'Akademik Kadro',
            },
        )
        noise_page = WebPage.objects.create(
            url='https://example.com/psikoloji-akademik-kadro',
            source='main_site',
            title='Psikoloji - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-vector-noise-staff',
        )
        noise_chunk = ContentChunk.objects.create(
            page=noise_page,
            chunk_index=0,
            text='Psikoloji akademik kadro | isim: Farklı Hoca | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Psikoloji',
                'unit_name': 'Psikoloji',
                'entity_name': 'Farklı Hoca',
                'section_title': 'Akademik Kadro',
            },
        )
        retrieve_context_mock.return_value = [noise_chunk]

        results = _retrieve_candidates('pc müh hocaları', [0.1, 0.2])

        self.assertEqual(results[0], staff_chunk)

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

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_discards_non_topic_sources_for_erasmus_queries(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        score_page = WebPage.objects.create(
            url='https://example.com/score',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-erasmus-score',
        )
        erasmus_page = WebPage.objects.create(
            url='https://example.com/erasmus',
            source='main_site',
            title='Erasmus Öğrenci Hareketliliği',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-erasmus-topic',
        )
        generic_page = WebPage.objects.create(
            url='https://example.com/program-about',
            source='bologna',
            title='Program Hakkında',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-erasmus-about',
        )
        score_chunk = ContentChunk.objects.create(
            page=score_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği kontenjan ve puan bilgileri.',
            metadata={
                'kind': 'structured_admissions_score',
                'record_type': 'quota_row',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'topic': 'admissions_scores',
            },
        )
        erasmus_chunk = ContentChunk.objects.create(
            page=erasmus_page,
            chunk_index=0,
            text='Erasmus öğrenci hareketliliği başvuru koşulları ve süreçleri.',
            metadata={
                'kind': 'candidate_topic_page',
                'topic': 'international',
                'topic_label': 'Uluslararası Olanaklar',
            },
        )
        generic_chunk = ContentChunk.objects.create(
            page=generic_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği program tanıtımı.',
            metadata={
                'kind': 'bologna_program_page',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
            },
        )
        retrieve_context_mock.return_value = [score_chunk, generic_chunk, erasmus_chunk]

        results = _retrieve_candidates(
            'erasmus bilgisi bilgisayar mühendisliği',
            [0.1, 0.2],
        )

        self.assertEqual(results, [erasmus_chunk])

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

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    def test_retrieve_candidates_prioritizes_semester_plan_for_course_queries(
        self,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        overview_page = WebPage.objects.create(
            url='https://example.com/overview',
            source='bologna',
            title='Program Özeti',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-overview',
        )
        semester_page = WebPage.objects.create(
            url='https://example.com/semester-3',
            source='bologna',
            title='3. Yarıyıl Ders Planı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-semester',
        )
        overview_chunk = ContentChunk.objects.create(
            page=overview_page,
            chunk_index=0,
            text='Toplam AKTS: 240',
            metadata={
                'program_title': 'Hemşirelik',
                'record_type': 'bologna_program_overview',
                'chunk_level': 'program_overview',
            },
        )
        semester_chunk = ContentChunk.objects.create(
            page=semester_page,
            chunk_index=0,
            text='HEM 301 İç Hastalıkları Hemşireliği | AKTS: 6',
            metadata={
                'program_title': 'Hemşirelik',
                'record_type': 'bologna_semester_plan',
                'chunk_level': 'semester_plan',
                'period_label': '3. Yarıyıl Ders Planı',
            },
        )
        retrieve_context_mock.return_value = [overview_chunk, semester_chunk]

        results = _retrieve_candidates('Hemşirelik 3. yarıyıl dersler', [0.1, 0.2])

        self.assertEqual(set(results), {semester_chunk, overview_chunk})

    @override_settings(RAG_RETRIEVE_LIMIT=3, RAG_PER_PAGE_LIMIT=2)
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context', return_value=[])
    def test_retrieve_candidates_directly_loads_course_chunks_for_program_queries(
        self,
        _retrieve_context_mock,
        _retrieve_keyword_context_mock,
    ):
        overview_page = WebPage.objects.create(
            url='https://example.com/mbg-overview',
            source='bologna',
            title='Moleküler Biyoloji ve Genetik (İngilizce) - Program Özeti',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-mbg-overview',
        )
        semester_page = WebPage.objects.create(
            url='https://example.com/mbg-semester-1',
            source='bologna',
            title='Moleküler Biyoloji ve Genetik (İngilizce) - 1.Yarıyıl Ders Planı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-mbg-semester-1',
        )
        overview_chunk = ContentChunk.objects.create(
            page=overview_page,
            chunk_index=0,
            text='Program: Moleküler Biyoloji ve Genetik (İngilizce)\nToplam Ders Sayısı: 44',
            metadata={
                'program_title': 'Moleküler Biyoloji ve Genetik (İngilizce)',
                'record_type': 'bologna_program_overview',
                'chunk_level': 'program_overview',
                'curriculum_year': '2025',
                'course_count': 44,
            },
        )
        semester_chunk = ContentChunk.objects.create(
            page=semester_page,
            chunk_index=0,
            text='- MBG 101 Moleküler Biyolojiye Giriş | AKTS: 6',
            metadata={
                'program_title': 'Moleküler Biyoloji ve Genetik (İngilizce)',
                'record_type': 'bologna_semester_plan',
                'chunk_level': 'semester_plan',
                'curriculum_year': '2025',
                'period_number': 1,
                'period_label': '1.Yarıyıl Ders Planı',
            },
        )

        results = _retrieve_candidates('mbg dersleri', [0.1, 0.2])

        self.assertEqual(results, [overview_chunk, semester_chunk])

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context', return_value=[])
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_structured_course_summary_without_llm(
        self,
        _embed_query_mock,
        _retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        overview_page = WebPage.objects.create(
            url='https://example.com/cse-overview',
            source='bologna',
            title='Bilgisayar Mühendisliği (İngilizce) - Program Özeti',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-cse-overview',
        )
        ContentChunk.objects.create(
            page=overview_page,
            chunk_index=0,
            text=(
                'Program: Bilgisayar Mühendisliği (İngilizce)\n'
                'Müfredat Yılı: 2025\n'
                'Dönem Sayısı: 8\n'
                'Toplam Ders Sayısı: 46\n'
                'Toplam AKTS: 251\n'
                '- 1.Yarıyıl Ders Planı: 31 AKTS\n'
                '- 2.Yarıyıl Ders Planı: 31 AKTS'
            ),
            metadata={
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'program_alias_text': 'Bilgisayar Mühendisliği',
                'record_type': 'bologna_program_overview',
                'chunk_level': 'program_overview',
                'curriculum_year': '2025',
                'period_count': 8,
                'course_count': 46,
                'total_ects_sum': 251,
            },
        )

        payload = chat('bilgisayar mühendisliği dersleri')

        self.assertIn(
            'Bilgisayar Mühendisliği (İngilizce) 2025 müfredat özeti:',
            payload['answer'],
        )
        self.assertIn('- Dönem sayısı: 8', payload['answer'])
        self.assertIn('- Toplam ders sayısı: 46', payload['answer'])
        self.assertIn('- Toplam AKTS: 251', payload['answer'])
        self.assertIn('- 1.Yarıyıl Ders Planı: 31 AKTS', payload['answer'])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context', return_value=[])
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_structured_course_semester_without_llm(
        self,
        _embed_query_mock,
        _retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        semester_page = WebPage.objects.create(
            url='https://example.com/hemsirelik-semester-3',
            source='bologna',
            title='Hemşirelik - 3.Yarıyıl Ders Planı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-hemsirelik-semester-3',
        )
        ContentChunk.objects.create(
            page=semester_page,
            chunk_index=0,
            text=(
                'Program: Hemşirelik\n'
                'Dönem: 3.Yarıyıl Ders Planı\n'
                '- HEM 301 İç Hastalıkları Hemşireliği | AKTS: 6 | Saat: 3+2+0 | Zorunlu\n'
                '- HEM 303 Cerrahi Hastalıkları Hemşireliği | AKTS: 6 | Saat: 3+2+0 | Zorunlu'
            ),
            metadata={
                'program_title': 'Hemşirelik',
                'record_type': 'bologna_semester_plan',
                'chunk_level': 'semester_plan',
                'curriculum_year': '2025',
                'period_number': 3,
                'period_label': '3.Yarıyıl Ders Planı',
            },
        )

        payload = chat('hemşirelik 3. yarıyıl dersleri')

        self.assertIn('Hemşirelik 2025 müfredatında 3.Yarıyıl Ders Planı dersleri:', payload['answer'])
        self.assertIn('HEM 301 İç Hastalıkları Hemşireliği', payload['answer'])
        self.assertIn('HEM 303 Cerrahi Hastalıkları Hemşireliği', payload['answer'])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer', return_value='LLM cevabı')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_calls_llm_when_scope_mismatch_but_chunks_exist(
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

        generate_answer_mock.assert_called_once()
        self.assertEqual(payload['answer'], 'LLM cevabı')

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
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_structured_fee_answer_without_llm(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        fee_page = WebPage.objects.create(
            url='https://example.com/structured-fee',
            source='structured',
            title='Bilgisayar Mühendisliği - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-fee',
        )
        fee_chunk = ContentChunk.objects.create(
            page=fee_page,
            chunk_index=0,
            text='Ücretli: 900.000₺, %50 İndirimli: 450.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'fee_full': '900.000₺',
                'fee_50': '450.000₺',
                'notes': 'Ek burs seçeneği bulunmaktadır.',
            },
        )
        retrieve_context_mock.return_value = [fee_chunk]

        payload = chat('Bilgisayar mühendisliği ücreti ne kadar?')

        self.assertIn('Bilgisayar Mühendisliği (İngilizce) için resmi öğrenim ücreti bilgileri:', payload['answer'])
        self.assertIn('ücretli 900.000₺', payload['answer'])
        self.assertIn('%50 indirimli 450.000₺', payload['answer'])
        self.assertEqual(len(payload['sources']), 1)
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_clean_structured_fee_names(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        fee_page = WebPage.objects.create(
            url='https://example.com/structured-fee-clean',
            source='structured',
            title='Bilgisayar Mühendisliği - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-fee-clean',
        )
        fee_chunk = ContentChunk.objects.create(
            page=fee_page,
            chunk_index=0,
            text='Ücretli: 675.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)**',
                'fee_full': '675.000₺',
                'notes': 'Burs hakkında bilgi almak için tıklayın. Ek destek vardır.',
            },
        )
        retrieve_context_mock.return_value = [fee_chunk]

        payload = chat('Bilgisayar mühendisliği ücreti')

        self.assertIn('Bilgisayar Mühendisliği (İngilizce) için', payload['answer'])
        self.assertNotIn('**', payload['answer'])
        self.assertNotIn('tıklayın', payload['answer'])
        self.assertIn('Ek destek vardır.', payload['answer'])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_resolves_pc_muh_fee_to_computer_engineering(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        engineering_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-muhendisligi-fee',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce)** - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-pc-muh-fee',
        )
        engineering_chunk = ContentChunk.objects.create(
            page=engineering_page,
            chunk_index=0,
            text='Ücretli: 675.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)**',
                'fee_full': '675.000₺',
                'fee_25': '506.250₺',
            },
        )
        programming_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-programciligi-fee',
            source='structured',
            title='Bilgisayar Programcılığı - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-pc-programming-fee',
        )
        programming_chunk = ContentChunk.objects.create(
            page=programming_page,
            chunk_index=0,
            text='Ücretli: 225.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Bilgisayar Programcılığı',
                'fee_full': '225.000₺',
            },
        )
        sociology_page = WebPage.objects.create(
            url='https://example.com/sosyoloji-fee',
            source='structured',
            title='Sosyoloji - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-sosyoloji-fee',
        )
        sociology_chunk = ContentChunk.objects.create(
            page=sociology_page,
            chunk_index=0,
            text='Ücretli: 365.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Sosyoloji',
                'fee_full': '365.000₺',
            },
        )
        retrieve_context_mock.return_value = [sociology_chunk, programming_chunk]

        payload = chat('pc müh ücreti')

        self.assertIn('Bilgisayar Mühendisliği (İngilizce) için resmi öğrenim ücreti bilgileri:', payload['answer'])
        self.assertIn('ücretli 675.000₺', payload['answer'])
        self.assertNotIn('Sosyoloji', payload['answer'])
        self.assertNotIn('Bilgisayar Programcılığı', payload['answer'])
        self.assertEqual(len(payload['sources']), 1)
        self.assertEqual(payload['sources'][0]['title'], engineering_page.title)
        generate_answer_mock.assert_not_called()

        cache.clear()
        retrieve_context_mock.return_value = [programming_chunk, sociology_chunk]

        payload = chat('pc müh bölümü ücreti')

        self.assertIn('Bilgisayar Mühendisliği (İngilizce) için resmi öğrenim ücreti bilgileri:', payload['answer'])
        self.assertNotIn('Sosyoloji', payload['answer'])
        self.assertNotIn('Bilgisayar Programcılığı', payload['answer'])
        self.assertEqual(payload['sources'][0]['title'], engineering_page.title)

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_resolves_program_initialism_fee(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        mbg_page = WebPage.objects.create(
            url='https://example.com/mbg-fee',
            source='structured',
            title='Moleküler Biyoloji ve Genetik (İngilizce)** - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-mbg-fee',
        )
        mbg_chunk = ContentChunk.objects.create(
            page=mbg_page,
            chunk_index=0,
            text='Ücretli: 675.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Moleküler Biyoloji ve Genetik (İngilizce)**',
                'fee_full': '675.000₺',
                'fee_25': '506.250₺',
            },
        )
        mbg_score_page = WebPage.objects.create(
            url='https://example.com/mbg-score',
            source='structured',
            title='Moleküler Biyoloji ve Genetik (İngilizce) (Burslu) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-mbg-score',
        )
        ContentChunk.objects.create(
            page=mbg_score_page,
            chunk_index=0,
            text='Taban puan: 500',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Moleküler Biyoloji ve Genetik (İngilizce)',
                'placement_label': 'Moleküler Biyoloji ve Genetik (İngilizce) (Burslu)',
            },
        )
        ContentChunk.objects.create(
            page=mbg_score_page,
            chunk_index=1,
            text='Taban puan: 450',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Moleküler Biyoloji ve Genetik (İngilizce)',
                'placement_label': 'Moleküler Biyoloji ve Genetik (İngilizce) (%50 İndirimli)',
            },
        )
        sociology_page = WebPage.objects.create(
            url='https://example.com/initialism-sosyoloji-fee',
            source='structured',
            title='Sosyoloji - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-initialism-sosyoloji-fee',
        )
        sociology_chunk = ContentChunk.objects.create(
            page=sociology_page,
            chunk_index=0,
            text='Ücretli: 365.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Sosyoloji',
                'fee_full': '365.000₺',
            },
        )
        retrieve_context_mock.return_value = [sociology_chunk]

        payload = chat('mbg ücreti')

        self.assertIn('Moleküler Biyoloji ve Genetik (İngilizce) için resmi öğrenim ücreti bilgileri:', payload['answer'])
        self.assertIn('ücretli 675.000₺', payload['answer'])
        self.assertNotIn('Sosyoloji', payload['answer'])
        self.assertEqual(payload['sources'][0]['title'], mbg_page.title)
        generate_answer_mock.assert_not_called()

        cache.clear()
        retrieve_context_mock.return_value = [mbg_chunk]

        payload = chat('Moleküler Biyoloji ve Genetik ücreti')

        self.assertIn('Moleküler Biyoloji ve Genetik (İngilizce) için resmi öğrenim ücreti bilgileri:', payload['answer'])
        self.assertNotIn('Sosyoloji', payload['answer'])
        self.assertEqual(payload['sources'][0]['title'], mbg_page.title)

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_keeps_bilgisayar_programciligi_fee(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        engineering_page = WebPage.objects.create(
            url='https://example.com/programcilik-guard-engineering-fee',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce)** - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-programcilik-guard-engineering-fee',
        )
        engineering_chunk = ContentChunk.objects.create(
            page=engineering_page,
            chunk_index=0,
            text='Ücretli: 675.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)**',
                'fee_full': '675.000₺',
            },
        )
        programming_page = WebPage.objects.create(
            url='https://example.com/programcilik-guard-fee',
            source='structured',
            title='Bilgisayar Programcılığı - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-programcilik-guard-fee',
        )
        programming_chunk = ContentChunk.objects.create(
            page=programming_page,
            chunk_index=0,
            text='Ücretli: 225.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Bilgisayar Programcılığı',
                'fee_full': '225.000₺',
            },
        )
        retrieve_context_mock.return_value = [engineering_chunk]

        payload = chat('Bilgisayar Programcılığı ücreti')

        self.assertIn('Bilgisayar Programcılığı için resmi öğrenim ücreti bilgileri:', payload['answer'])
        self.assertIn('ücretli 225.000₺', payload['answer'])
        self.assertNotIn('Bilgisayar Mühendisliği', payload['answer'])
        self.assertEqual(payload['sources'][0]['title'], programming_page.title)
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_fee_sources_follow_answer_order_and_skip_empty_fee_chunks(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        programming_page = WebPage.objects.create(
            url='https://example.com/source-order-programming-fee',
            source='structured',
            title='Bilgisayar Programcılığı - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-source-order-programming-fee',
        )
        programming_chunk = ContentChunk.objects.create(
            page=programming_page,
            chunk_index=0,
            text='Ücretli: 225.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Bilgisayar Programcılığı',
                'fee_full': '225.000₺',
            },
        )
        engineering_page = WebPage.objects.create(
            url='https://example.com/source-order-engineering-fee',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce)** - Öğrenim Ücreti',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-source-order-engineering-fee',
        )
        engineering_chunk = ContentChunk.objects.create(
            page=engineering_page,
            chunk_index=0,
            text='Ücretli: 675.000₺',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)**',
                'fee_full': '675.000₺',
            },
        )
        generic_page = WebPage.objects.create(
            url='https://example.com/generic-fee-page',
            source='main_site',
            title='Lisans/Ön Lisans Öğrenim Ücretleri 2025-2026',
            content_text='Genel ücret sayfası',
            raw_html='{}',
            content_hash='hash-generic-fee-page',
        )
        generic_chunk = ContentChunk.objects.create(
            page=generic_page,
            chunk_index=0,
            text='Lisans/Ön Lisans Öğrenim Ücretleri',
            metadata={
                'kind': 'structured_admissions_fee',
                'record_type': 'tuition_fee',
                'program_title': '',
            },
        )
        retrieve_context_mock.return_value = [programming_chunk, engineering_chunk, generic_chunk]

        payload = chat('öğrenim ücreti bilgileri')

        self.assertLess(
            payload['answer'].find('- Bilgisayar Mühendisliği'),
            payload['answer'].find('- Bilgisayar Programcılığı'),
        )
        self.assertNotIn('ücret bilgisi kaynakta yer almıyor', payload['answer'])
        self.assertEqual(
            [source['title'] for source in payload['sources']],
            [engineering_page.title, programming_page.title],
        )
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_structured_staff_count_without_llm(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        staff_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-akademik-kadro',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-staff',
            metadata={
                'kind': 'main_site_staff_page',
                'program_title': 'Bilgisayar Mühendisliği',
                'staff_count': 3,
            },
        )
        first_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Ahmet Bulut | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'entity_name': 'Ahmet Bulut',
                'staff_title': 'Prof. Dr.',
                'staff_count': 3,
            },
        )
        ContentChunk.objects.create(
            page=staff_page,
            chunk_index=1,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Seda Nilgün Dumlu | unvan: Öğr. Gör. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'entity_name': 'Seda Nilgün Dumlu',
                'staff_title': 'Öğr. Gör. Dr.',
                'staff_count': 3,
            },
        )
        ContentChunk.objects.create(
            page=staff_page,
            chunk_index=2,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Seher Sonkaya | unvan: Arş. Gör.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'entity_name': 'Seher Sonkaya',
                'staff_title': 'Arş. Gör.',
                'staff_count': 3,
            },
        )
        retrieve_context_mock.return_value = [first_chunk]

        payload = chat('Bilgisayar mühendisliğinin kaç tane hocası var')

        self.assertEqual(
            payload['answer'],
            'Bilgisayar Mühendisliği akademik kadro kaynağında 3 hoca kaydı var.',
        )
        self.assertEqual(len(payload['sources']), 1)
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_resolves_pc_muh_staff_count_and_names_without_llm(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        staff_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-akademik-kadro',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-pc-muh-chat-staff',
            metadata={
                'kind': 'main_site_staff_page',
                'program_title': 'Bilgisayar Mühendisliği',
                'staff_count': 3,
            },
        )
        for index, (name, title) in enumerate(
            (
                ('Ahmet Bulut', 'Prof. Dr.'),
                ('Seda Nilgün Dumlu', 'Öğr. Gör. Dr.'),
                ('Seher Sonkaya', 'Arş. Gör.'),
            )
        ):
            ContentChunk.objects.create(
                page=staff_page,
                chunk_index=index,
                text=f'Bilgisayar Mühendisliği akademik kadro | isim: {name} | unvan: {title}',
                metadata={
                    'kind': 'main_site_staff_page',
                    'record_type': 'academic_staff_member',
                    'program_title': 'Bilgisayar Mühendisliği',
                    'unit_name': 'Bilgisayar Mühendisliği',
                    'entity_name': name,
                    'staff_title': title,
                    'staff_count': 3,
                    'section_title': 'Akademik Kadro',
                },
            )
        score_page = WebPage.objects.create(
            url='https://example.com/structured-score',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-pc-muh-chat-score',
        )
        score_chunk = ContentChunk.objects.create(
            page=score_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği kontenjan ve puan bilgisi.',
            metadata={
                'kind': 'structured_admissions_score',
                'record_type': 'quota_row',
                'program_title': 'Bilgisayar Mühendisliği',
                'section_title': 'Kontenjan ve Puan',
            },
        )
        retrieve_context_mock.return_value = [score_chunk]

        payload = chat('pc müh hocaları kaç tane ve isimleri')

        self.assertIn('Bilgisayar Mühendisliği akademik kadro kaynağında 3 hoca kaydı var:', payload['answer'])
        self.assertIn('- Prof. Dr. Ahmet Bulut', payload['answer'])
        self.assertIn('- Öğr. Gör. Dr. Seda Nilgün Dumlu', payload['answer'])
        self.assertIn('- Arş. Gör. Seher Sonkaya', payload['answer'])
        self.assertTrue(payload['sources'])
        self.assertTrue(all('Akademik Kadro' in source['title'] for source in payload['sources']))
        self.assertTrue(all('Kontenjan' not in source['title'] for source in payload['sources']))
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_resolves_followup_staff_count_from_conversation(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        conversation = Conversation.objects.create(title='Bilgisayar Mühendisliği')
        Message.objects.create(
            conversation=conversation,
            role='user',
            content='Bilgisayar mühendisliği hocaları kimler?',
        )
        staff_page = WebPage.objects.create(
            url='https://example.com/followup-staff',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-followup-staff',
            metadata={'kind': 'main_site_staff_page', 'program_title': 'Bilgisayar Mühendisliği'},
        )
        chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Ahmet Bulut | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'entity_name': 'Ahmet Bulut',
                'staff_title': 'Prof. Dr.',
                'staff_count': 1,
            },
        )
        retrieve_context_mock.return_value = [chunk]

        payload = chat('kaç tane hocası var', conversation_id=conversation.id)

        self.assertEqual(
            payload['answer'],
            'Bilgisayar Mühendisliği akademik kadro kaynağında 1 hoca kaydı var.',
        )
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_program_presence_from_placement_label_without_llm(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/bilgisayar-muhendisligi',
            source='structured',
            title='Bilgisayar Mühendisliği (İngilizce) (Burslu) - Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-program-presence',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği (İngilizce) kontenjan ve puan bilgileri',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Bilgisayar Mühendisliği (İngilizce)',
                'placement_label': 'Bilgisayar Mühendisliği (İngilizce) (Burslu)',
                'faculty': 'Mühendislik ve Doğa Bilimleri Fakültesi',
                'admission_level': 'lisans',
            },
        )
        retrieve_context_mock.return_value = [chunk]

        payload = chat('bilgisayar mühendisliği var mı acaba?')

        self.assertIn('Bilgisayar Mühendisliği (İngilizce) programı var', payload['answer'])
        self.assertIn('Mühendislik ve Doğa Bilimleri Fakültesi', payload['answer'])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_prioritizes_department_page_for_general_faculty_info(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        staff_page = WebPage.objects.create(
            url='https://example.com/eczacilik-akademik-kadro',
            source='main_site',
            title='Eczacılık Fakültesi - Akademik Kadro',
            content_text='staff',
            raw_html='{}',
            content_hash='hash-ecz-staff-info',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Eczacılık Fakültesi akademik kadro | isim: Ayşe Yılmaz | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Eczacılık Fakültesi',
                'section_title': 'Eczacılık Fakültesi - Akademik Kadro',
                'source_group': 'department',
            },
        )
        department_page = WebPage.objects.create(
            url='https://example.com/eczacilik-fakultesi',
            source='main_site',
            title='Eczacılık Fakültesi',
            content_text='Eczacılık Fakültesi hakkında genel bilgi.',
            raw_html='{}',
            content_hash='hash-ecz-department-info',
        )
        department_chunk = ContentChunk.objects.create(
            page=department_page,
            chunk_index=0,
            text='Eczacılık Fakültesi Hakkında misyon, eğitim ve laboratuvar bilgileri.',
            metadata={
                'kind': 'main_site_page',
                'section_title': 'Eczacılık Fakültesi',
                'page_title': 'Eczacılık Fakültesi',
                'source_group': 'department',
            },
        )
        retrieve_context_mock.return_value = [staff_chunk, department_chunk]

        payload = chat('Eczacılık fakültesi bilgi')

        self.assertIn('Eczacılık Fakültesi hakkında resmi kaynakta', payload['answer'])
        self.assertIn('misyon, eğitim ve laboratuvar bilgileri', payload['answer'])
        self.assertIn('[1]', payload['answer'])
        self.assertEqual(payload['sources'][0]['title'], 'Eczacılık Fakültesi')
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_distinguishes_dentistry_from_oral_health_program(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/agiz-dis',
            source='bologna',
            title='Ağız ve Diş Sağlığı - Programı Bilgileri',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-agiz-dis',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Ağız ve Diş Sağlığı Programı',
            metadata={
                'kind': 'bologna_program_page',
                'program_title': 'Ağız ve Diş Sağlığı',
                'faculty': 'Sağlık Hizmetleri Meslek Yüksekokulu',
                'admission_level': 'onlisans',
            },
        )
        retrieve_context_mock.return_value = [chunk]

        payload = chat('Acıbadem üniversitesinde dişçilik bölümü var mı')

        self.assertIn('Diş Hekimliği lisans programı bulamadım', payload['answer'])
        self.assertIn('Ağız ve Diş Sağlığı programı var', payload['answer'])
        self.assertIn('ön lisans', payload['answer'])
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

    @patch('chat.services._acquire_llm_slot', side_effect=LLMBusyError('busy'))
    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_busy_answer_without_caching(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
        _acquire_llm_slot_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/genel',
            source='main_site',
            title='Genel Bilgi',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-busy',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Model gerektiren genel resmi bilgi.',
        )
        retrieve_context_mock.return_value = [chunk]

        payload = chat('Genel bilgi verir misin?')

        self.assertEqual(payload['answer'], LLM_BUSY_ANSWER)
        self.assertEqual(payload['sources'], [])
        self.assertTrue(payload['busy'])
        self.assertIsNone(cache.get(cache_key('Genel bilgi verir misin?')))
        self.assertEqual(Message.objects.filter(role='assistant').get().content, LLM_BUSY_ANSWER)
        generate_answer_mock.assert_not_called()

    @patch('chat.services._acquire_llm_slot', side_effect=LLMBusyError('busy'))
    @patch('chat.services.generate_answer_stream')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_stream_returns_busy_answer_without_caching(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_stream_mock,
        _acquire_llm_slot_mock,
    ):
        page = WebPage.objects.create(
            url='https://example.com/stream-busy',
            source='main_site',
            title='Genel Bilgi',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-stream-busy',
        )
        chunk = ContentChunk.objects.create(
            page=page,
            chunk_index=0,
            text='Streaming model gerektiren genel resmi bilgi.',
        )
        retrieve_context_mock.return_value = [chunk]

        payload = ''.join(chat_stream('Streaming genel bilgi verir misin?'))

        self.assertIn(LLM_BUSY_ANSWER, payload)
        self.assertIn('event: sources', payload)
        self.assertIn('"sources": []', payload)
        self.assertIsNone(cache.get(cache_key('Streaming genel bilgi verir misin?')))
        self.assertEqual(Message.objects.filter(role='assistant').get().content, LLM_BUSY_ANSWER)
        generate_answer_stream_mock.assert_not_called()

    def test_is_staff_query_matches_baskan(self):
        self.assertTrue(_is_staff_query('bilgisayar mühendisliği bölüm başkanı kimdir'))

    def test_is_staff_query_matches_dekan(self):
        self.assertTrue(_is_staff_query('dekan kimdir'))

    def test_is_staff_query_matches_mudur(self):
        self.assertTrue(_is_staff_query('müdür'))

    def test_is_staff_query_matches_baskan_ascii(self):
        self.assertTrue(_is_staff_query('bolum baskani kim'))

    def test_is_staff_query_no_match_for_non_staff(self):
        self.assertFalse(_is_staff_query('bilgisayar mühendisliği dersleri'))

    def test_filter_candidates_prioritizes_staff_chunks_for_staff_queries(self):
        staff_page = WebPage.objects.create(
            url='https://example.com/staff-page',
            source='bologna',
            title='Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-staff',
        )
        score_page = WebPage.objects.create(
            url='https://example.com/score-page',
            source='structured',
            title='Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-score',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Akademik kadro listesi.',
            metadata={
                'kind': 'bologna_staff_page',
                'program_title': 'Bilgisayar Mühendisliği',
            },
        )
        score_chunk = ContentChunk.objects.create(
            page=score_page,
            chunk_index=0,
            text='Taban puan: 450.',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Bilgisayar Mühendisliği',
            },
        )

        result = _filter_candidates_for_query(
            'bilgisayar mühendisliği bölüm başkanı kimdir',
            [score_chunk, staff_chunk],
        )

        self.assertEqual(result, [staff_chunk])

    def test_filter_candidates_falls_back_for_staff_queries_without_staff_chunks(self):
        score_page = WebPage.objects.create(
            url='https://example.com/score-page',
            source='structured',
            title='Kontenjan ve Puan',
            content_text='icerik',
            raw_html='{}',
            content_hash='hash-score',
        )
        score_chunk = ContentChunk.objects.create(
            page=score_page,
            chunk_index=0,
            text='Taban puan: 450.',
            metadata={
                'kind': 'structured_admissions_score',
                'program_title': 'Bilgisayar Mühendisliği',
            },
        )

        result = _filter_candidates_for_query(
            'bilgisayar mühendisliği bölüm başkanı kimdir',
            [score_chunk],
        )

        self.assertEqual(result, [score_chunk])

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_department_head_without_academic_staff_list(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        head_page = WebPage.objects.create(
            url='https://example.com/bolum-baskani',
            source='main_site',
            title='Bilgisayar Mühendisliği - Bölüm Başkanının Mesajı',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-head',
        )
        head_chunk = ContentChunk.objects.create(
            page=head_page,
            chunk_index=0,
            text='Bölüm Başkanı - Prof. Dr. Ahmet Bulut',
            metadata={
                'record_type': 'department_head_message',
                'program_title': 'Bilgisayar Mühendisliği',
                'section_title': 'Bölüm Başkanının Mesajı',
            },
        )
        staff_page = WebPage.objects.create(
            url='https://example.com/bilgisayar-akademik-kadro',
            source='main_site',
            title='Bilgisayar Mühendisliği - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-staff-for-head',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Bilgisayar Mühendisliği akademik kadro | isim: Ahmet Bulut | unvan: Prof. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'program_title': 'Bilgisayar Mühendisliği',
                'entity_name': 'Ahmet Bulut',
                'staff_title': 'Prof. Dr.',
                'staff_count': 1,
            },
        )
        retrieve_context_mock.return_value = [staff_chunk]

        payload = chat('Bilgisayar mühendisliği bölüm başkanı kimdir?')

        self.assertEqual(
            payload['answer'],
            'Bilgisayar Mühendisliği bölüm başkanı Prof. Dr. Ahmet Bulut olarak görünüyor.',
        )
        self.assertNotIn('akademik kadro kaynağında', payload['answer'])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_returns_dean_from_role_page_without_academic_staff_list(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        role_page = WebPage.objects.create(
            url='https://example.com/tip-fakultesi-yonetimi',
            source='main_site',
            title='Tıp Fakültesi Yönetimi',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-tip-role',
        )
        role_chunk = ContentChunk.objects.create(
            page=role_page,
            chunk_index=0,
            text='Birim: Tıp Fakültesi\nTıp Fakültesi yönetim | rol: Dekan | isim: Prof. Dr. Nadi Bakırcı',
            metadata={
                'kind': 'main_site_role_page',
                'record_type': 'staff_role_assignment',
                'faculty': 'Tıp Fakültesi',
                'program_title': 'Tıp Fakültesi Yönetimi',
                'section_title': 'Tıp Fakültesi Yönetimi',
            },
        )
        staff_page = WebPage.objects.create(
            url='https://example.com/tip-akademik-kadro',
            source='main_site',
            title='Tıp Fakültesi - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-tip-staff',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Tıp Fakültesi akademik kadro | isim: Bora Özveren | unvan: Doç. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'faculty': 'Tıp Fakültesi',
                'entity_name': 'Bora Özveren',
                'staff_title': 'Doç. Dr.',
                'staff_count': 249,
            },
        )
        retrieve_context_mock.return_value = [staff_chunk]

        payload = chat('Tıp fakültesi dekanı kim?')

        self.assertEqual(
            payload['answer'],
            'Tıp Fakültesi dekanı Prof. Dr. Nadi Bakırcı olarak görünüyor.',
        )
        self.assertNotIn('akademik kadro kaynağında', payload['answer'])
        generate_answer_mock.assert_not_called()

    @patch('chat.services.generate_answer')
    @patch('chat.services.retrieve_keyword_context', return_value=[])
    @patch('chat.services.retrieve_context')
    @patch('chat.services.embed_query', return_value=[0.1, 0.2, 0.3])
    def test_chat_role_specific_staff_query_does_not_return_staff_list_without_role_source(
        self,
        _embed_query_mock,
        retrieve_context_mock,
        _retrieve_keyword_context_mock,
        generate_answer_mock,
    ):
        staff_page = WebPage.objects.create(
            url='https://example.com/tip-akademik-kadro-only',
            source='main_site',
            title='Tıp Fakültesi - Akademik Kadro',
            content_text='icerik',
            raw_html='<main>icerik</main>',
            content_hash='hash-tip-staff-only',
        )
        staff_chunk = ContentChunk.objects.create(
            page=staff_page,
            chunk_index=0,
            text='Tıp Fakültesi akademik kadro | isim: Bora Özveren | unvan: Doç. Dr.',
            metadata={
                'kind': 'main_site_staff_page',
                'record_type': 'academic_staff_member',
                'faculty': 'Tıp Fakültesi',
                'entity_name': 'Bora Özveren',
                'staff_title': 'Doç. Dr.',
                'staff_count': 249,
            },
        )
        retrieve_context_mock.return_value = [staff_chunk]

        payload = chat('Tıp fakültesi dekanı kim?')

        self.assertEqual(
            payload['answer'],
            'Bu rol için doğrulanmış yönetici kaynağı bulamadım.',
        )
        self.assertNotIn('akademik kadro kaynağında', payload['answer'])
        generate_answer_mock.assert_not_called()


class WarmModelsCommandTests(TestCase):
    @override_settings(EMBEDDING_BACKEND='local')
    @patch('chat.management.commands.warm_models.warm_llm_model')
    @patch('chat.management.commands.warm_models.warm_embedding_model')
    def test_warm_models_command_skips_llm_warmup_by_default(
        self, warm_embedding_model_mock, warm_llm_model_mock
    ):
        stdout = StringIO()

        call_command('warm_models', stdout=stdout)

        warm_embedding_model_mock.assert_called_once()
        warm_llm_model_mock.assert_not_called()
        output = stdout.getvalue()
        self.assertIn('Warmup completed for embedding_model.', output)
        self.assertIn('LLM warmup skipped: disabled by settings.', output)

    @override_settings(EMBEDDING_BACKEND='local', LLM_WARMUP_ENABLED=True)
    @patch('chat.management.commands.warm_models.warm_llm_model')
    @patch('chat.management.commands.warm_models.warm_embedding_model')
    def test_warm_models_command_logs_llm_success(
        self, warm_embedding_model_mock, warm_llm_model_mock
    ):
        stdout = StringIO()

        call_command('warm_models', stdout=stdout)

        warm_embedding_model_mock.assert_called_once()
        warm_llm_model_mock.assert_called_once()
        output = stdout.getvalue()
        self.assertIn('Warmup completed for embedding_model.', output)
        self.assertIn('Warmup completed for llm_model.', output)

    @override_settings(EMBEDDING_BACKEND='local')
    @patch('chat.management.commands.warm_models.warm_llm_model')
    @patch('chat.management.commands.warm_models.warm_embedding_model')
    def test_warm_models_command_llm_flag_overrides_default(
        self, warm_embedding_model_mock, warm_llm_model_mock
    ):
        stdout = StringIO()

        call_command('warm_models', llm=True, stdout=stdout)

        warm_embedding_model_mock.assert_called_once()
        warm_llm_model_mock.assert_called_once()
        self.assertIn('Warmup completed for llm_model.', stdout.getvalue())

    @override_settings(EMBEDDING_BACKEND='local', LLM_WARMUP_ENABLED=True)
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

    @override_settings(EMBEDDING_BACKEND='api')
    @patch('chat.management.commands.warm_models.warm_llm_model')
    @patch('chat.management.commands.warm_models.warm_embedding_model')
    def test_warm_models_skips_embedding_warmup_in_api_mode(
        self, warm_embedding_model_mock, warm_llm_model_mock
    ):
        stdout = StringIO()

        call_command('warm_models', stdout=stdout)

        warm_embedding_model_mock.assert_not_called()
        warm_llm_model_mock.assert_not_called()
        self.assertIn('skipping local warmup', stdout.getvalue())
        self.assertIn('LLM warmup skipped', stdout.getvalue())

    def test_campus_life_topic_matches_sosyal_imkanlar(self):
        topics = _question_topics('Kampüste hangi sosyal imkanlar var?')
        self.assertIn('campus_life', topics)

    def test_campus_life_topic_matches_kampus_yasam(self):
        topics = _question_topics('Kampüs yaşam hakkında bilgi verir misiniz?')
        self.assertIn('campus_life', topics)

    def test_campus_life_topic_matches_yemekhane(self):
        topics = _question_topics('Yemekhane var mı?')
        self.assertIn('campus_life', topics)

    def test_campus_life_topic_matches_ogrenci_hayati(self):
        topics = _question_topics('Öğrenci hayatı nasıl?')
        self.assertIn('campus_life', topics)

    def test_campus_life_topic_does_not_match_general_program_query(self):
        topics = _question_topics('Bilgisayar mühendisliği dersleri neler?')
        self.assertNotIn('campus_life', topics)

    def test_campus_life_topic_combines_with_library_and_sports(self):
        topics = _question_topics('Kampüste kütüphane ve spor imkanları var mı?')
        self.assertIn('campus_life', topics)
        self.assertIn('library', topics)
        self.assertIn('sports', topics)

    @override_settings(
        RAG_RETRIEVE_LIMIT=3,
        RAG_PER_PAGE_LIMIT=3,
        RAG_VECTOR_DISTANCE_STRICT=0.72,
        RAG_QUERY_EXPANSION_ENABLED=True,
    )
    def test_direct_facility_retrieval_includes_campus_life_terms(self):
        campus_page = WebPage.objects.create(
            url='https://example.com/campus-life',
            source='main_site',
            title='Kampüs Yaşam',
            content_text='Yemekhane ve kafeterya',
            raw_html='<main>Yemekhane ve kafeterya</main>',
            content_hash='hash-campus',
        )
        campus_chunk = ContentChunk.objects.create(
            page=campus_page,
            chunk_index=0,
            text='Kampüste yemekhane, kafeterya ve sosyal alanlar bulunmaktadır.',
            metadata={'kind': 'main_site_page', 'source_group': 'department'},
        )
        results = _retrieve_direct_facility_chunks('Kampüste hangi sosyal imkanlar var?', limit=5)
        chunk_ids = [c.id for c in results]
        self.assertIn(campus_chunk.id, chunk_ids)

    @override_settings(
        RAG_RETRIEVE_LIMIT=3,
        RAG_PER_PAGE_LIMIT=3,
    )
    def test_filter_candidates_returns_campus_chunks_for_campus_life_query(self):
        campus_page = WebPage.objects.create(
            url='https://example.com/campus',
            source='main_site',
            title='Kampüs Olanakları',
            content_text='Yemekhane ve spor',
            raw_html='<main>Yemekhane ve spor</main>',
            content_hash='hash-campus2',
        )
        irrelevant_page = WebPage.objects.create(
            url='https://example.com/score',
            source='structured',
            title='Kontenjan ve Puan',
            content_text='Taban puan bilgisi',
            raw_html='{}',
            content_hash='hash-score2',
        )
        campus_chunk = ContentChunk.objects.create(
            page=campus_page,
            chunk_index=0,
            text='Kampüste yemekhane, kafeterya ve sosyal alanlar bulunmaktadır.',
            metadata={'kind': 'main_site_page', 'source_group': 'department', 'topic': 'campus_life'},
        )
        score_chunk = ContentChunk.objects.create(
            page=irrelevant_page,
            chunk_index=0,
            text='Taban puan: 350',
            metadata={'kind': 'structured_admissions_score', 'record_type': 'quota_row'},
        )
        filtered = _filter_candidates_for_query(
            'Kampüste hangi sosyal imkanlar var?',
            [campus_chunk, score_chunk],
        )
        self.assertIn(campus_chunk, filtered)
        self.assertNotIn(score_chunk, filtered)
