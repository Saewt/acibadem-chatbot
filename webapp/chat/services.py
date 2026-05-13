import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import requests
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector, TrigramSimilarity
from django.core.cache import cache
from django.db import transaction
from django.db.models import F, Q, Value
from django.db.models.functions import Coalesce
from openai import OpenAI
from pgvector.django import CosineDistance

from scraper.embeddings import embed_text
from scraper.models import ContentChunk

from .models import Conversation, Message

_thread_local = threading.local()
logger = logging.getLogger(__name__)
NO_CONTEXT_ANSWER = (
    'Bu konuda doğrulanmış üniversite kaynağı bulamadım. '
    'İstersen soruyu daha spesifik sorabilir veya başka bir resmi sayfayı hedefleyebilirsin.'
)
LLM_BUSY_ANSWER = (
    'Model şu anda başka bir yanıt üretiyor. '
    'Lütfen birkaç saniye sonra tekrar deneyin.'
)
LLM_FORMAT_ERROR_ANSWER = (
    'Bu soruya yanıt oluşturulurken model çıktısı beklenen biçimde gelmedi. '
    'Lütfen tekrar dener misin?'
)
QUESTION_HASH_LENGTH = 12
CACHE_KEY_VERSION = 'v30'
CANDIDATE_LIMIT_MULTIPLIER = 3
SSE_DONE_SENTINEL = '[DONE]'
PROGRAM_ABBREVIATION_MIN_LENGTH = 3
PROGRAM_ABBREVIATION_STOP_WORDS = frozenset({
    've',
    'ile',
    'and',
    'of',
    'the',
    'ingilizce',
    'türkçe',
    'turkce',
})


@dataclass
class RetrievalHit:
    chunk: ContentChunk
    method: str
    rank: int
    weight: float
    protected: bool = False
    distance: float | None = None
    keyword_rank: float | None = None
STAFF_QUERY_PATTERN = re.compile(
    r'\b('
    r'hoca\w*|'
    r'hocası\w*|'
    r'akademik\s*kadro\w*|'
    r'akademik\s*personel\w*|'
    r'öğretim\s*üye\w*|'
    r'ogretim\s*uye\w*|'
    r'öğretim\s*görevli\w*|'
    r'ogretim\s*gorevli\w*|'
    r'personel\w*|'
    r'profesör\w*|'
    r'profesor\w*|'
    r'başkan\w*|'
    r'baskan\w*|'
    r'dekan\w*|'
    r'müdür\w*|'
    r'mudur\w*|'
    r'yönetim\w*|'
    r'yonetim\w*|'
    r'görevli\w*|'
    r'gorevli\w*|'
    r'yetkili\w*'
    r')\b'
)
STAFF_COUNT_QUERY_PATTERN = re.compile(
    r'\b(kaç|kac|ne\s*kadar|say[ıi]s[ıi]|adet|tane)\b'
)
DEPARTMENT_HEAD_QUERY_PATTERN = re.compile(
    r'\b(bölüm\s*başkan\w*|bolum\s*baskan\w*)\b'
)
DEPUTY_DEAN_QUERY_PATTERN = re.compile(r'\bdekan\s+yardimc\w*\b')
DEAN_QUERY_PATTERN = re.compile(r'\bdekan\w*\b')
DIRECTOR_QUERY_PATTERN = re.compile(r'\bmudur\w*\b')
PROGRAM_EXISTS_QUERY_PATTERN = re.compile(
    r'\b(var\s*m[ıi]|bulunuyor\s*mu|mevcut\s*mu|aç[ıi]k\s*m[ıi])\b'
)
PROGRAM_LIST_QUERY_PATTERN = re.compile(
    r'\b(?:hangi|hangileri|neler|nelerdir|listele\w*|göster\w*|goster\w*)\b'
    r'.*\b(?:bölüm\w*|bolum\w*|program\w*)\b|'
    r'\b(?:bölüm\w*|bolum\w*|program\w*)\b'
    r'.*\b(?:hangi|hangileri|neler|nelerdir|listele\w*|göster\w*|goster\w*)\b',
    re.IGNORECASE,
)
DENTISTRY_QUERY_PATTERN = re.compile(
    r'\b(dişçilik|discilik|diş\s*hekimliği|dis\s*hekimligi)\b'
)
SCORE_QUERY_PATTERN = re.compile(
    r'\b('
    r'puan\w*|'
    r'kontenjan\w*|'
    r'sıralama\w*|'
    r'siralama\w*|'
    r'başarı\s*sıras\w*|'
    r'basari\s*siras\w*|'
    r'taban\s*puan\w*|'
    r'tavan\s*puan\w*|'
    r'kaç\s*(?:kişi|kisi|öğrenci|ogrenci|tane|adet)|'
    r'kac\s*(?:kisi|ogrenci|tane|adet)|'
    r'kaç\s*kontenjan\w*|'
    r'kac\s*kontenjan\w*|'
    r'al(?:ı|i)yor\b|'
    r'al(?:ı|i)m\s*(?:say[ıi]s[ıi]|miktar[ıi])'
    r')\b'
)
FEE_QUERY_PATTERN = re.compile(
    r'\b('
    r'ücret\w*|'
    r'ucret\w*|'
    r'öğrenim\s*ücret\w*|'
    r'ogrenim\s*ucret\w*|'
    r'harç\w*|'
    r'harc\w*'
    r')\b'
)
EXPLICIT_PROGRAM_ABBREVIATIONS = (
    (
        re.compile(r'\b(?:pc|bilg(?:isayar)?)\s+(?:müh\w*|muh\w*)\b'),
        'Bilgisayar Mühendisliği',
    ),
)
SCHOLARSHIP_QUERY_PATTERN = re.compile(r'\b(burs\w*|scholarship\w*)\b')
DORMITORY_QUERY_PATTERN = re.compile(r'\b(yurt\w*|depozito\w*|konaklama\w*|dorm\w*)\b')
INTERNATIONAL_QUERY_PATTERN = re.compile(
    r'\b(uluslararası\w*|uluslararasi\w*|erasmus\w*|yurtdış\w*|yurtdis\w*|hareketlilik\w*)\b'
)
DOUBLE_MAJOR_MINOR_QUERY_PATTERN = re.compile(
    r'\b(çift\s*anadal\w*|cift\s*anadal\w*|yandal\w*|çap\w*|cap\w*|minor\w*|major\w*)\b'
)
LIBRARY_QUERY_PATTERN = re.compile(
    r'\b(kütüphane\w*|kutuphane\w*|library\w*)\b'
)
SPORTS_QUERY_PATTERN = re.compile(
    r'\b(spor\w*|sport\w*|fitness\w*|gym\w*|yüzme\w*|yuzme\w*|havuz\w*|basketbol\w*|antrenman\w*)\b'
)
CAMPUS_LIFE_QUERY_PATTERN = re.compile(
    r'(?:'
    r'kampüs\w*|kampus\w*|'
    r'sosyal\s*(?:imkan|olanak|tesis|etkinlik|yasam|haya|yaşam)\w*|'
    r'imkan\w*|olanak\w*|'
    r'yemekhane\w*|kafeterya\w*|kafe\w*|'
    r'öğrenci\s*(?:hayat|yaşam|haya)\w*|ogrenci\s*(?:hayat|yasam|haya)\w*|'
    r'kampüs\w*\s*(?:imkan|olanak|yaşam|yasam|haya|gezi)\w*|'
    r'sosyal\s*haya\w*|sosyal\s*yaşa\w*|'
    r'acuda\s*yaşam\w*|acuda\s*yasam\w*|acu\s*da\s*yaşam\w*|acu\s*da\s*yasam\w*|'
    r'kampus\s*yaşam\w*|kampus\s*yasam\w*'
    r')',
    re.IGNORECASE,
)
STUDENT_CLUB_QUERY_PATTERN = re.compile(
    r'\b('
    r'kulüp\w*|'
    r'kulup\w*|'
    r'kulub\w*|'
    r'topluluk\w*|'
    r'öğrenci\s*kulüb\w*|'
    r'ogrenci\s*kulub\w*|'
    r'öğrenci\s*topluluk\w*|'
    r'ogrenci\s*topluluk\w*|'
    r'student\s*club\w*'
    r')\b',
    re.IGNORECASE,
)

TRANSPORT_QUERY_PATTERN = re.compile(
    r'\b(ulaşım\w*|ulasim\w*|otobüs\w*|otobus\w*|metro\w*|ring\w*|servis\w*|otopark\w*|trafik\w*|dolmuş\w*|dolmus\w*)\b'
)

PREP_QUERY_PATTERN = re.compile(
    r'\b(hazırlık\w*|hazirlik\w*|muaf\w*|muafiyet\w*|ingilizce.*hazırlık|hazırlık.*ingilizce|i̇ngilizce.*hazırlık|hazırlık.*i̇ngilizce)\b'
)

COMPARISON_QUERY_PATTERN = re.compile(
    r'\b(vs\b|veya\b|yoksa\b|arasındaki\s+fark|farkı\s+nedir|karşılaştır\w*|karsilastir\w*|hangisi\s+daha)\b'
)
RANK_QUERY_PATTERN = re.compile(
    r'\b(sıralama\w*|siralama\w*|başarı\s*sıras\w*|basari\s*siras\w*)\b'
)
POINTS_QUERY_PATTERN = re.compile(r'\b(puan\w*|taban\s*puan\w*|tavan\s*puan\w*)\b')
QUOTA_QUERY_PATTERN = re.compile(r'\b(kontenjan\w*)\b')
COURSE_QUERY_PATTERN = re.compile(
    r'\b('
    r'ders\w*|'
    r'müfredat\w*|'
    r'mufredat\w*|'
    r'yarıyıl\w*|'
    r'yariyil\w*|'
    r'sınıf\w*|'
    r'sinif\w*|'
    r'akts\w*|'
    r'ects\w*|'
    r'semester\w*'
    r')\b'
)
COURSE_FULL_LIST_QUERY_PATTERN = re.compile(
    r'\b(tüm\s+ders\w*|tum\s+ders\w*|heps\w*|tam\s+liste\w*|listele\w*)\b'
)
TOPIC_KEYWORDS = {
    'scholarships': (
        'burs',
        'scholarship',
    ),
    'dormitory': (
        'yurt',
        'depozito',
        'konaklama',
        'dorm',
    ),
    'international': (
        'erasmus',
        'uluslararası',
        'uluslararasi',
        'hareketlilik',
        'yurtdış',
        'yurtdis',
        'değişim',
        'degisim',
    ),
    'double_major_minor': (
        'çift anadal',
        'cift anadal',
        'yandal',
        'çap',
        'cap',
        'minor',
        'major',
    ),
    'library': (
        'kütüphane',
        'kutuphane',
        'library',
        'çalışma alanı',
        'calisma alani',
        'veritabanı',
        'veritabani',
        'kaynak',
    ),
    'sports': (
        'spor',
        'spor merkezi',
        'fitness',
        'yüzme',
        'yuzme',
        'basketbol',
        'antrenman',
    ),
    'campus_life': (
        'kampüs',
        'kampus',
        'sosyal',
        'yemekhane',
        'kafeterya',
        'kütüphane',
        'kutuphane',
        'spor',
        'spor merkezi',
        'yurt',
        'hizmetlerimiz',
        'acuda',
    ),
    'student_clubs': (
        'öğrenci kulüpleri',
        'ogrenci kulupleri',
        'kulüp',
        'kulup',
        'kulub',
        'topluluk',
        'topluluğu',
        'toplulugu',
    ),
    'transport': (
        'ulaşım',
        'ulasim',
        'otobüs',
        'otobus',
        'metro',
        'ring',
        'servis',
        'otopark',
    ),
    'prep': (
        'hazırlık',
        'hazirlik',
        'muafiyet',
        'muaf',
        'ingilizce',
        'i̇ngilizce',
        'yeterlilik',
        'hazırlık sınıfı',
        'hazirlik sinifi',
    ),
    'comparison': (
        'fark',
        'karşılaştırma',
        'karsilastirma',
        'avantaj',
        'dezavantaj',
    ),
}
QUERY_EXPANSIONS = {
    'scholarships': (
        'burs',
        'başarı bursu',
        'basari bursu',
        'indirim',
        'öğrenim ücreti',
        'ogrenim ucreti',
        'karşılıksız',
        'karsiliksiz',
        'burs başvuru',
        'burs basvuru',
    ),
    'library': (
        'kütüphane',
        'kutuphane',
        'library',
        'çalışma alanı',
        'calisma alani',
        'veritabanı',
        'veritabani',
        'kaynak',
    ),
    'sports': (
        'spor',
        'spor merkezi',
        'fitness',
        'gym',
        'havuz',
        'yüzme',
        'yuzme',
        'basketbol',
        'antrenman',
    ),
    'dormitory': (
        'yurt',
        'konaklama',
        'depozito',
        'başvuru',
        'basvuru',
        'dorm',
    ),
    'international': (
        'erasmus',
        'uluslararası',
        'uluslararasi',
        'değişim',
        'degisim',
        'hareketlilik',
    ),
    'double_major_minor': (
        'çift anadal',
        'cift anadal',
        'yandal',
        'çap',
        'cap',
        'minor',
        'major',
    ),
    'campus_life': (
        'kampüs yaşam',
        'kampus yasam',
        'kampüs olanakları',
        'kampus olanaklari',
        'acuda yaşam',
        'acuda yasam',
        'sosyal alan',
        'kampüs',
        'kampus',
        'sosyal',
        'yemekhane',
        'kafeterya',
        'kütüphane',
        'kutuphane',
        'spor',
        'spor merkezi',
        'yurt',
    ),
    'student_clubs': (
        'öğrenci kulüpleri',
        'ogrenci kulupleri',
        'öğrenci toplulukları',
        'ogrenci topluluklari',
        'kulübü',
        'kulubu',
        'kulüp',
        'kulup',
        'topluluğu',
        'toplulugu',
        'topluluk',
        'acuda yaşam',
        'acuda yasam',
    ),
    'transport': (
        'ulaşım',
        'ulasim',
        'otobüs',
        'otobus',
        'metro',
        'ring',
        'servis',
        'otopark',
        'kampüs ulaşım',
        'kampus ulasim',
    ),
    'prep': (
        'hazırlık',
        'hazirlik',
        'muafiyet',
        'muaf',
        'ingilizce',
        'i̇ngilizce',
        'yeterlilik',
        'hazırlık sınıfı',
        'hazirlik sinifi',
    ),
    'comparison': (
        'fark',
        'karşılaştırma',
        'karsilastirma',
        'avantaj',
        'dezavantaj',
    ),
}


TOPIC_DESCRIPTIONS: dict[str, list[str]] = {
    'campus_life': [
        'kampüs yaşamı sosyal imkanlar',
        'üniversitede sosyal hayat etkinlikler',
        'kampüs olanakları tesisler',
        'yemekhane kafeterya sosyal alanlar',
        'öğrenci aktiviteleri sosyal alanlar',
        'okulda hangi imkanlar var',
        'üniversite sosyal yaşam',
    ],
    'student_clubs': [
        'öğrenci kulüpleri topluluklar',
        'üniversitedeki kulüpler ve topluluklar',
        'kulüp listesi öğrenci toplulukları',
        'acuda yaşam öğrenci kulüpleri',
    ],
    'library': [
        'kütüphane kaynak çalışma alanı',
        'kütüphane saatleri veritabanı erişim',
    ],
    'sports': [
        'spor merkezi fitness salon',
        'spor imkanları yüzme basketbol',
    ],
    'dormitory': [
        'öğrenci yurdu konaklama',
        'yurt başvuru ücret kampüs içi barınma',
    ],
    'international': [
        'uluslararası öğrenci Erasmus değişim programları',
        'yurtdışı hareketlilik bilateral agreements',
    ],
    'scholarships': [
        'burs başarı bursu öğrenim desteği',
        'burs başvuru karşılıksız indirim',
    ],
    'transport': [
        'kampüs ulaşım servis shuttle ring',
        'otobüs metro ulaşım imkanları',
    ],
    'prep': [
        'İngilizce hazırlık sınıfı muafiyet sınavı',
        'hazırlık programı dil eğitimi yeterlilik',
    ],
}

GREETING_WORDS = frozenset({
    'merhaba', 'selam', 'hello', 'selamlar', 'hey', 'hi', 'slm', 'merhabalar'
})
STATE_WORDS = frozenset({
    'naber', 'nasılsın', 'nasilsin', 'ne haber'
})
TEST_WORDS = frozenset({
    'test', 'deneme'
})
_THINK_TAG_PATTERN = re.compile(r'<think\b[^>]*>.*?</think>', re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_MARKER_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:possible answer|final answer|cevap|nihai cevap)\s*:\s*',
    re.IGNORECASE,
)
_ANALYSIS_OPENING_PATTERN = re.compile(
    r"^\s*(?:okay|ok|let's|first,|the user|i need|i should|we need|"
    r"source\s+\d+\s*:|wait,|looking at|from the sources)",
    re.IGNORECASE,
)
_ANALYSIS_LINE_PATTERN = re.compile(
    r"^\s*(?:source\s+\d+\s*:|the user|i need|i should|wait,|possible answer\s*:)",
    re.IGNORECASE,
)
_TOPIC_MATCH_TERMS = {
    'campus_life': (
        'kampüs',
        'kampus',
        'kampüs yaşam',
        'kampus yasam',
        'kampüs olanakları',
        'kampus olanaklari',
        'acuda yaşam',
        'acuda yasam',
        'yemekhane',
        'kafeterya',
        'sosyal alan',
    ),
    'student_clubs': (
        'öğrenci kulüpleri',
        'ogrenci kulupleri',
        'kulübü',
        'kulubu',
        'kulüp',
        'kulup',
        'topluluğu',
        'toplulugu',
        'topluluk',
    ),
}
_TOPIC_EXCLUDED_SOURCE_GROUPS = {
    'campus_life': frozenset({'scholarship', 'quota', 'tuition'}),
    'student_clubs': frozenset({'scholarship', 'quota', 'tuition'}),
}


def _get_greeting_response(question: str) -> str | None:
    import string
    normalized = ' '.join(question.lower().strip().split())
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    
    if normalized in GREETING_WORDS:
        return 'Merhaba, sorularınızı sorabilirsiniz.'
    if normalized in STATE_WORDS:
        return 'İyiyim, teşekkürler! Sana nasıl yardımcı olabilirim?'
    if normalized in TEST_WORDS:
        return 'Sistem aktif. Sorularınızı sorabilirsiniz.'
    return None


class ConversationNotFoundError(Exception):
    pass


class LLMBusyError(RuntimeError):
    pass


_LLM_SEMAPHORE_LOCK = threading.Lock()
_LLM_SEMAPHORE: threading.BoundedSemaphore | None = None
_LLM_SEMAPHORE_LIMIT: int | None = None


def _get_llm_semaphore() -> threading.BoundedSemaphore:
    global _LLM_SEMAPHORE, _LLM_SEMAPHORE_LIMIT
    limit = max(int(settings.LLM_MAX_CONCURRENT_REQUESTS), 1)
    with _LLM_SEMAPHORE_LOCK:
        if _LLM_SEMAPHORE is None or _LLM_SEMAPHORE_LIMIT != limit:
            _LLM_SEMAPHORE = threading.BoundedSemaphore(limit)
            _LLM_SEMAPHORE_LIMIT = limit
        return _LLM_SEMAPHORE


def _acquire_llm_slot() -> threading.BoundedSemaphore:
    semaphore = _get_llm_semaphore()
    queue_timeout = max(float(settings.LLM_QUEUE_TIMEOUT), 0.0)
    if queue_timeout:
        acquired = semaphore.acquire(timeout=queue_timeout)
    else:
        acquired = semaphore.acquire(blocking=False)
    if not acquired:
        raise LLMBusyError('llm request limit reached')
    return semaphore


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=getattr(settings, 'LLM_API_KEY', 'not-needed'),
        timeout=settings.LLM_TIMEOUT,
    )


def _ollama_api_base_url() -> str:
    base_url = settings.LLM_BASE_URL.rstrip('/')
    if base_url.endswith('/v1'):
        base_url = base_url[:-3].rstrip('/')
    return base_url


def _use_ollama_backend() -> bool:
    return getattr(settings, 'LLM_BACKEND', 'ollama') == 'ollama'


def _ollama_chat_payload(
    messages: list[dict[str, str]],
    *,
    stream: bool,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        'model': settings.LLM_MODEL,
        'messages': messages,
        'stream': stream,
        'think': getattr(settings, 'LLM_THINK', True),
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        },
    }


def _ollama_chat(messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
    response = requests.post(
        f'{_ollama_api_base_url()}/api/chat',
        json=_ollama_chat_payload(
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        timeout=settings.LLM_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return _clean_llm_answer(str(payload.get('message', {}).get('content') or ''))


def _ollama_chat_stream(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> Iterator[str]:
    with requests.post(
        f'{_ollama_api_base_url()}/api/chat',
        json=_ollama_chat_payload(
            messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        stream=True,
        timeout=settings.LLM_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if payload.get('error'):
                raise RuntimeError(str(payload['error']))
            content = payload.get('message', {}).get('content')
            if content:
                yield str(content)
            if payload.get('done'):
                break


def _llm_system_prompt() -> str:
    return (
        '/no_think\n'
        'Sen Acıbadem Üniversitesi asistanısın. '
        'Kullanıcıya doğrudan, kısa ve net Türkçe cevap ver. '
        'Sadece kullanıcıya gösterilecek nihai cevabı yaz. '
        'Yanıtını CEVAP: etiketiyle başlat. '
        'Kaynakları nasıl incelediğini, adım adım analizini veya taslağını yazma. '
        'Asla <think> veya </think> yazma. '
        'İç muhakeme, analiz, taslak veya İngilizce düşünme metni gösterme.'
    )


def _looks_like_analysis_text(text: str) -> bool:
    if not text:
        return False
    if _ANALYSIS_OPENING_PATTERN.search(text):
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    analysis_lines = sum(1 for line in lines[:8] if _ANALYSIS_LINE_PATTERN.search(line))
    return analysis_lines >= 2


def _extract_final_answer_from_analysis(text: str) -> str:
    matches = list(_FINAL_ANSWER_MARKER_PATTERN.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end():].strip()


def _has_final_answer_marker(text: str) -> bool:
    return bool(_FINAL_ANSWER_MARKER_PATTERN.search(text))


def _clean_llm_answer(answer: str) -> str:
    text = (answer or '').strip()
    if not text:
        return ''
    text = _THINK_TAG_PATTERN.sub('', text)
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1]
    text = re.sub(r'</?think\b[^>]*>', '', text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return ''

    if _has_final_answer_marker(text):
        text = _extract_final_answer_from_analysis(text)
    elif _looks_like_analysis_text(text):
        text = _extract_final_answer_from_analysis(text)
    if _looks_like_analysis_text(text):
        return LLM_FORMAT_ERROR_ANSWER
    return text.strip()


def _is_llm_format_error_answer(answer: str) -> bool:
    return (answer or '').strip() == LLM_FORMAT_ERROR_ANSWER


def _llm_retry_prompt(prompt: str) -> str:
    return (
        '/no_think\n'
        'Önceki çıktı geçersizdi çünkü analiz veya taslak içeriyordu.\n'
        'Aşağıdaki görevi yeniden uygula.\n'
        'Yalnızca "CEVAP:" ile başlayan nihai Türkçe cevabı yaz.\n'
        'CEVAP etiketi öncesine hiçbir metin yazma; kaynak analizi, İngilizce açıklama '
        'veya düşünme süreci yazma.\n\n'
        f'{prompt}'
    )


def warm_embedding_model() -> None:
    embed_text('warmup')


def warm_llm_model() -> None:
    messages = [
        {
            'role': 'system',
            'content': 'Return a one-word acknowledgement.',
        },
        {'role': 'user', 'content': 'Ping'},
    ]
    if _use_ollama_backend():
        _ = _ollama_chat(messages, temperature=0, max_tokens=1)
        return

    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0,
        max_tokens=1,
        messages=messages,
    )
    _ = response.choices[0].message.content


def normalize_question(question: str) -> str:
    normalized = unicodedata.normalize('NFKC', question or '').casefold()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    return ' '.join(normalized.split())


def embed_query(question: str) -> list[float]:
    return embed_text(question)


def _normalize_lookup_text(value: str) -> str:
    return normalize_question(value)


def _get_metadata_value(metadata: dict | None, key: str) -> str:
    if not metadata:
        return ''
    value = metadata.get(key, '')
    return ' '.join(str(value).split()).strip()


def _get_chunk_metadata_value(chunk: ContentChunk, key: str) -> str:
    return _get_metadata_value(chunk.metadata, key) or _get_metadata_value(chunk.page.metadata, key)


def _clean_display_text(value: str) -> str:
    value = ' '.join(str(value or '').split())
    value = re.sub(r'\*+', '', value)
    return value.strip()


def _clean_note_text(value: str) -> str:
    value = _clean_display_text(value)
    value = re.sub(r'\s*Burs hakkında bilgi almak için tıklayın\.?', '', value)
    return value.strip()


def _normalized_contains(text: str, needle: str) -> bool:
    normalized_text = _normalize_lookup_text(text)
    normalized_needle = _normalize_lookup_text(needle)
    if not normalized_text or not normalized_needle:
        return False
    return bool(re.search(rf'(^|\s){re.escape(normalized_needle)}($|\s)', normalized_text))


def _build_scope_aliases(value: str) -> set[str]:
    aliases: set[str] = set()
    for raw_value in str(value or '').split('|'):
        normalized_raw_value = normalize_question(raw_value)
        if normalized_raw_value:
            aliases.add(normalized_raw_value)

        simplified = re.sub(r'\([^)]*\)', ' ', raw_value or '')
        simplified = _normalize_lookup_text(simplified)
        if simplified:
            aliases.add(simplified)
    return aliases


def _build_question_hint_aliases(value: str) -> set[str]:
    tokens = [token for token in _normalize_lookup_text(value).split() if token]
    aliases: set[str] = set()
    for start in range(len(tokens)):
        alias = ' '.join(tokens[start:])
        if alias:
            aliases.add(alias)
    return aliases


def _clean_scope_hint_value(value: str) -> str:
    stop_words = {'kaç', 'kac', 'tane', 'adet', 'sayısı', 'sayisi', 'ne', 'kadar'}
    tokens = [
        token
        for token in _normalize_lookup_text(value).split()
        if token and token not in stop_words
    ]
    if not tokens:
        return ''

    suffixes = ('nde', 'nda', 'de', 'da', 'den', 'dan', 'nin', 'nın', 'nun', 'nün')
    last_token = tokens[-1]
    for suffix in suffixes:
        if last_token.endswith(suffix) and len(last_token) > len(suffix) + 2:
            tokens[-1] = last_token[: -len(suffix)]
            break
    return ' '.join(tokens)


def _extract_question_scope_hints(question: str) -> list[tuple[str, set[str]]]:
    normalized_question = _normalize_lookup_text(question)
    hints: list[tuple[str, set[str]]] = []
    for field, pattern in (
        ('program_title', r'(?P<value>(?:\w+\s+){0,5}\w+)\s+(?:bölüm\w*|bolum\w*|program\w*)'),
        ('faculty', r'(?P<value>(?:\w+\s+){0,5}\w+)\s+(?:fakülte\w*|fakulte\w*)'),
        (
            'program_title',
            (
                r'(?P<value>(?:\w+\s+){0,4}\w+?)'
                r'(?:nde|nda|de|da|den|dan|nin|nın|nun|nün|in|ın|un|ün)?\s+'
                r'(?:(?:kaç|kac)\s+)?'
                r'(?:hoca\w*|akademik\s*kadro\w*|öğretim\s*üye\w*|'
                r'ogretim\s*uye\w*|profesör\w*|profesor\w*)'
            ),
        ),
    ):
        for match in re.finditer(pattern, normalized_question):
            aliases = _build_question_hint_aliases(_clean_scope_hint_value(match.group('value')))
            if aliases:
                hints.append((field, aliases))
    return hints


def _question_mentions_alias(question: str, alias: str) -> bool:
    if not alias:
        return False
    return bool(re.search(rf'(^|\s){re.escape(alias)}($|\s)', question))


def _known_program_candidates() -> list[str]:
    candidates: set[str] = set()
    metadata_fields = ('program_title', 'unit_name', 'placement_label')
    for field in metadata_fields:
        values = (
            ContentChunk.objects.filter(page__is_active=True)
            .exclude(**{f'metadata__{field}': ''})
            .values_list(f'metadata__{field}', flat=True)
            .distinct()
        )
        candidates.update(_clean_display_text(value) for value in values if value)

    page_titles = (
        ContentChunk.objects.filter(page__is_active=True)
        .values_list('page__title', flat=True)
        .distinct()
    )
    for title in page_titles:
        title = _clean_display_text(title)
        if ' - ' in title:
            candidates.add(title.split(' - ', 1)[0].strip())

    return sorted(
        {candidate for candidate in candidates if len(_normalize_lookup_text(candidate)) > 2},
        key=lambda value: len(_normalize_lookup_text(value)),
        reverse=True,
    )


def _extract_known_program_from_text(text: str) -> str:
    normalized_text = _normalize_lookup_text(text)
    if not normalized_text:
        return ''
    matches: list[str] = []
    for candidate in _known_program_candidates():
        aliases = _build_scope_aliases(candidate)
        if any(_question_mentions_alias(normalized_text, alias) for alias in aliases):
            matches.append(_clean_display_text(candidate))
    if not matches:
        return ''

    wants_english = _mentions_english(text)
    if wants_english:
        english_matches = [candidate for candidate in matches if _mentions_english(candidate)]
        if english_matches:
            matches = english_matches
    else:
        non_english_matches = [candidate for candidate in matches if not _mentions_english(candidate)]
        if non_english_matches:
            matches = non_english_matches

    return min(matches, key=lambda value: (len(_normalize_lookup_text(value)), value))


def _extract_program_hint_from_text(text: str) -> str:
    for field, aliases in _extract_question_scope_hints(text):
        if field == 'program_title' and aliases:
            return max(aliases, key=len)
    return ''


_PROGRAM_EXTRACT_STOP_WORDS = frozenset({
    'var', 'mi', 'mı', 'mu', 'mü', 'ne', 'kaç', 'kac', 'kim', 'nasil',
    'nasıl', 'nedir', 'nerede', 'hangi', 'neler', 'acaba', 'kadar',
    'programı', 'programi', 'bölümü', 'bolumu', 'fakültesi', 'fakultesi',
    'üniversitesi', 'universitesi', 'hakkında', 'hakkinda', 'bilgi',
    'hangisi', 'diyorum', 'sence',
})


def _extract_program_from_tokens(text: str) -> str:
    normalized_text = _normalize_lookup_text(text)
    if not normalized_text:
        return ''
    tokens = [t for t in normalized_text.split()
              if len(t) > 2 and t not in _PROGRAM_EXTRACT_STOP_WORDS]
    if not tokens:
        return ''
    candidates = _known_program_candidates()
    matches: list[tuple[str, int]] = []
    for candidate in candidates:
        match_count = sum(
            1 for token in tokens if _normalized_contains(candidate, token)
        )
        if match_count == len(tokens):
            matches.append((_clean_display_text(candidate), match_count, len(candidate)))
    if not matches:
        return ''
    matches.sort(key=lambda x: (-x[1], x[2]))
    return matches[0][0]


def _program_initialism(value: str) -> str:
    cleaned = _program_initialism_base(value)
    tokens = [
        token
        for token in _normalize_lookup_text(cleaned).split()
        if token and token not in PROGRAM_ABBREVIATION_STOP_WORDS
    ]
    initialism = ''.join(token[0] for token in tokens)
    if len(initialism) >= PROGRAM_ABBREVIATION_MIN_LENGTH:
        return initialism
    return ''


def _program_initialism_base(value: str) -> str:
    cleaned = re.sub(r'\([^)]*\)', ' ', value or '')
    cleaned = re.sub(r'\*+', ' ', cleaned)
    return _clean_display_text(cleaned)


def _program_initialism_candidates() -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    for program_title in _known_program_candidates():
        initialism = _program_initialism(program_title)
        if initialism:
            candidates.setdefault(initialism, set()).add(_clean_display_text(program_title))
    return candidates


def _extract_program_abbreviation_from_text(text: str) -> str:
    normalized_text = _normalize_lookup_text(text)
    if not normalized_text:
        return ''

    for pattern, program_title in EXPLICIT_PROGRAM_ABBREVIATIONS:
        if pattern.search(normalized_text):
            return program_title

    matches: list[tuple[str, str]] = []
    for initialism, program_titles in _program_initialism_candidates().items():
        base_titles = {
            _program_initialism_base(program_title)
            for program_title in program_titles
            if _program_initialism_base(program_title)
        }
        normalized_base_titles = {_normalize_lookup_text(title) for title in base_titles}
        if len(normalized_base_titles) != 1:
            continue
        if _question_mentions_alias(normalized_text, initialism):
            matches.append((initialism, min(base_titles, key=len)))
    if matches:
        _initialism, program_title = max(
            matches,
            key=lambda item: (len(item[0]), len(_normalize_lookup_text(item[1]))),
        )
        return program_title
    return ''


def _program_lookup_terms(program_title: str) -> set[str]:
    terms = {_clean_display_text(program_title)}
    for alias in _build_scope_aliases(program_title):
        if alias:
            terms.add(alias)
    return {term for term in terms if term}


def _question_has_explicit_scope(question: str) -> bool:
    if _is_dentistry_query(question):
        return True
    if _extract_question_scope_hints(question):
        return True
    if _extract_program_abbreviation_from_text(question):
        return True
    return bool(_extract_known_program_from_text(question))


_FOLLOWUP_STOP_WORDS = frozenset({
    'var', 'mi', 'mı', 'mu', 'mü', 'ne', 'kaç', 'kac', 'kim', 'nasil',
    'nasıl', 'nedir', 'nerede', 'hangi', 'kadar', 'bu', 'şu', 'o',
    'neler', 'nelerdir', 'hangileri', 'hangileridir',
})


def _looks_like_self_contained_question(question: str) -> bool:
    normalized = _normalize_lookup_text(question)
    content_tokens = [t for t in normalized.split() if t not in _FOLLOWUP_STOP_WORDS]
    return len(content_tokens) >= 2


def _is_followup_question(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    tokens = normalized_question.split()
    if len(tokens) <= 6 and any(
        token in normalized_question
        for token in ('hocası', 'hocaları', 'ücreti', 'puanı', 'kontenjanı', 'var mı')
    ):
        return True
    return bool(re.search(r'\b(bunun|onun|programın|bölümün|bolumun)\b', normalized_question))


def _resolve_question_with_conversation(question: str, conversation: Conversation) -> str:
    if _question_has_explicit_scope(question) or not _is_followup_question(question):
        return question

    previous_user_messages = (
        conversation.messages.filter(role='user')
        .order_by('-created_at')
        .values_list('content', flat=True)[:8]
    )
    for previous_question in previous_user_messages:
        program_title = (
            _extract_known_program_from_text(previous_question)
            or _extract_program_hint_from_text(previous_question)
        )
        if program_title:
            return f'{program_title} {question}'
    return question


def _dedupe_chunks(chunks: list[ContentChunk]) -> list[ContentChunk]:
    selected: list[ContentChunk] = []
    seen_chunk_ids: set[int] = set()
    for chunk in chunks:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)
        selected.append(chunk)
    return selected


def _limit_chunks(
    chunks: list[ContentChunk], limit: int | None = None, per_page_limit: int | None = None
) -> list[ContentChunk]:
    limit = limit or settings.RAG_RETRIEVE_LIMIT
    per_page_limit = per_page_limit or settings.RAG_PER_PAGE_LIMIT
    selected: list[ContentChunk] = []
    per_page_counts: dict[int, int] = {}

    for chunk in _dedupe_chunks(chunks):
        count = per_page_counts.get(chunk.page_id, 0)
        if count >= per_page_limit:
            continue
        per_page_counts[chunk.page_id] = count + 1
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


_SCOPE_STOP_WORDS = frozenset({
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    'sınıf', 'sınıfın', 'sınıfta', 'yarıyıl', 'ders', 'dersler', 'dersleri',
    'var', 'hangileri', 'nedir', 'kaç', 'kim', 'kimdir', 'hoca',
    'müfredat', 'planı', 'plan', 'akts', 'ects', 'zorunlu', 'seçmeli',
    've', 'veya', 'ile', 'için', 'hakkında', 'nerede', 'nasıl',
    'listele', 'ver', 'bul', 'göster', 'bölüm', 'fakülte', 'program',
})


GENERAL_INFO_QUERY_PATTERN = re.compile(
    r'\b(bilgi\w*|hakk[ıi]nda|nedir|tan[ıi]t[ıi]m\w*)\b'
)
GENERAL_INFO_SCOPE_PATTERN = re.compile(
    r'\b(fakülte\w*|fakulte\w*|bölüm\w*|bolum\w*|program\w*)\b'
)
SCOPE_METADATA_FIELDS = (
    'program_alias_text',
    'placement_label',
    'program_title',
    'faculty',
    'unit_name',
    'section_title',
    'page_title',
)


def _chunk_scope_aliases(chunk: ContentChunk) -> set[str]:
    aliases: set[str] = set()
    for field in SCOPE_METADATA_FIELDS:
        aliases.update(_build_scope_aliases(_get_chunk_metadata_value(chunk, field)))
    aliases.update(_build_scope_aliases(chunk.page.title))
    return aliases


def _build_scope_constraint(question: str, chunks: list[ContentChunk]) -> dict | None:
    normalized_question = _normalize_lookup_text(question)
    if not normalized_question:
        return None

    for field in SCOPE_METADATA_FIELDS:
        matched_aliases: dict[int, set[str]] = {}
        for chunk in chunks:
            for alias in _build_scope_aliases(_get_chunk_metadata_value(chunk, field)):
                if _question_mentions_alias(normalized_question, alias):
                    matched_aliases.setdefault(len(alias), set()).add(alias)
        if matched_aliases:
            longest_alias_length = max(matched_aliases)
            return {
                'field': field,
                'aliases': matched_aliases[longest_alias_length],
            }
    for field, aliases in _extract_question_scope_hints(question):
        if aliases:
            longest_alias_length = max(len(alias) for alias in aliases)
            return {
                'field': field,
                'aliases': {alias for alias in aliases if len(alias) == longest_alias_length},
            }

    # Fallback: match distinctive question tokens against chunk metadata
    question_tokens = normalized_question.split()
    distinctive = [t for t in question_tokens if t not in _SCOPE_STOP_WORDS and len(t) > 2]
    for token in distinctive:
        for field in ('program_title', 'faculty'):
            matching_aliases: set[str] = set()
            for chunk in chunks:
                value = _get_chunk_metadata_value(chunk, field)
                normalized_value = _normalize_lookup_text(value)
                if re.search(rf'(^|\s){re.escape(token)}($|\s)', normalized_value):
                    matching_aliases.update(_build_scope_aliases(value))
            if matching_aliases:
                return {'field': field, 'aliases': matching_aliases}

    return None


def _apply_scope_filter(question: str, chunks: list[ContentChunk]) -> tuple[list[ContentChunk], bool]:
    scope = _build_scope_constraint(question, chunks)
    if not scope:
        return chunks, False

    filtered = [
        chunk
        for chunk in chunks
        if _chunk_scope_aliases(chunk) & scope['aliases']
    ]
    return filtered, True


def _build_source_label(chunk: ContentChunk) -> str:
    program_title = _get_chunk_metadata_value(chunk, 'program_title')
    placement_label = _get_chunk_metadata_value(chunk, 'placement_label')
    faculty = _get_chunk_metadata_value(chunk, 'faculty')
    section_title = _get_chunk_metadata_value(chunk, 'section_title') or chunk.page.title

    if placement_label and placement_label != program_title:
        return f'{placement_label} / {section_title}'
    if program_title:
        return f'{program_title} / {section_title}'
    if faculty:
        return f'{faculty} / {section_title}'
    return chunk.page.title


def _get_chunk_source_url(chunk: ContentChunk) -> str:
    return _get_chunk_metadata_value(chunk, 'source_url') or chunk.page.url


def _is_staff_query(question: str) -> bool:
    return bool(STAFF_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_staff_count_query(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    return _is_staff_query(question) and bool(STAFF_COUNT_QUERY_PATTERN.search(normalized_question))


def _is_staff_list_query(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    return bool(re.search(r'\b(kimler|listele\w*|adlar[ıi]|isimler[ıi])\b', normalized_question))


def _is_department_head_query(question: str) -> bool:
    return bool(DEPARTMENT_HEAD_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _requested_staff_role(question: str) -> str:
    normalized_question = _normalize_lookup_text(question)
    if _is_department_head_query(question):
        return 'department_head'
    if DEPUTY_DEAN_QUERY_PATTERN.search(normalized_question):
        return 'deputy_dean'
    if DEAN_QUERY_PATTERN.search(normalized_question):
        return 'dean'
    if DIRECTOR_QUERY_PATTERN.search(normalized_question):
        return 'director'
    return ''


def _is_role_specific_staff_query(question: str) -> bool:
    return bool(_requested_staff_role(question))


def _is_program_exists_query(question: str) -> bool:
    return bool(PROGRAM_EXISTS_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_program_list_query(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    if not PROGRAM_LIST_QUERY_PATTERN.search(normalized_question):
        return False
    return not (
        _is_staff_query(question)
        or _is_score_query(question)
        or _is_fee_query(question)
        or _is_course_query(question)
    )


def _is_engineering_program_list_query(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    return _is_program_list_query(question) and bool(
        re.search(r'\bm[üu]hendislik\b|\bmuhendislik\b', normalized_question)
    )


def _is_engineering_and_natural_sciences_faculty_query(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    return bool(
        re.search(
            r'm[üu]hendislik\s+ve\s+do[ğg]a\s+bilimleri|'
            r'muhendislik\s+ve\s+doga\s+bilimleri',
            normalized_question,
        )
    )


def _is_engineering_and_natural_sciences_chunk(chunk: ContentChunk) -> bool:
    faculty = _normalize_lookup_text(_get_chunk_metadata_value(chunk, 'faculty'))
    title = _normalize_lookup_text(chunk.page.title)
    text = _normalize_lookup_text(chunk.text[:500])
    return (
        'muhendislik ve doga bilimleri' in faculty
        or 'mühendislik ve doğa bilimleri' in faculty
        or 'muhendislik ve doga bilimleri' in title
        or 'mühendislik ve doğa bilimleri' in title
        or 'muhendislik ve doga bilimleri' in text
        or 'mühendislik ve doğa bilimleri' in text
    )


def _is_general_info_query(question: str) -> bool:
    normalized_question = _normalize_lookup_text(question)
    return bool(
        GENERAL_INFO_QUERY_PATTERN.search(normalized_question)
        and GENERAL_INFO_SCOPE_PATTERN.search(normalized_question)
    )


def _is_dentistry_query(question: str) -> bool:
    return bool(DENTISTRY_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_score_query(question: str) -> bool:
    return bool(SCORE_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_fee_query(question: str) -> bool:
    return bool(FEE_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _fee_program_title(question: str) -> str:
    return (
        _extract_program_abbreviation_from_text(question)
        or _extract_known_program_from_text(question)
        or _extract_program_hint_from_text(question)
        or _extract_program_from_tokens(question)
    )


def _is_rank_query(question: str) -> bool:
    return bool(RANK_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_points_query(question: str) -> bool:
    return bool(POINTS_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_quota_query(question: str) -> bool:
    return bool(QUOTA_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_prep_query(question: str) -> bool:
    return bool(PREP_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_course_query(question: str) -> bool:
    if _is_prep_query(question):
        return False
    return bool(COURSE_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _course_program_candidates() -> set[str]:
    titles = (
        ContentChunk.objects.filter(page__is_active=True)
        .filter(
            Q(metadata__chunk_level='program_overview')
            | Q(metadata__chunk_level='semester_plan')
            | Q(metadata__record_type='bologna_program_overview')
            | Q(metadata__record_type='bologna_semester_plan')
        )
        .exclude(metadata__program_title='')
        .values_list('metadata__program_title', flat=True)
        .distinct()
    )
    return {_clean_display_text(title) for title in titles if title}


def _extract_course_program_from_text(text: str) -> str:
    normalized_text = _normalize_lookup_text(text)
    if not normalized_text:
        return ''
    matches: list[str] = []
    for title in _course_program_candidates():
        base_title = _program_initialism_base(title)
        aliases = _build_scope_aliases(title) | _build_scope_aliases(base_title)
        if any(_question_mentions_alias(normalized_text, alias) for alias in aliases):
            matches.append(title)
    if not matches:
        return ''

    wants_english = _mentions_english(text)
    if wants_english:
        english_matches = [
            title for title in matches if _mentions_english(title)
        ]
        if english_matches:
            matches = english_matches
    else:
        non_english_matches = [
            title for title in matches if not _mentions_english(title)
        ]
        if non_english_matches:
            matches = non_english_matches

    return min(matches, key=lambda title: (len(_normalize_lookup_text(title)), title))


def _course_program_title(question: str) -> str:
    return (
        _extract_program_abbreviation_from_text(question)
        or _extract_course_program_from_text(question)
        or _extract_known_program_from_text(question)
        or _extract_program_hint_from_text(question)
        or _extract_program_from_tokens(question)
    )


def _requested_course_period_number(question: str) -> int | None:
    normalized_question = _normalize_lookup_text(question)
    match = re.search(r'\b([1-8])\s*(?:yar[ıi]y[ıi]l|donem|dönem|semester)\b', normalized_question)
    if match:
        return int(match.group(1))
    match = re.search(r'\b(?:yar[ıi]y[ıi]l|donem|dönem|semester)\s*([1-8])\b', normalized_question)
    if match:
        return int(match.group(1))
    ordinal_map = {
        'birinci': 1,
        'ikinci': 2,
        'ucuncu': 3,
        'dorduncu': 4,
        'besinci': 5,
        'altinci': 6,
        'yedinci': 7,
        'sekizinci': 8,
    }
    for word, period_number in ordinal_map.items():
        if re.search(rf'\b{word}\s+(?:yar[ıi]y[ıi]l|donem|dönem|semester)\b', normalized_question):
            return period_number
    return None


def _requested_class_number(question: str) -> int | None:
    normalized_question = _normalize_lookup_text(question)
    match = re.search(r'\b([1-8])\s*(?:sınıf|sinif)\b', normalized_question)
    if match:
        return int(match.group(1))
    match = re.search(r'\b(?:sınıf|sinif)\s*([1-8])\b', normalized_question)
    if match:
        return int(match.group(1))
    return None


def _is_full_course_list_query(question: str) -> bool:
    return bool(COURSE_FULL_LIST_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _mentions_english(value: str) -> bool:
    normalized = _normalize_lookup_text(value)
    return 'ingilizce' in normalized or 'i ngilizce' in normalized


_topic_centroids: dict[str, list[float]] | None = None
_topic_centroids_lock = threading.Lock()
_request_topics_key = '_semantic_request_topics'


def _get_topic_centroids() -> dict[str, list[float]]:
    global _topic_centroids
    if _topic_centroids is not None:
        return _topic_centroids

    with _topic_centroids_lock:
        if _topic_centroids is not None:
            return _topic_centroids

        from scraper.embeddings import embed_texts
        centroids: dict[str, list[float]] = {}
        for topic, descriptions in TOPIC_DESCRIPTIONS.items():
            embeddings = embed_texts(descriptions)
            dim = len(embeddings[0])
            centroid = [0.0] * dim
            for emb in embeddings:
                for i in range(dim):
                    centroid[i] += emb[i]
            n = len(embeddings)
            centroid = [c / n for c in centroid]
            norm = sum(c ** 2 for c in centroid) ** 0.5
            if norm > 0:
                centroid = [c / norm for c in centroid]
            centroids[topic] = centroid

        _topic_centroids = centroids
        return _topic_centroids


def _semantic_question_topics(query_embedding: list[float]) -> set[str]:
    if not settings.SEMANTIC_TOPIC_ENABLED:
        return set()
    centroids = _get_topic_centroids()
    topics: set[str] = set()
    threshold = settings.SEMANTIC_TOPIC_THRESHOLD
    for topic, centroid in centroids.items():
        sim = sum(q * c for q, c in zip(query_embedding, centroid))
        if sim > threshold:
            topics.add(topic)
    return topics


def _set_request_topics(topics: set[str]) -> None:
    setattr(_thread_local, _request_topics_key, topics)


def _clear_request_topics() -> None:
    setattr(_thread_local, _request_topics_key, None)


_HIGH_CONFIDENCE_ANSWER_TYPES = frozenset({'staff', 'program', 'score', 'fee', 'course', 'general_info'})


def _should_bypass_structured_answer(answer_type: str) -> bool:
    if settings.ANSWER_MODE == 'llm_preferred':
        return True
    if settings.ANSWER_MODE == 'hybrid':
        return answer_type not in _HIGH_CONFIDENCE_ANSWER_TYPES
    return False


def _question_topics(question: str) -> set[str]:
    cached = getattr(_thread_local, _request_topics_key, None)
    if cached is not None:
        return cached
    return _regex_question_topics(question)


def _regex_question_topics(question: str) -> set[str]:
    normalized_question = _normalize_lookup_text(question)
    topics: set[str] = set()
    if SCHOLARSHIP_QUERY_PATTERN.search(normalized_question):
        topics.add('scholarships')
    if DORMITORY_QUERY_PATTERN.search(normalized_question):
        topics.add('dormitory')
    if INTERNATIONAL_QUERY_PATTERN.search(normalized_question):
        topics.add('international')
    if DOUBLE_MAJOR_MINOR_QUERY_PATTERN.search(normalized_question):
        topics.add('double_major_minor')
    if LIBRARY_QUERY_PATTERN.search(normalized_question):
        topics.add('library')
    if SPORTS_QUERY_PATTERN.search(normalized_question):
        topics.add('sports')
    if CAMPUS_LIFE_QUERY_PATTERN.search(normalized_question):
        topics.add('campus_life')
    if STUDENT_CLUB_QUERY_PATTERN.search(normalized_question):
        topics.add('student_clubs')
    if TRANSPORT_QUERY_PATTERN.search(normalized_question):
        topics.add('transport')
    if PREP_QUERY_PATTERN.search(normalized_question):
        topics.add('prep')
    if COMPARISON_QUERY_PATTERN.search(normalized_question):
        topics.add('comparison')
    return topics


def _query_expansion_topics(question: str) -> set[str]:
    if not settings.RAG_QUERY_EXPANSION_ENABLED:
        return set()
    if (
        _is_staff_query(question)
        or _is_fee_query(question)
        or _is_score_query(question)
        or _is_course_query(question)
        or _is_program_list_query(question)
    ):
        return set()
    topics = _question_topics(question)
    if topics:
        return topics
    if _is_general_info_query(question):
        return topics
    return set()


def _expanded_query_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for topic in _query_expansion_topics(question):
        terms.update(QUERY_EXPANSIONS.get(topic, ()))
    return {term for term in terms if term}


def _expanded_keyword_query_text(question: str) -> str:
    terms = sorted(_expanded_query_terms(question), key=lambda term: _normalize_lookup_text(term))
    if not terms:
        return question
    return ' '.join([question, *terms])


def _chunk_matches_question_topic(question_topics: set[str], chunk: ContentChunk) -> bool:
    if not question_topics:
        return False

    topic = _get_chunk_metadata_value(chunk, 'topic')
    if topic in question_topics:
        return True

    source_group = _get_chunk_metadata_value(chunk, 'source_group')
    searchable_text = _normalize_lookup_text(
        ' '.join(
            [
                chunk.page.title,
                _get_chunk_metadata_value(chunk, 'topic_label'),
                _get_chunk_metadata_value(chunk, 'section_title'),
                _get_chunk_metadata_value(chunk, 'source_url'),
                chunk.text[:1000],
            ]
        )
    )
    for question_topic in question_topics:
        if source_group in _TOPIC_EXCLUDED_SOURCE_GROUPS.get(question_topic, frozenset()):
            continue
        keywords = _TOPIC_MATCH_TERMS.get(
            question_topic,
            TOPIC_KEYWORDS.get(question_topic, ()) + QUERY_EXPANSIONS.get(question_topic, ()),
        )
        for keyword in keywords:
            if _normalize_lookup_text(keyword) in searchable_text:
                return True
    return False


_NON_ACADEMIC_TOPICS = frozenset({
    'campus_life', 'student_clubs', 'sports', 'library', 'dormitory',
    'international', 'scholarships', 'transport', 'prep',
})
_BOLOGNA_ACADEMIC_KINDS = frozenset({'bologna_staff_page', 'bologna_program_page'})
_BOLOGNA_ACADEMIC_RECORD_TYPES = frozenset({
    'academic_staff_member', 'program_yeterlikleri',
    'bologna_program_overview', 'bologna_semester_plan',
})


def _is_chunk_excluded_for_topics(topics: set[str], chunk: ContentChunk) -> bool:
    if not (topics & _NON_ACADEMIC_TOPICS):
        return False
    if chunk.page.source != 'bologna':
        return False
    kind = _get_chunk_metadata_value(chunk, 'kind')
    record_type = _get_chunk_metadata_value(chunk, 'record_type')
    if kind in _BOLOGNA_ACADEMIC_KINDS or record_type in _BOLOGNA_ACADEMIC_RECORD_TYPES:
        return True
    return False


def _chunk_priority(question: str, chunk: ContentChunk) -> int:
    kind = _get_chunk_metadata_value(chunk, 'kind')
    record_type = _get_chunk_metadata_value(chunk, 'record_type')
    chunk_level = _get_chunk_metadata_value(chunk, 'chunk_level')
    staff_page_type = _get_chunk_metadata_value(chunk, 'staff_page_type')
    topic = _get_chunk_metadata_value(chunk, 'topic')
    staff_count_text = _get_chunk_metadata_value(chunk, 'staff_count')
    try:
        staff_count = int(staff_count_text) if staff_count_text else 0
    except ValueError:
        staff_count = 0

    if _is_staff_query(question):
        if record_type == 'staff_role_assignment' or kind == 'main_site_role_page':
            return 0
        if record_type == 'academic_staff_member':
            if _is_role_specific_staff_query(question):
                return 4
            return 0
        if record_type == 'department_head_message':
            return 0
        if kind == 'bologna_staff_page':
            if staff_page_type == 'academic_staff' and staff_count > 1:
                return 0
            if staff_page_type == 'officials':
                return 2
            return 1
        if kind == 'main_site_staff_page':
            return 2
        if kind == 'bologna_program_page':
            return 4
        return 5

    if _is_general_info_query(question):
        if kind == 'main_site_page' and _get_chunk_metadata_value(chunk, 'source_group') == 'department':
            return 0
        if kind == 'bologna_program_page':
            return 1
        if kind == 'main_site_staff_page' or record_type == 'academic_staff_member':
            return 4
        return 2

    if _is_score_query(question):
        if kind == 'structured_admissions_score' or record_type == 'quota_row':
            return 0
        if kind == 'candidate_topic_page' and topic == 'admissions_scores':
            return 1
        if kind == 'structured_admissions_fee' or record_type == 'tuition_fee':
            return 4
        return 5

    if _is_fee_query(question):
        if kind == 'structured_admissions_fee' or record_type == 'tuition_fee':
            return 0
        if kind == 'candidate_topic_page' and topic == 'tuition':
            return 1
        if kind == 'structured_admissions_score' or record_type == 'quota_row':
            return 4
        return 5

    if _is_course_query(question):
        requested_period = _requested_course_period_number(question)
        if chunk_level == 'semester_plan':
            period_number = _get_chunk_metadata_value(chunk, 'period_number')
            if requested_period is not None and str(period_number) == str(requested_period):
                return 0
            if requested_period is None and _is_full_course_list_query(question):
                return 1
            if requested_period is None:
                return 2
            return 3
        if chunk_level == 'program_overview':
            return 0 if requested_period is None else 1
        if kind == 'bologna_program_page':
            return 2
        return 5

    if _is_program_list_query(question):
        if record_type == 'bologna_program_overview':
            return 0
        if kind == 'bologna_program_page' and _get_chunk_metadata_value(chunk, 'program_title'):
            return 1
        if kind == 'main_site_page' and _get_chunk_metadata_value(chunk, 'program_title') == 'Bölümler':
            return 1
        if kind in {'structured_admissions_score', 'structured_admissions_fee', 'main_site_staff_page'}:
            return 5
        if record_type in {'quota_row', 'tuition_fee', 'academic_staff_member'}:
            return 5
        return 3

    question_topics = _question_topics(question)
    if question_topics:
        if _chunk_matches_question_topic(question_topics, chunk):
            return 0
        return 5

    return 0


def _sort_candidate_chunks(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    deduped = _dedupe_chunks(chunks)
    return [
        chunk
        for _index, chunk in sorted(
            enumerate(deduped),
            key=lambda item: (_chunk_priority(question, item[1]), item[0]),
        )
    ]


def _program_list_chunk_sort_key(question: str, chunk: ContentChunk) -> tuple[int, str, str]:
    record_type = _get_chunk_metadata_value(chunk, 'record_type')
    kind = _get_chunk_metadata_value(chunk, 'kind')
    title = chunk.page.title
    if record_type == 'bologna_program_overview':
        priority = 0
    elif 'Programı Bilgileri' in title or 'Programi Bilgileri' in title:
        priority = 1
    elif kind == 'bologna_program_page' and _get_chunk_metadata_value(chunk, 'program_title'):
        priority = 2
    elif kind == 'main_site_page' and _get_chunk_metadata_value(chunk, 'program_title') == 'Bölümler':
        priority = 3
    else:
        priority = 4
    return (
        priority,
        _normalize_lookup_text(_get_chunk_metadata_value(chunk, 'program_title') or title),
        title,
    )


def _program_list_chunks_for_context(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    if not _is_program_list_query(question):
        return chunks

    selected_by_program: dict[str, ContentChunk] = {}
    list_chunks: list[ContentChunk] = []
    for chunk in sorted(chunks, key=lambda item: _program_list_chunk_sort_key(question, item)):
        program_title = _get_chunk_metadata_value(chunk, 'program_title')
        normalized_program = _normalize_lookup_text(program_title)
        if _is_engineering_program_list_query(question) and not _is_engineering_and_natural_sciences_faculty_query(question):
            if not _is_engineering_and_natural_sciences_chunk(chunk):
                continue
            if not re.search(r'\bm[üu]hendisli[ğg]i\b|\bmuhendisligi\b', normalized_program):
                continue
        if program_title and program_title != 'Bölümler':
            selected_by_program.setdefault(normalized_program, chunk)
            continue
        if _get_chunk_metadata_value(chunk, 'program_title') == 'Bölümler':
            list_chunks.append(chunk)

    selected = list(selected_by_program.values())
    if selected:
        return selected[:settings.RAG_RETRIEVE_LIMIT]
    return list_chunks[:settings.RAG_RETRIEVE_LIMIT] or chunks


def _filter_candidates_for_query(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    if _is_program_list_query(question):
        program_chunks = []
        for chunk in chunks:
            kind = _get_chunk_metadata_value(chunk, 'kind')
            record_type = _get_chunk_metadata_value(chunk, 'record_type')
            program_title = _get_chunk_metadata_value(chunk, 'program_title')
            if kind in {'structured_admissions_score', 'structured_admissions_fee', 'main_site_staff_page'}:
                continue
            if record_type in {'quota_row', 'tuition_fee', 'academic_staff_member'}:
                continue
            if _get_chunk_metadata_value(chunk, 'source_group') in {'quota', 'tuition'}:
                continue
            if _get_chunk_metadata_value(chunk, 'record_type') == 'cap_combination':
                continue
            if _is_engineering_program_list_query(question):
                if not _is_engineering_and_natural_sciences_chunk(chunk):
                    continue
                if program_title and not re.search(r'\bm[üu]hendisli[ğg]i\b|\bmuhendisligi\b', _normalize_lookup_text(program_title)):
                    continue
                if not program_title and not re.search(r'\bm[üu]hendisli[ğg]i\b|\bmuhendisligi\b', _normalize_lookup_text(chunk.text)):
                    continue
            if (
                record_type == 'bologna_program_overview'
                or (kind == 'bologna_program_page' and program_title)
                or (
                    kind == 'main_site_page'
                    and _get_chunk_metadata_value(chunk, 'program_title') == 'Bölümler'
                )
            ):
                program_chunks.append(chunk)
        if program_chunks:
            return sorted(program_chunks, key=lambda chunk: _chunk_priority(question, chunk))

    if _is_staff_query(question):
        role_chunks = [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'kind') == 'main_site_role_page'
            or _get_chunk_metadata_value(chunk, 'record_type')
            in {'department_head_message', 'staff_role_assignment'}
        ]
        if _is_role_specific_staff_query(question) and role_chunks:
            return role_chunks

        staff_chunks = [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'kind')
            in {'bologna_staff_page', 'main_site_staff_page', 'main_site_role_page'}
            or _get_chunk_metadata_value(chunk, 'record_type')
            in {'academic_staff_member', 'department_head_message', 'staff_role_assignment'}
        ]
        if staff_chunks:
            return staff_chunks

    if _is_score_query(question):
        return [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'kind') == 'structured_admissions_score'
            or _get_chunk_metadata_value(chunk, 'record_type') == 'quota_row'
            or (
                _get_chunk_metadata_value(chunk, 'kind') == 'candidate_topic_page'
                and _get_chunk_metadata_value(chunk, 'topic') == 'admissions_scores'
            )
        ]

    if _is_fee_query(question):
        return [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'kind') == 'structured_admissions_fee'
            or _get_chunk_metadata_value(chunk, 'record_type') == 'tuition_fee'
            or (
                _get_chunk_metadata_value(chunk, 'kind') == 'candidate_topic_page'
                and _get_chunk_metadata_value(chunk, 'topic') == 'tuition'
            )
        ]

    if _is_course_query(question):
        filtered = [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'chunk_level') in {'semester_plan', 'program_overview'}
            or chunk.page.source == 'bologna'
            or _get_chunk_metadata_value(chunk, 'curriculum_year')
            or _get_chunk_metadata_value(chunk, 'period_label')
        ]
        if filtered:
            return sorted(filtered, key=lambda chunk: _course_sort_key(question, chunk))

    question_topics = _question_topics(question)
    if question_topics:
        return [
            chunk
            for chunk in chunks
            if _chunk_matches_question_topic(question_topics, chunk)
            and not _is_chunk_excluded_for_topics(question_topics, chunk)
        ]

    return chunks


def _question_specific_prompt_rules(question: str) -> list[str]:
    rules: list[str] = []
    if _is_score_query(question):
        rules.extend(
            [
                'Kontenjan, puan ve sıralama sorularında yalnızca aday öğrenci admissions kaynaklarını kullan.',
                'Kullanıcı sadece "sıralama" dediyse bunu "taban başarı sırası" olarak yorumla.',
                'Aynı program için birden fazla yerleşim tipi varsa ve kullanıcı birini belirtmediyse her yerleşim tipini ayrı satırda listele.',
                'Bir yerleşim tipinde istenen alan eksikse sadece o yerleşim tipi için bilginin kaynakta yer almadığını belirt.',
            ]
        )
    elif _is_staff_query(question):
        rules.extend(
            [
                'Akademik kadro ve yönetici sorularında yalnızca personel kaynaklarını kullan.',
                'Bölüm başkanı, dekan veya müdür sorularında "Program Yetkilileri" ve "Akademik Kadro" kaynaklarını önceliklendir.',
                'Bir kişinin görevini belirtirken kaynakta geçtiği gibi yaz.',
            ]
        )
    elif _is_fee_query(question):
        rules.append('Ücret sorularında yalnızca resmi öğrenim ücreti veya ilgili aday öğrenci kaynaklarını kullan.')
    elif _is_course_query(question):
        rules.append('Ders planı ve AKTS sorularında öncelikle müfredat yılı ve dönem metadata bilgilerini dikkate al.')
    elif _is_program_list_query(question):
        rules.append(
            'Bölüm veya program listeleme sorularında yalnızca verilen program listesi ve program overview kaynaklarındaki distinct program adlarını listele.'
        )
        rules.append(
            'Kontenjan-puan, akademik kadro, ücret veya yandal kaynaklarından bölüm/program varlığı sonucu çıkarma.'
        )
    elif _question_topics(question):
        topics = _question_topics(question)
        if 'student_clubs' in topics:
            rules.append('Öğrenci kulübü sorularında yalnızca kulüp liste veya kulüp detay kaynaklarını kullan.')
        elif topics & {'library', 'sports', 'campus_life'}:
            rules.append('Kütüphane, spor merkezi ve kampüs olanakları sorularında yalnızca ilgili tesis kaynaklarını kullan.')
            if 'campus_life' in topics:
                rules.append('Kampüs olanakları sorularında yurt, spor, kütüphane, yemekhane ve sosyal tesis bilgilerini bağlamda varsa birleştirerek cevapla.')
        elif topics & {'transport', 'prep'}:
            rules.append('Yalnızca ilgili konuya ait sayfa kaynaklarını kullan, akademik Bologna sayfalarını kullanma.')
        else:
            rules.append('Burs, yurt, Erasmus, uluslararası olanak veya ÇAP-yandal sorularında yalnızca ilgili konu kaynaklarını kullan.')
    return rules


def _build_structured_score_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_score_query(question):
        return ''

    score_chunks = [
        chunk
        for chunk in chunks
        if _get_chunk_metadata_value(chunk, 'kind') == 'structured_admissions_score'
    ]
    if not score_chunks:
        return ''

    wants_rank = _is_rank_query(question)
    wants_points = _is_points_query(question)
    wants_quota = _is_quota_query(question)
    if not any((wants_rank, wants_points, wants_quota)):
        wants_rank = True
        wants_points = True
        wants_quota = True

    def _sort_key(chunk: ContentChunk) -> tuple[int, str]:
        placement_type = _normalize_lookup_text(_get_chunk_metadata_value(chunk, 'placement_type'))
        priority = {
            'burslu': 0,
            '%25 indirimli': 1,
            '%50 indirimli': 2,
            'ucretli': 3,
            'ücretli': 3,
        }.get(placement_type, 9)
        return priority, _get_chunk_metadata_value(chunk, 'placement_label') or chunk.page.title

    program_titles = {
        _clean_display_text(_get_chunk_metadata_value(chunk, 'program_title'))
        for chunk in score_chunks
        if _get_chunk_metadata_value(chunk, 'program_title')
    }
    if len(program_titles) == 1:
        intro = f'{next(iter(program_titles))} için resmi aday öğrenci kaynağındaki bilgiler:'
    else:
        intro = 'Resmi aday öğrenci kaynağındaki bilgiler:'

    lines = [intro]
    for chunk in sorted(score_chunks, key=_sort_key):
        placement = _get_chunk_metadata_value(chunk, 'placement_label') or _get_chunk_metadata_value(
            chunk, 'program_title'
        ) or chunk.page.title
        placement = _clean_display_text(placement)
        base_rank = _get_chunk_metadata_value(chunk, 'base_rank')
        base_score = _get_chunk_metadata_value(chunk, 'base_score')
        quota = _get_chunk_metadata_value(chunk, 'quota')
        score_type = _get_chunk_metadata_value(chunk, 'score_type')

        metrics: list[str] = []
        if wants_quota:
            metrics.append(
                f'kontenjan {quota}' if quota else 'kontenjan bilgisi kaynakta yer almıyor'
            )
            if score_type:
                metrics.append(f'puan türü {score_type}')
        if wants_points or (wants_rank and not wants_quota):
            metrics.append(
                f'taban puan {base_score}'
                if base_score
                else 'taban puan bilgisi kaynakta yer almıyor'
            )
        if wants_rank or (wants_points and not wants_quota):
            metrics.append(
                f'taban başarı sırası {base_rank}'
                if base_rank
                else 'taban başarı sırası bilgisi kaynakta yer almıyor'
            )
        if not metrics:
            metrics.extend(
                metric
                for metric in (
                    f'kontenjan {quota}' if quota else '',
                    f'taban puan {base_score}' if base_score else '',
                    f'taban başarı sırası {base_rank}' if base_rank else '',
                )
                if metric
            )
        if not metrics:
            metrics.append('istenen admissions bilgisi kaynakta yer almıyor')
        lines.append(f'- {placement}: {", ".join(metrics)}.')
    return '\n'.join(lines)


def _chunk_has_fee_amount(chunk: ContentChunk) -> bool:
    return any(
        _get_chunk_metadata_value(chunk, key)
        for key in ('fee_full', 'fee_25', 'fee_50', 'fee_kav_support')
    )


def _chunk_matches_program_title(chunk: ContentChunk, program_title: str) -> bool:
    if not program_title:
        return False
    searchable_text = ' '.join(
        value
        for value in (
            _get_chunk_metadata_value(chunk, 'program_title'),
            _get_chunk_metadata_value(chunk, 'placement_label'),
            _get_chunk_metadata_value(chunk, 'unit_name'),
            _get_chunk_metadata_value(chunk, 'program_alias_text'),
            chunk.page.title,
            chunk.text[:1000],
        )
        if value
    )
    return any(
        _normalized_contains(searchable_text, term)
        for term in _program_lookup_terms(program_title)
    )


def _sort_fee_chunks(chunks: list[ContentChunk]) -> list[ContentChunk]:
    return sorted(
        chunks,
        key=lambda item: (
            _get_chunk_metadata_value(item, 'program_title') or item.page.title,
            item.page.title,
        ),
    )


def _fee_chunks_for_answer(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    fee_chunks = [
        chunk
        for chunk in chunks
        if (
            _get_chunk_metadata_value(chunk, 'kind') == 'structured_admissions_fee'
            or _get_chunk_metadata_value(chunk, 'record_type') == 'tuition_fee'
        )
        and _chunk_has_fee_amount(chunk)
    ]
    program_title = _fee_program_title(question)
    if program_title:
        scoped_chunks = [
            chunk for chunk in fee_chunks if _chunk_matches_program_title(chunk, program_title)
        ]
        if scoped_chunks:
            fee_chunks = scoped_chunks
    return _sort_fee_chunks(fee_chunks)


def _build_structured_fee_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_fee_query(question):
        return ''

    fee_chunks = _fee_chunks_for_answer(question, chunks)
    if not fee_chunks:
        return ''

    program_titles = {
        _clean_display_text(_get_chunk_metadata_value(chunk, 'program_title'))
        for chunk in fee_chunks
        if _get_chunk_metadata_value(chunk, 'program_title')
    }
    if len(program_titles) == 1:
        intro = f'{next(iter(program_titles))} için resmi öğrenim ücreti bilgileri:'
    else:
        intro = 'Resmi öğrenim ücreti bilgileri:'

    lines = [intro]
    for chunk in fee_chunks:
        label = _get_chunk_metadata_value(chunk, 'program_title') or chunk.page.title
        metrics: list[str] = []
        for title, key in (
            ('ücretli', 'fee_full'),
            ('%25 indirimli', 'fee_25'),
            ('%50 indirimli', 'fee_50'),
            ('ilave %25 KAV destek burslu', 'fee_kav_support'),
        ):
            value = _get_chunk_metadata_value(chunk, key)
            if value:
                metrics.append(f'{title} {value}')
        notes = _get_chunk_metadata_value(chunk, 'notes')
        if notes:
            cleaned_notes = _clean_note_text(notes)
            if cleaned_notes:
                metrics.append(f'notlar: {cleaned_notes}')
        if not metrics:
            metrics.append('ücret bilgisi kaynakta yer almıyor')
        lines.append(f'- {_clean_display_text(label)}: {", ".join(metrics)}.')
    return '\n'.join(lines)


def _is_course_chunk(chunk: ContentChunk) -> bool:
    return (
        _get_chunk_metadata_value(chunk, 'chunk_level') in {'program_overview', 'semester_plan'}
        or _get_chunk_metadata_value(chunk, 'record_type')
        in {'bologna_program_overview', 'bologna_semester_plan'}
    )


def _course_curriculum_year(chunk: ContentChunk) -> int:
    try:
        return int(_get_chunk_metadata_value(chunk, 'curriculum_year') or 0)
    except ValueError:
        return 0


def _latest_course_year(chunks: list[ContentChunk]) -> int:
    return max((_course_curriculum_year(chunk) for chunk in chunks), default=0)


def _course_sort_key(question: str, chunk: ContentChunk) -> tuple[int, int, str]:
    requested_period = _requested_course_period_number(question)
    requested_class = _requested_class_number(question)
    chunk_level = _get_chunk_metadata_value(chunk, 'chunk_level')
    period_number_text = _get_chunk_metadata_value(chunk, 'period_number')
    try:
        period_number = int(period_number_text) if period_number_text else 99
    except ValueError:
        period_number = 99

    if requested_period is not None and period_number == requested_period:
        priority = 0
    elif requested_class is not None:
        semester_1 = 2 * requested_class - 1
        semester_2 = 2 * requested_class
        if period_number in (semester_1, semester_2, requested_class):
            priority = 0
        elif chunk_level == 'program_overview':
            priority = 1
        elif chunk_level == 'semester_plan':
            priority = 2
        else:
            priority = 3
    elif chunk_level == 'program_overview':
        priority = 0 if requested_period is None else 1
    elif chunk_level == 'semester_plan':
        priority = 1 if _is_full_course_list_query(question) else 2
    else:
        priority = 3
    return priority, period_number, chunk.page.title


def _course_chunks_for_answer(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    course_chunks = [chunk for chunk in chunks if _is_course_chunk(chunk)]
    if not course_chunks:
        return []

    program_title = _course_program_title(question)
    if program_title:
        scoped_chunks = [
            chunk for chunk in course_chunks if _chunk_matches_program_title(chunk, program_title)
        ]
        if scoped_chunks:
            course_chunks = scoped_chunks
        normalized_program_title = _normalize_lookup_text(program_title)
        exact_chunks = [
            chunk
            for chunk in course_chunks
            if _normalize_lookup_text(_get_chunk_metadata_value(chunk, 'program_title'))
            == normalized_program_title
        ]
        if not exact_chunks:
            exact_chunks = [
                chunk
                for chunk in course_chunks
                if _normalize_lookup_text(
                    _program_initialism_base(_get_chunk_metadata_value(chunk, 'program_title'))
                )
                == normalized_program_title
            ]
        if exact_chunks:
            course_chunks = exact_chunks

        wants_english = _mentions_english(program_title)
        non_english_chunks = [
            chunk
            for chunk in course_chunks
            if not _mentions_english(_get_chunk_metadata_value(chunk, 'program_title'))
        ]
        if not wants_english and non_english_chunks:
            course_chunks = non_english_chunks

    latest_year = _latest_course_year(course_chunks)
    if latest_year:
        course_chunks = [
            chunk for chunk in course_chunks if _course_curriculum_year(chunk) == latest_year
        ]

    requested_period = _requested_course_period_number(question)
    if requested_period is not None:
        semester_chunks = [
            chunk
            for chunk in course_chunks
            if _get_chunk_metadata_value(chunk, 'chunk_level') == 'semester_plan'
            and _get_chunk_metadata_value(chunk, 'period_number') == str(requested_period)
        ]
        if semester_chunks:
            return sorted(semester_chunks, key=lambda chunk: chunk.page.title)

    requested_class = _requested_class_number(question)
    if requested_class is not None:
        samples = [chunk for chunk in course_chunks
                    if _get_chunk_metadata_value(chunk, 'chunk_level') == 'semester_plan']
        first_label = _normalize_lookup_text(
            _get_chunk_metadata_value(samples[0], 'period_label') if samples else ''
        )
        uses_class_labels = bool(first_label and re.search(r'\bs(ı|i)n(ı|i)f\b', first_label))

        if uses_class_labels:
            class_chunks = [
                chunk
                for chunk in course_chunks
                if _get_chunk_metadata_value(chunk, 'chunk_level') == 'semester_plan'
                and _get_chunk_metadata_value(chunk, 'period_number') == str(requested_class)
            ]
            if class_chunks:
                return sorted(class_chunks, key=lambda chunk: chunk.page.title)
        else:
            semester_1 = 2 * requested_class - 1
            semester_2 = 2 * requested_class
            class_chunks = [
                chunk
                for chunk in course_chunks
                if _get_chunk_metadata_value(chunk, 'chunk_level') == 'semester_plan'
                and _get_chunk_metadata_value(chunk, 'period_number') in (str(semester_1), str(semester_2))
            ]
            if class_chunks:
                return sorted(
                    class_chunks,
                    key=lambda chunk: int(_get_chunk_metadata_value(chunk, 'period_number') or '0'),
                )

    return sorted(course_chunks, key=lambda chunk: _course_sort_key(question, chunk))


def _metadata_number(chunk: ContentChunk, key: str) -> str:
    value = _get_chunk_metadata_value(chunk, key)
    return value if value else ''


def _overview_summary_lines(chunk: ContentChunk) -> list[str]:
    lines: list[str] = []
    course_count = _metadata_number(chunk, 'course_count')
    period_count = _metadata_number(chunk, 'period_count')
    total_ects = _metadata_number(chunk, 'total_ects_sum')
    if period_count:
        lines.append(f'Dönem sayısı: {period_count}')
    if course_count:
        lines.append(f'Toplam ders sayısı: {course_count}')
    if total_ects:
        lines.append(f'Toplam AKTS: {total_ects}')

    for raw_line in chunk.text.splitlines():
        line = _clean_display_text(raw_line)
        if re.search(r'^\-\s*\d+\.\s*yar[ıi]y[ıi]l\s+ders\s+plan[ıi]\s*:', line, flags=re.IGNORECASE):
            lines.append(line)
    return lines


def _semester_course_lines(chunk: ContentChunk) -> list[str]:
    lines = []
    for raw_line in chunk.text.splitlines():
        line = _clean_display_text(raw_line)
        if line.startswith('- '):
            lines.append(line)
    return lines


def _build_structured_course_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_course_query(question):
        return ''

    course_chunks = _course_chunks_for_answer(question, chunks)
    if not course_chunks:
        program_title = _course_program_title(question)
        if program_title:
            return f'{program_title} için doğrulanmış ders planı kaynağı bulamadım.'
        return ''

    first_chunk = course_chunks[0]
    program_title = _clean_display_text(
        _get_chunk_metadata_value(first_chunk, 'program_title') or first_chunk.page.title
    )
    curriculum_year = _get_chunk_metadata_value(first_chunk, 'curriculum_year')
    requested_period = _requested_course_period_number(question)
    requested_class = _requested_class_number(question)

    if requested_period is not None:
        semester_chunk = next(
            (
                chunk
                for chunk in course_chunks
                if _get_chunk_metadata_value(chunk, 'chunk_level') == 'semester_plan'
            ),
            first_chunk,
        )
        period_label = _clean_display_text(
            _get_chunk_metadata_value(semester_chunk, 'period_label')
            or _get_chunk_metadata_value(semester_chunk, 'section_title')
            or f'{requested_period}. Yarıyıl Ders Planı'
        )
        lines = [
            f'{program_title} {curriculum_year} müfredatında {period_label} dersleri:'
        ]
        course_lines = _semester_course_lines(semester_chunk)
        if not course_lines:
            return f'{program_title} için {period_label} ders listesi kaynakta yer almıyor.'
        lines.extend(course_lines)
        return '\n'.join(lines)

    if requested_class is not None:
        first_label = _normalize_lookup_text(
            _get_chunk_metadata_value(course_chunks[0], 'period_label') or ''
        )
        uses_class_labels = bool(first_label and re.search(r'\bs(ı|i)n(ı|i)f\b', first_label))

        if uses_class_labels:
            chunk = course_chunks[0]
            period_label = _clean_display_text(
                _get_chunk_metadata_value(chunk, 'period_label')
                or f'{requested_class}. Sınıf Ders Planı'
            )
            lines = [f'{program_title} {curriculum_year} müfredatında {period_label} dersleri:']
            course_lines = _semester_course_lines(chunk)
            if course_lines:
                lines.extend(course_lines)
            else:
                return f'{program_title} için {period_label} ders listesi kaynakta yer almıyor.'
            return '\n'.join(lines)
        else:
            lines = [f'{program_title} {curriculum_year} müfredatında {requested_class}. Sınıf dersleri:']
            for chunk in course_chunks:
                period_label = _clean_display_text(
                    _get_chunk_metadata_value(chunk, 'period_label')
                    or _get_chunk_metadata_value(chunk, 'section_title')
                )
                if period_label:
                    lines.append(f'\n{period_label}:')
                lines.extend(_semester_course_lines(chunk))
            return '\n'.join(lines)

    overview_chunk = next(
        (
            chunk
            for chunk in course_chunks
            if _get_chunk_metadata_value(chunk, 'chunk_level') == 'program_overview'
        ),
        course_chunks[0],
    )
    if _is_full_course_list_query(question):
        lines = [f'{program_title} {curriculum_year} müfredatındaki dersler:']
        semester_chunks = [
            chunk
            for chunk in course_chunks
            if _get_chunk_metadata_value(chunk, 'chunk_level') == 'semester_plan'
        ]
        if not semester_chunks:
            semester_chunks = [overview_chunk]
        for semester_chunk in semester_chunks:
            period_label = _clean_display_text(
                _get_chunk_metadata_value(semester_chunk, 'period_label')
                or _get_chunk_metadata_value(semester_chunk, 'section_title')
            )
            if period_label:
                lines.append(period_label)
            lines.extend(_semester_course_lines(semester_chunk))
        return '\n'.join(lines)

    lines = [f'{program_title} {curriculum_year} müfredat özeti:']
    summary_lines = _overview_summary_lines(overview_chunk)
    if summary_lines:
        lines.extend(f'- {line.lstrip("- ").strip()}' for line in summary_lines)
    else:
        lines.append('- Ders planı özeti kaynakta yer alıyor; dönem detayları için yarıyıl belirtebilirsin.')
    return '\n'.join(lines)


def _general_info_excerpt(chunk: ContentChunk) -> str:
    text = _clean_display_text(chunk.text)
    title = _clean_display_text(
        _get_chunk_metadata_value(chunk, 'section_title')
        or _get_chunk_metadata_value(chunk, 'page_title')
        or chunk.page.title
    )
    markers = [
        f'{title} Hakkında',
        'Hakkında',
        f'{title} Hakkinda',
        'Hakkinda',
    ]
    for marker in markers:
        index = text.casefold().find(marker.casefold())
        if index >= 0:
            text = text[index + len(marker):].strip(' :-')
            break

    sentences = [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', text)
        if len(sentence.strip()) > 20
    ]
    excerpt = ' '.join(sentences[:2]).strip()
    if excerpt:
        return excerpt
    return text[:500].rsplit(' ', 1)[0].strip()


def _build_general_info_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_general_info_query(question):
        return ''

    info_chunks = [
        chunk
        for chunk in chunks
        if _get_chunk_metadata_value(chunk, 'kind') == 'main_site_page'
        and _get_chunk_metadata_value(chunk, 'source_group') == 'department'
    ]
    if not info_chunks:
        return ''

    chunk = info_chunks[0]
    title = _clean_display_text(
        _get_chunk_metadata_value(chunk, 'section_title')
        or _get_chunk_metadata_value(chunk, 'page_title')
        or chunk.page.title
    )
    excerpt = _general_info_excerpt(chunk)
    if not excerpt:
        return ''
    return f'{title} hakkında resmi kaynakta şu bilgi yer alıyor: {excerpt} [1]'


def _extract_facility_excerpts(chunks: list[ContentChunk], keyword_pattern: re.Pattern) -> list[str]:
    excerpts: list[str] = []
    seen_chars: set[str] = set()
    for chunk in chunks:
        text = ' '.join(_clean_display_text(chunk.text).split())
        if not text or text[:80] in seen_chars:
            continue
        if _looks_like_navigation(text):
            continue
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
        relevant = [s for s in sentences if keyword_pattern.search(_normalize_lookup_text(s))]
        if relevant:
            seen_chars.add(text[:80])
            excerpt = ' '.join(relevant[:3])
            if excerpt:
                excerpts.append(excerpt)
    return excerpts[:3]


def _looks_like_navigation(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 5:
        return False
    capital_words = sum(1 for t in tokens if t and t[0].isupper())
    if capital_words > len(tokens) * 0.5:
        return True
    if len(text) > 200 and text.count('.') == 0 and text.count('?') == 0:
        return True
    return False


_FACILITY_NOISE_PAGE_PATTERNS = re.compile(
    r'sürdürülebilir|i̇klim|sürdürülebilirlik|iklim|başarı\s*sıras'
    r'|burs\b|bursu|öğrenim\s*ücret|i̇ndirim|öden|kontenjan'
    r'|acu-burs|aydinlar\s*ünivers',
    re.IGNORECASE,
)
_FACILITY_NOISE_KINDS = frozenset({
    'structured_admissions_score',
    'structured_admissions_fee',
})
_FACILITY_NOISE_RECORD_TYPES = frozenset({
    'scholarship_rule',
    'candidate_requirement',
    'tuition_fee',
    'quota_row',
})


def _build_facility_info_answer(question: str, chunks: list[ContentChunk]) -> tuple[str, list[ContentChunk]]:
    topics = _question_topics(question)
    facility_topics = topics & {'library', 'sports', 'campus_life', 'student_clubs'}
    if not facility_topics:
        return '', []

    library_chunks = []
    sports_chunks = []
    campus_chunks = []
    club_chunks = []
    used_chunks: list[ContentChunk] = []
    for chunk in chunks:
        searchable = _normalize_lookup_text(chunk.text[:1000])
        page_title_normalized = _normalize_lookup_text(chunk.page.title)
        if _FACILITY_NOISE_PAGE_PATTERNS.search(page_title_normalized):
            continue
        kind = _get_chunk_metadata_value(chunk, 'kind')
        if kind in _FACILITY_NOISE_KINDS:
            continue
        record_type = _get_chunk_metadata_value(chunk, 'record_type')
        if record_type in _FACILITY_NOISE_RECORD_TYPES:
            continue
        if 'library' in facility_topics and re.search(r'\bk[uü]t[uü]phane', _normalize_lookup_text(chunk.page.title)):
            library_chunks.append(chunk)
        elif 'library' in facility_topics and re.search(r'\bkütüphane|\bkutuphane', searchable):
            library_chunks.append(chunk)
        if 'sports' in facility_topics and re.search(r'\bspor\b|\bfitness\b|\by[uü]zme\b', searchable):
            sports_chunks.append(chunk)
        if 'campus_life' in facility_topics and re.search(
            r'\b(kampüs|kampus|sosyal|yemekhane|kafeterya|hizmet|i̇mkan|imkan|olanak|yurt|öğrenci|ogrenci|spor|kütüphane|kutuphane)\b',
            searchable,
        ):
            campus_chunks.append(chunk)
        if 'student_clubs' in facility_topics and re.search(
            r'\b(kulüp|kulup|kulub|topluluk|öğrenci kulüpleri|ogrenci kulupleri)\b',
            searchable,
        ):
            club_chunks.append(chunk)

    if not library_chunks and not sports_chunks and not campus_chunks and not club_chunks:
        return '', []

    lines: list[str] = []
    if library_chunks:
        lib_excerpts = _extract_facility_excerpts(
            library_chunks, re.compile(r'\bkütüphane|\bkutuphane|\blibrary', re.IGNORECASE)
        )
        if lib_excerpts:
            used_chunks.extend(library_chunks)
            lines.append('Kütüphane:')
            for excerpt in lib_excerpts:
                lines.append(f'- {excerpt}')
    if sports_chunks:
        sport_excerpts = _extract_facility_excerpts(
            sports_chunks, re.compile(r'\bspor\b|\bfitness\b|\byüzme|\bbasketbol|\bhavuz', re.IGNORECASE)
        )
        if sport_excerpts:
            used_chunks.extend(sports_chunks)
            lines.append('Spor Merkezi:')
            for excerpt in sport_excerpts:
                lines.append(f'- {excerpt}')
    if campus_chunks:
        campus_excerpts = _extract_facility_excerpts(
            campus_chunks,
            re.compile(
                r'\b(kampüs|kampus|sosyal|yemekhane|kafeterya|hizmet|imkan|olanak|yurt|öğrenci|spor|kütüphane|kafetarya|kafe)\b',
                re.IGNORECASE,
            ),
        )
        if campus_excerpts:
            used_chunks.extend(campus_chunks)
            lines.append('Kampüs Olanakları:')
            for excerpt in campus_excerpts:
                lines.append(f'- {excerpt}')
    if club_chunks:
        club_excerpts = _extract_facility_excerpts(
            club_chunks,
            re.compile(
                r'\b(kulüp|kulup|kulub|topluluk|öğrenci kulüpleri|ogrenci kulupleri)\b',
                re.IGNORECASE,
            ),
        )
        if club_excerpts:
            used_chunks.extend(club_chunks)
            lines.append('Öğrenci Kulüpleri:')
            for excerpt in club_excerpts:
                lines.append(f'- {excerpt}')

    if not lines:
        return '', []
    return '\n'.join(lines), used_chunks


def _staff_member_chunks_for_context(chunks: list[ContentChunk]) -> list[ContentChunk]:
    staff_page_ids = {
        chunk.page_id
        for chunk in chunks
        if _get_chunk_metadata_value(chunk, 'kind') in {'main_site_staff_page', 'bologna_staff_page'}
        or _get_chunk_metadata_value(chunk, 'record_type') == 'academic_staff_member'
    }
    if not staff_page_ids:
        return []

    return list(
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True, page_id__in=staff_page_ids)
        .filter(metadata__record_type='academic_staff_member')
        .order_by('page_id', 'chunk_index')
    )


def _staff_program_title(chunk: ContentChunk) -> str:
    program_title = (
        _get_chunk_metadata_value(chunk, 'program_title')
        or _get_chunk_metadata_value(chunk, 'unit_name')
        or chunk.page.title.split(' - ', 1)[0]
    )
    return _clean_display_text(program_title)


def _looks_like_person_label(value: str) -> bool:
    label = _clean_display_text(value)
    lowered = _normalize_lookup_text(label)
    if not label or any(char.isdigit() for char in label):
        return False
    if any(
        token in lowered
        for token in (
            'akademik',
            'bolumu',
            'fakultesi',
            'genetik',
            'kadro',
            'mesaji',
            'muhendisligi',
            'programi',
            'universitesi',
            'yonetimi',
            'yuksekokulu',
        )
    ):
        return False
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü'’.-]+", label)
    if len(words) < 2 or len(words) > 8:
        return False
    return sum(1 for word in words if word and word[0].isupper()) >= 2


def _metadata_person_label(chunk: ContentChunk) -> str:
    scope_values = {
        _normalize_lookup_text(value)
        for value in (
            _get_chunk_metadata_value(chunk, 'program_title'),
            _get_chunk_metadata_value(chunk, 'unit_name'),
            _get_chunk_metadata_value(chunk, 'faculty'),
            _get_chunk_metadata_value(chunk, 'section_title'),
            chunk.page.title,
        )
        if value
    }
    for key in ('entity_label', 'entity_name', 'staff_name'):
        label = _clean_display_text(_get_chunk_metadata_value(chunk, key))
        if not label:
            continue
        if _normalize_lookup_text(label) in scope_values:
            continue
        if _looks_like_person_label(label):
            return label
    return ''


def _extract_department_head_label(chunk: ContentChunk) -> str:
    patterns = (
        r'bölüm\s*başkan[ıi]?\s*[-:]\s*([^|/\n\r]+)',
        r'bolum\s*baskan[ıi]?\s*[-:]\s*([^|/\n\r]+)',
        r'([^|/\n\r]+?)\s*[-:]\s*bölüm\s*başkan[ıi]?\b',
        r'([^|/\n\r]+?)\s*[-:]\s*bolum\s*baskan[ıi]?\b',
    )
    for pattern in patterns:
        match = re.search(pattern, chunk.text, flags=re.IGNORECASE)
        if not match:
            continue
        label = _clean_display_text(match.group(1).strip(' -:'))
        if _looks_like_person_label(label):
            return label
    return _metadata_person_label(chunk)


def _role_assignments_from_chunk(chunk: ContentChunk) -> list[tuple[str, str]]:
    assignments: list[tuple[str, str]] = []
    metadata_role = _clean_display_text(_get_chunk_metadata_value(chunk, 'role'))
    metadata_name = _clean_display_text(
        _get_chunk_metadata_value(chunk, 'entity_name')
        or _get_chunk_metadata_value(chunk, 'staff_name')
    )
    if metadata_role and _looks_like_person_label(metadata_name):
        assignments.append((metadata_role, metadata_name))

    for line in re.split(r'\n+', chunk.text):
        role_match = re.search(r'(?:^|\|)\s*rol\s*:\s*([^|\n\r]+)', line, flags=re.IGNORECASE)
        name_match = re.search(r'(?:^|\|)\s*isim\s*:\s*([^|\n\r]+)', line, flags=re.IGNORECASE)
        if not role_match or not name_match:
            continue
        role = _clean_display_text(role_match.group(1))
        name = _clean_display_text(name_match.group(1))
        if role and _looks_like_person_label(name):
            assignments.append((role, name))
    return list(dict.fromkeys(assignments))


def _role_matches_question(role: str, requested_role: str) -> bool:
    normalized_role = _normalize_lookup_text(role)
    if requested_role == 'department_head':
        return 'bolum baskan' in normalized_role
    if requested_role == 'deputy_dean':
        return 'dekan' in normalized_role and 'yardimc' in normalized_role
    if requested_role == 'dean':
        return 'dekan' in normalized_role and 'yardimc' not in normalized_role
    if requested_role == 'director':
        return 'mudur' in normalized_role and 'yardimc' not in normalized_role
    return False


def _role_scope_title(chunk: ContentChunk, requested_role: str) -> str:
    if requested_role in {'dean', 'deputy_dean'}:
        scope = _get_chunk_metadata_value(chunk, 'faculty') or _get_chunk_metadata_value(chunk, 'program_title')
    else:
        scope = _get_chunk_metadata_value(chunk, 'program_title') or _get_chunk_metadata_value(chunk, 'unit_name')
    scope = scope or _get_chunk_metadata_value(chunk, 'faculty') or chunk.page.title.split(' - ', 1)[0]
    scope = _clean_display_text(re.sub(r'\s+(yönetimi|yonetimi)$', '', scope, flags=re.IGNORECASE))
    return scope


def _build_role_specific_staff_answer(question: str, chunks: list[ContentChunk]) -> str:
    requested_role = _requested_staff_role(question)
    if not requested_role:
        return ''

    if requested_role == 'department_head':
        for chunk in chunks:
            if _get_chunk_metadata_value(chunk, 'record_type') != 'department_head_message':
                continue
            label = _extract_department_head_label(chunk)
            if label:
                program_title = _staff_program_title(chunk)
                return f'{program_title} bölüm başkanı {label} olarak görünüyor.'

    role_labels = {
        'department_head': 'bölüm başkanı',
        'dean': 'dekanı',
        'deputy_dean': 'dekan yardımcısı',
        'director': 'müdürü',
    }
    for chunk in chunks:
        for role, name in _role_assignments_from_chunk(chunk):
            if not _role_matches_question(role, requested_role):
                continue
            scope = _role_scope_title(chunk, requested_role)
            return f'{scope} {role_labels[requested_role]} {name} olarak görünüyor.'

    return 'Bu rol için doğrulanmış yönetici kaynağı bulamadım.'


def _build_structured_staff_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_staff_query(question):
        return ''

    role_answer = _build_role_specific_staff_answer(question, chunks)
    if role_answer:
        return role_answer

    staff_chunks = _staff_member_chunks_for_context(chunks)
    if not staff_chunks:
        staff_count_chunks = [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'staff_count')
            and _get_chunk_metadata_value(chunk, 'kind') in {'main_site_staff_page', 'bologna_staff_page'}
        ]
        if not staff_count_chunks:
            return ''
        chunk = staff_count_chunks[0]
        program_title = _staff_program_title(chunk)
        staff_count = _get_chunk_metadata_value(chunk, 'staff_count')
        return f'{program_title} akademik kadro kaynağında {staff_count} hoca kaydı görünüyor.'

    unique_members: dict[str, tuple[str, str]] = {}
    for chunk in staff_chunks:
        name = _clean_display_text(_get_chunk_metadata_value(chunk, 'entity_name'))
        if not name:
            match = re.search(r'isim:\s*([^|]+)', chunk.text)
            name = _clean_display_text(match.group(1)) if match else ''
        if not name:
            continue

        title = _clean_display_text(
            _get_chunk_metadata_value(chunk, 'staff_title')
            or (re.search(r'unvan:\s*([^|]+)', chunk.text).group(1) if re.search(r'unvan:\s*([^|]+)', chunk.text) else '')
        )
        unique_members.setdefault(_normalize_lookup_text(name), (name, title))

    if not unique_members:
        return ''

    first_chunk = staff_chunks[0]
    program_title = _staff_program_title(first_chunk)
    staff_count_text = _get_chunk_metadata_value(first_chunk, 'staff_count')
    try:
        staff_count = int(staff_count_text) if staff_count_text else len(unique_members)
    except ValueError:
        staff_count = len(unique_members)
    staff_count = max(staff_count, len(unique_members))

    if _is_staff_count_query(question) and not _is_staff_list_query(question):
        return (
            f'{program_title} akademik kadro kaynağında {staff_count} hoca kaydı var.'
        )

    lines = [f'{program_title} akademik kadro kaynağında {staff_count} hoca kaydı var:']
    for name, title in unique_members.values():
        label = f'{title} {name}'.strip() if title else name
        lines.append(f'- {label}')
    return '\n'.join(lines)


def _build_program_presence_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_program_exists_query(question):
        return ''

    if _is_dentistry_query(question):
        has_dentistry = any(
            _normalize_lookup_text(candidate) == 'diş hekimliği'
            for candidate in _known_program_candidates()
        )
        oral_chunks = [
            chunk
            for chunk in chunks
            if 'ağız ve diş sağlığı' in _normalize_lookup_text(
                _get_chunk_metadata_value(chunk, 'program_title') or chunk.page.title
            )
        ]
        if has_dentistry:
            return 'Evet, indekslenmiş resmi kaynaklarda Diş Hekimliği programı görünüyor.'
        if oral_chunks:
            faculty = _get_chunk_metadata_value(oral_chunks[0], 'faculty')
            level = _get_chunk_metadata_value(oral_chunks[0], 'admission_level')
            level_text = 'ön lisans' if level == 'onlisans' else level
            details = []
            if level_text:
                details.append(level_text)
            if faculty:
                details.append(faculty)
            suffix = f" ({', '.join(details)})" if details else ''
            return (
                'İndekslenmiş resmi kaynaklarda Diş Hekimliği lisans programı bulamadım. '
                f'Buna yakın fakat farklı bir program olarak Ağız ve Diş Sağlığı programı var{suffix}.'
            )
        return 'İndekslenmiş resmi kaynaklarda Diş Hekimliği programı bulamadım.'

    program_title = (
        _extract_program_abbreviation_from_text(question)
        or _extract_known_program_from_text(question)
        or _extract_program_hint_from_text(question)
        or _extract_program_from_tokens(question)
    )
    if not program_title:
        return ''

    program_terms = _program_lookup_terms(program_title)
    matching_chunks = [
        chunk
        for chunk in chunks
        if any(
            _normalized_contains(
                ' '.join(
                    value
                    for value in (
                        _get_chunk_metadata_value(chunk, 'program_title'),
                        _get_chunk_metadata_value(chunk, 'placement_label'),
                        _get_chunk_metadata_value(chunk, 'unit_name'),
                        chunk.page.title,
                    )
                    if value
                ),
                term,
            )
            for term in program_terms
        )
    ]
    if not matching_chunks:
        return ''

    chunk = matching_chunks[0]
    display_title = _get_chunk_metadata_value(chunk, 'program_title') or program_title
    faculty = _get_chunk_metadata_value(chunk, 'faculty')
    level = _get_chunk_metadata_value(chunk, 'admission_level')
    details = []
    if faculty:
        details.append(faculty)
    if level:
        details.append('ön lisans' if level == 'onlisans' else level)
    suffix = f" ({', '.join(details)})" if details else ''
    return f'Evet, Acıbadem Üniversitesi resmi kaynaklarında {display_title} programı var{suffix}.'


def _vector_distance_threshold(question: str) -> float:
    if (_is_score_query(question) or _is_fee_query(question)
            or _is_staff_query(question) or _is_course_query(question)):
        return settings.RAG_VECTOR_DISTANCE_STRICT
    return settings.RAG_VECTOR_DISTANCE_BROAD


def retrieve_context(
    query_embedding: list[float],
    limit: int | None = None,
    per_page_limit: int | None = None,
    question: str = '',
) -> list[ContentChunk]:
    limit = limit or settings.RAG_RETRIEVE_LIMIT
    per_page_limit = per_page_limit or settings.RAG_PER_PAGE_LIMIT
    threshold = _vector_distance_threshold(question)
    fallback_threshold = max(threshold, settings.RAG_VECTOR_DISTANCE_BROAD)
    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(embedding__isnull=False, page__is_active=True)
        .annotate(distance=CosineDistance('embedding', query_embedding))
        .order_by('distance')
    )
    candidates = list(queryset[: limit * 4])

    def _select_with_threshold(max_distance: float) -> list[ContentChunk]:
        selected: list[ContentChunk] = []
        for chunk in candidates:
            if chunk.distance is None or chunk.distance > max_distance:
                continue
            selected.append(chunk)
            if len(selected) >= limit * 2:
                break
        return selected

    selected = _select_with_threshold(threshold)
    if not selected and fallback_threshold > threshold:
        selected = _select_with_threshold(fallback_threshold)
    return _limit_chunks(selected, limit=limit, per_page_limit=per_page_limit)


def _chunks_to_hits(
    chunks: list[ContentChunk],
    *,
    method: str,
    weight: float,
    protected: bool = False,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for index, chunk in enumerate(chunks, start=1):
        hits.append(
            RetrievalHit(
                chunk=chunk,
                method=method,
                rank=index,
                weight=weight,
                protected=protected,
                distance=getattr(chunk, 'distance', None),
                keyword_rank=getattr(chunk, 'rank', None),
            )
        )
    return hits


def _sort_candidate_hits(question: str, hits: list[RetrievalHit]) -> list[ContentChunk]:
    grouped: dict[int, dict] = {}
    for order, hit in enumerate(hits):
        chunk_key = hit.chunk.pk or id(hit.chunk)
        if chunk_key not in grouped:
            grouped[chunk_key] = {
                'chunk': hit.chunk,
                'score': 0.0,
                'protected': False,
                'first_order': order,
            }
        grouped_hit = grouped[chunk_key]
        grouped_hit['score'] += hit.weight / (settings.RAG_RRF_K + hit.rank)
        grouped_hit['protected'] = grouped_hit['protected'] or hit.protected
        grouped_hit['first_order'] = min(grouped_hit['first_order'], order)

    return [
        item['chunk']
        for item in sorted(
            grouped.values(),
            key=lambda item: (
                0 if item['protected'] else 1,
                _chunk_priority(question, item['chunk']),
                -item['score'],
                item['first_order'],
            ),
        )
    ]


def _retrieve_direct_program_list_chunks(question: str, limit: int) -> list[ContentChunk]:
    if not _is_program_list_query(question):
        return []

    queryset = ContentChunk.objects.select_related('page').filter(page__is_active=True)
    faculty_lookup = (
        Q(metadata__faculty__icontains='Mühendislik ve Doğa Bilimleri')
        | Q(page__title__icontains='Mühendislik ve Doğa Bilimleri')
        | Q(text__icontains='Mühendislik ve Doğa Bilimleri')
    )
    program_source_lookup = (
        Q(metadata__record_type='bologna_program_overview')
        | Q(metadata__kind='bologna_program_page')
        | (
            Q(metadata__kind='main_site_page')
            & Q(metadata__program_title='Bölümler')
        )
    )
    queryset = queryset.filter(faculty_lookup).filter(program_source_lookup)

    if _is_engineering_program_list_query(question) and not _is_engineering_and_natural_sciences_faculty_query(question):
        engineering_lookup = (
            Q(metadata__program_title__icontains='Mühendisliği')
            | Q(metadata__program_title__icontains='Muhendisligi')
            | Q(page__title__icontains='Mühendisliği')
            | Q(page__title__icontains='Muhendisligi')
        )
        queryset = queryset.filter(engineering_lookup)

    chunks = list(queryset.order_by('metadata__program_title', 'page_id', 'chunk_index')[: limit * 2])
    return _limit_chunks(chunks, limit=limit, per_page_limit=1)


def _retrieve_direct_score_chunks(question: str, limit: int) -> list[ContentChunk]:
    if not _is_score_query(question):
        return []

    program_title = (
        _extract_program_abbreviation_from_text(question)
        or _extract_known_program_from_text(question)
        or _extract_program_hint_from_text(question)
        or _extract_program_from_tokens(question)
    )

    program_lookup = Q()
    if program_title:
        for term in _program_lookup_terms(program_title):
            program_lookup |= (
                Q(metadata__program_title__icontains=term)
                | Q(metadata__placement_label__icontains=term)
                | Q(metadata__unit_name__icontains=term)
                | Q(metadata__program_alias_text__icontains=term)
                | Q(page__title__icontains=term)
                | Q(text__icontains=term)
            )

    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(Q(metadata__kind='structured_admissions_score') | Q(metadata__record_type='quota_row'))
    )
    if program_lookup:
        queryset = queryset.filter(program_lookup)
    return list(queryset.order_by('page_id', 'chunk_index')[:limit])


def _direct_candidate_hits(question: str, candidate_limit: int) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_staff_chunks(question, limit=candidate_limit * 2),
            method='direct_staff',
            weight=2.0,
            protected=_is_staff_query(question),
        )
    )
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_program_chunks(question, limit=candidate_limit),
            method='direct_program',
            weight=2.0,
            protected=_is_program_exists_query(question),
        )
    )
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_program_list_chunks(question, limit=candidate_limit),
            method='direct_program_list',
            weight=4.0,
            protected=_is_program_list_query(question),
        )
    )
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_score_chunks(question, limit=candidate_limit),
            method='direct_score',
            weight=2.0,
            protected=_is_score_query(question),
        )
    )
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_fee_chunks(question, limit=candidate_limit),
            method='direct_fee',
            weight=2.0,
            protected=_is_fee_query(question),
        )
    )
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_course_chunks(question, limit=candidate_limit * 2),
            method='direct_course',
            weight=2.0,
            protected=_is_course_query(question),
        )
    )
    hits.extend(
        _chunks_to_hits(
            _retrieve_direct_facility_chunks(question, limit=candidate_limit * 2),
            method='direct_facility',
            weight=3.0,
            protected=bool(_question_topics(question) & {'library', 'sports', 'campus_life', 'student_clubs', 'dormitory', 'international', 'transport', 'prep', 'scholarships'}),
        )
    )
    return hits


def retrieve_keyword_context(
    question: str, limit: int | None = None, per_page_limit: int | None = None
) -> list[ContentChunk]:
    limit = limit or settings.RAG_RETRIEVE_LIMIT
    per_page_limit = per_page_limit or settings.RAG_PER_PAGE_LIMIT
    normalized_question = ' '.join(question.split())
    if not normalized_question:
        return []

    query = SearchQuery(normalized_question, search_type='plain', config='simple')
    for term in _expanded_query_terms(question):
        query |= SearchQuery(term, search_type='plain', config='simple')
    vector = (
        SearchVector('text', config='simple')
        + SearchVector('page__title', config='simple')
        + SearchVector('metadata__program_title', config='simple')
        + SearchVector('metadata__program_alias_text', config='simple')
        + SearchVector('metadata__placement_label', config='simple')
        + SearchVector('metadata__faculty', config='simple')
        + SearchVector('metadata__entity_name', config='simple')
        + SearchVector('metadata__record_type', config='simple')
        + SearchVector('metadata__chunk_level', config='simple')
        + SearchVector('metadata__curriculum_year', config='simple')
        + SearchVector('metadata__period_label', config='simple')
        + SearchVector('metadata__topic_label', config='simple')
        + SearchVector('metadata__section_title', config='simple')
    )
    trgm_similarity = (
        TrigramSimilarity('text', normalized_question)
        + TrigramSimilarity('page__title', normalized_question)
    ) * 0.5

    trgm_sim_safe = Coalesce('trgm_sim', Value(0.0))
    combined_rank_expr = F('fts_rank') * 0.6 + trgm_sim_safe * 0.4
    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .annotate(fts_rank=SearchRank(vector, query))
        .annotate(trgm_sim=trgm_similarity)
        .annotate(combined_rank=combined_rank_expr)
        .filter(
            Q(fts_rank__gt=0) | Q(trgm_sim__gt=0.15)
        )
        .order_by('-combined_rank', 'page_id', 'chunk_index')
    )

    return _limit_chunks(list(queryset[: limit * 4]), limit=limit, per_page_limit=per_page_limit)


def _truncate_context_text(text: str, max_chars: int) -> str:
    normalized = ' '.join((text or '').split())
    if max_chars <= 0:
        return ''
    if len(normalized) <= max_chars:
        return normalized
    if max_chars == 1:
        return '…'
    return normalized[: max_chars - 1].rstrip() + '…'


def _prompt_context_char_limit(question: str) -> int:
    return settings.RAG_MAX_CONTEXT_CHARS


def _select_prompt_chunks(question: str, chunks: list[ContentChunk]) -> list[tuple[ContentChunk, str]]:
    max_chunk_chars = settings.RAG_MAX_CHUNK_CHARS
    max_context_chars = _prompt_context_char_limit(question)
    selected: list[tuple[ContentChunk, str]] = []
    total_chars = 0

    for chunk in chunks:
        excerpt = _truncate_context_text(chunk.text, max_chunk_chars)
        block_index = len(selected) + 1
        block = '\n'.join(
            [
                f'Kaynak {block_index}',
                f'Baslik: {chunk.page.title}',
                f'Program: {_get_chunk_metadata_value(chunk, "program_title") or "-"}',
                f'Fakulte: {_get_chunk_metadata_value(chunk, "faculty") or "-"}',
                f'Mufredat Yili: {_get_chunk_metadata_value(chunk, "curriculum_year") or "-"}',
                f'Donem: {_get_chunk_metadata_value(chunk, "period_label") or "-"}',
                f'Bolum/Sayfa: {_get_chunk_metadata_value(chunk, "section_title") or chunk.page.title}',
                f'URL: {_get_chunk_source_url(chunk)}',
                f'Icerik: {excerpt}',
            ]
        )

        if total_chars and total_chars + len(block) > max_context_chars:
            break
        if total_chars + len(block) > max_context_chars:
            remaining_chars = max_context_chars - total_chars
            overhead = len(block) - len(excerpt)
            excerpt = _truncate_context_text(chunk.text, remaining_chars - overhead)
            block = '\n'.join(
                [
                    f'Kaynak {block_index}',
                    f'Baslik: {chunk.page.title}',
                    f'Program: {_get_chunk_metadata_value(chunk, "program_title") or "-"}',
                    f'Fakulte: {_get_chunk_metadata_value(chunk, "faculty") or "-"}',
                    f'Mufredat Yili: {_get_chunk_metadata_value(chunk, "curriculum_year") or "-"}',
                    f'Donem: {_get_chunk_metadata_value(chunk, "period_label") or "-"}',
                    f'Bolum/Sayfa: {_get_chunk_metadata_value(chunk, "section_title") or chunk.page.title}',
                    f'URL: {_get_chunk_source_url(chunk)}',
                    f'Icerik: {excerpt}',
                ]
            )
            if total_chars + len(block) > max_context_chars or not excerpt:
                break

        selected.append((chunk, excerpt))
        total_chars += len(block) + 2

    return selected


def build_prompt(question: str, chunks: list[ContentChunk]) -> tuple[str, list[ContentChunk]]:
    prompt_chunks = _select_prompt_chunks(question, chunks)
    context_blocks = []
    used_chunks = []
    for index, (chunk, excerpt) in enumerate(prompt_chunks, start=1):
        used_chunks.append(chunk)
        context_blocks.append(
            '\n'.join(
                [
                    f'Kaynak {index}',
                    f'Baslik: {chunk.page.title}',
                    f'Program: {_get_chunk_metadata_value(chunk, "program_title") or "-"}',
                    f'Fakulte: {_get_chunk_metadata_value(chunk, "faculty") or "-"}',
                    f'Mufredat Yili: {_get_chunk_metadata_value(chunk, "curriculum_year") or "-"}',
                    f'Donem: {_get_chunk_metadata_value(chunk, "period_label") or "-"}',
                    f'Bolum/Sayfa: {_get_chunk_metadata_value(chunk, "section_title") or chunk.page.title}',
                    f'URL: {_get_chunk_source_url(chunk)}',
                    f'Icerik: {excerpt}',
                ]
            )
        )

    context = '\n\n'.join(context_blocks)
    additional_rules = _question_specific_prompt_rules(question)
    rules_block = ''.join(f'- {rule}\n' for rule in additional_rules) if additional_rules else ''
    prompt = (
        '/no_think\n'
        'Sen Acıbadem Üniversitesi için resmi kaynaklardan cevap veren bir asistansın.\n'
        'Yalnızca verilen bağlama dayan.\n'
        'Cevap biçimi: yalnızca kullanıcıya gösterilecek nihai Türkçe cevabı yaz.\n'
        'Yanıtına tam olarak "CEVAP:" ile başla; bu etiketin öncesine hiçbir metin yazma.\n'
        'Kaynakları nasıl incelediğini, analiz sürecini, İngilizce taslağı veya "Possible answer" gibi ara notları yazma.\n'
        '<think>, </think> veya benzeri muhakeme etiketi üretme.\n'
        'Bağlamda soruyla ilgili bilgi varsa bu bilgiden yararlanarak cevap ver.\n'
        'Eksik kısımlar varsa mevcut bilgiyi paylaşıp hangi kısımların eksik olduğunu belirt.\n'
        'Bağlamda hiç ilgili bilgi yoksa bunu açıkça söyle, tahmin yürütme.\n'
        'Sıralama, karşılaştırma veya istatistik gerektiren sorularda veri yoksa cevap uydurma.\n'
        'Program veya fakülte odaklı sorularda yalnızca metadata olarak eşleşen kaynakları kullan.\n'
        'Cevabı Türkçe ver.\n'
        'Metin içinde kaynak numaralarını [1], [2] gibi kullan.\n\n'
        f'Ek kurallar:\n{rules_block}\n'
        f'Bağlam:\n{context}\n\n'
        f'Kullanıcı sorusu: {question}'
    )
    return prompt, used_chunks


def generate_answer(prompt: str) -> str:
    messages = [
        {
            'role': 'system',
            'content': _llm_system_prompt(),
        },
        {'role': 'user', 'content': prompt},
    ]
    if _use_ollama_backend():
        answer = _ollama_chat(
            messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        if _is_llm_format_error_answer(answer):
            retry_messages = [
                {'role': 'system', 'content': _llm_system_prompt()},
                {'role': 'user', 'content': _llm_retry_prompt(prompt)},
            ]
            answer = _ollama_chat(
                retry_messages,
                temperature=0,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        return answer

    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.1,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=messages,
    )
    answer = _clean_llm_answer(response.choices[0].message.content or '')
    if _is_llm_format_error_answer(answer):
        retry_messages = [
            {'role': 'system', 'content': _llm_system_prompt()},
            {'role': 'user', 'content': _llm_retry_prompt(prompt)},
        ]
        response = get_llm_client().chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0,
            max_tokens=settings.LLM_MAX_TOKENS,
            messages=retry_messages,
        )
        answer = _clean_llm_answer(response.choices[0].message.content or '')
    return answer


def _generate_answer_with_slot(prompt: str) -> str:
    semaphore = _acquire_llm_slot()
    try:
        return generate_answer(prompt)
    finally:
        semaphore.release()


def _iter_delta_content_text(delta_content: object) -> Iterator[str]:
    if isinstance(delta_content, str):
        if delta_content:
            yield delta_content
        return

    if isinstance(delta_content, list):
        for part in delta_content:
            text = getattr(part, 'text', None)
            if text is None and isinstance(part, dict):
                text = part.get('text')
            if text:
                yield str(text)
        return

    text = getattr(delta_content, 'text', None)
    if text:
        yield str(text)


def generate_answer_stream(prompt: str) -> Iterator[str]:
    messages = [
        {
            'role': 'system',
            'content': _llm_system_prompt(),
        },
        {'role': 'user', 'content': prompt},
    ]
    if _use_ollama_backend():
        raw_answer = ''.join(
            _ollama_chat_stream(
                messages,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        )
        cleaned_answer = _clean_llm_answer(raw_answer)
        if _is_llm_format_error_answer(cleaned_answer):
            retry_messages = [
                {'role': 'system', 'content': _llm_system_prompt()},
                {'role': 'user', 'content': _llm_retry_prompt(prompt)},
            ]
            raw_answer = ''.join(
                _ollama_chat_stream(
                    retry_messages,
                    temperature=0,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
            )
            cleaned_answer = _clean_llm_answer(raw_answer)
        if cleaned_answer:
            yield cleaned_answer
        return

    stream = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.1,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=messages,
        stream=True,
    )
    answer_parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        answer_parts.extend(_iter_delta_content_text(delta))
    cleaned_answer = _clean_llm_answer(''.join(answer_parts))
    if _is_llm_format_error_answer(cleaned_answer):
        retry_messages = [
            {'role': 'system', 'content': _llm_system_prompt()},
            {'role': 'user', 'content': _llm_retry_prompt(prompt)},
        ]
        stream = get_llm_client().chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0,
            max_tokens=settings.LLM_MAX_TOKENS,
            messages=retry_messages,
            stream=True,
        )
        retry_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            retry_parts.extend(_iter_delta_content_text(delta))
        cleaned_answer = _clean_llm_answer(''.join(retry_parts))
    if cleaned_answer:
        yield cleaned_answer


def _generate_answer_stream_with_slot(prompt: str) -> Iterator[str]:
    semaphore = _acquire_llm_slot()
    try:
        yield from generate_answer_stream(prompt)
    finally:
        semaphore.release()


def build_sources(chunks: list[ContentChunk]) -> list[dict]:
    seen_page_ids: set[int] = set()
    sources: list[dict] = []
    for chunk in chunks:
        if chunk.page_id in seen_page_ids:
            continue
        seen_page_ids.add(chunk.page_id)
        sources.append(
            {
                'title': chunk.page.title,
                'label': _build_source_label(chunk),
                'url': _get_chunk_source_url(chunk),
                'source': chunk.page.source,
                'chunk_index': chunk.chunk_index,
            }
        )
    return sources


def cache_key(question: str) -> str:
    normalized = normalize_question(question)
    return f'chat-answer:{CACHE_KEY_VERSION}:{hashlib.sha256(normalized.encode("utf-8")).hexdigest()}'


def get_conversation(conversation_id: int | None, question: str) -> Conversation:
    if conversation_id is None:
        return Conversation.objects.create(title=question[:80])
    try:
        return Conversation.objects.get(pk=conversation_id)
    except Conversation.DoesNotExist as exc:
        raise ConversationNotFoundError(str(conversation_id)) from exc


def _persist_exchange(conversation: Conversation, question: str, answer: str) -> None:
    Message.objects.create(conversation=conversation, role='user', content=question)
    Message.objects.create(conversation=conversation, role='assistant', content=answer)


def _question_hash(question: str) -> str:
    return cache_key(question).rsplit(':', 1)[1][:QUESTION_HASH_LENGTH]


def _retrieve_direct_staff_chunks(question: str, limit: int) -> list[ContentChunk]:
    if not _is_staff_query(question):
        return []

    program_title = (
        _extract_program_abbreviation_from_text(question)
        or _extract_known_program_from_text(question)
        or _extract_program_hint_from_text(question)
    )
    if not program_title:
        return []

    program_lookup = Q()
    for term in _program_lookup_terms(program_title):
        program_lookup |= (
            Q(metadata__program_title__icontains=term)
            | Q(metadata__unit_name__icontains=term)
            | Q(metadata__faculty__icontains=term)
            | Q(metadata__program_alias_text__icontains=term)
            | Q(page__title__icontains=term)
            | Q(text__icontains=term)
        )

    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(
            Q(metadata__kind='main_site_staff_page')
            | Q(metadata__kind='main_site_role_page')
            | Q(metadata__kind='bologna_staff_page')
            | Q(metadata__record_type='academic_staff_member')
            | Q(metadata__record_type='department_head_message')
            | Q(metadata__record_type='staff_role_assignment')
        )
        .filter(program_lookup)
        .order_by('page_id', 'chunk_index')
    )
    return list(queryset[:limit])


def _retrieve_direct_program_chunks(question: str, limit: int) -> list[ContentChunk]:
    if not _is_program_exists_query(question):
        return []

    if _is_dentistry_query(question):
        lookups = Q(metadata__program_title__icontains='Ağız ve Diş Sağlığı') | Q(
            page__title__icontains='Ağız ve Diş Sağlığı'
        )
    else:
        program_title = _extract_known_program_from_text(question) or _extract_program_hint_from_text(question) or _extract_program_from_tokens(question)
        if not program_title:
            return []
        lookups = Q()
        for term in _program_lookup_terms(program_title):
            lookups |= (
                Q(metadata__program_title__icontains=term)
                | Q(metadata__program_alias_text__icontains=term)
                | Q(page__title__icontains=term)
                | Q(text__icontains=term)
            )

    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(lookups)
        .order_by('page__source', 'page_id', 'chunk_index')
    )
    return list(queryset[:limit])


def _retrieve_direct_fee_chunks(question: str, limit: int) -> list[ContentChunk]:
    if not _is_fee_query(question):
        return []

    program_title = _fee_program_title(question)
    if not program_title:
        return []

    program_lookup = Q()
    for term in _program_lookup_terms(program_title):
        program_lookup |= (
            Q(metadata__program_title__icontains=term)
            | Q(metadata__placement_label__icontains=term)
            | Q(metadata__unit_name__icontains=term)
            | Q(metadata__program_alias_text__icontains=term)
            | Q(page__title__icontains=term)
            | Q(text__icontains=term)
        )

    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(Q(metadata__kind='structured_admissions_fee') | Q(metadata__record_type='tuition_fee'))
        .filter(program_lookup)
        .order_by('page_id', 'chunk_index')
    )
    return [chunk for chunk in queryset[:limit] if _chunk_has_fee_amount(chunk)]


def _retrieve_direct_course_chunks(question: str, limit: int) -> list[ContentChunk]:
    if not _is_course_query(question):
        return []

    program_title = _course_program_title(question)
    if not program_title:
        return []

    program_lookup = Q()
    for term in _program_lookup_terms(program_title):
        program_lookup |= (
            Q(metadata__program_title__icontains=term)
            | Q(metadata__program_alias_text__icontains=term)
            | Q(metadata__unit_name__icontains=term)
            | Q(metadata__faculty__icontains=term)
            | Q(page__title__icontains=term)
            | Q(text__icontains=term)
        )

    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(
            Q(metadata__chunk_level='program_overview')
            | Q(metadata__chunk_level='semester_plan')
            | Q(metadata__record_type='bologna_program_overview')
            | Q(metadata__record_type='bologna_semester_plan')
        )
        .filter(program_lookup)
    )
    return sorted(
        list(queryset[: limit * 3]),
        key=lambda chunk: (-_course_curriculum_year(chunk), _course_sort_key(question, chunk)),
    )[:limit]


def _retrieve_direct_student_club_chunks(limit: int) -> list[ContentChunk]:
    club_title_lookup = (
        Q(page__title__icontains='Öğrenci Kulüpleri')
        | Q(page__title__icontains='Ogrenci Kulupleri')
        | Q(page__title__icontains='Kulübü')
        | Q(page__title__icontains='Kulubu')
        | Q(page__title__icontains='Topluluğu')
        | Q(page__title__icontains='Toplulugu')
    )
    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(Q(metadata__topic='student_clubs') | club_title_lookup)
        .order_by('page_id', 'chunk_index')
    )
    return list(queryset[:limit])


def _retrieve_direct_facility_chunks(question: str, limit: int) -> list[ContentChunk]:
    topics = _question_topics(question)
    facility_topics = topics & {'library', 'sports', 'campus_life', 'student_clubs'}
    all_topic = topics & {'transport', 'prep', 'scholarships', 'dormitory', 'international'}
    if not (facility_topics or all_topic):
        return []

    if 'student_clubs' in facility_topics:
        club_hits = _retrieve_direct_student_club_chunks(limit)
        if club_hits and facility_topics == {'student_clubs'} and not all_topic:
            return club_hits

    text_lookup = Q()
    title_lookup = Q()
    for topic_key in facility_topics | all_topic:
        if topic_key not in QUERY_EXPANSIONS:
            continue
        for term in QUERY_EXPANSIONS[topic_key]:
            text_lookup |= Q(text__icontains=term) | Q(page__title__icontains=term)
            title_lookup |= Q(page__title__icontains=term)

    base_queryset = ContentChunk.objects.select_related('page').filter(page__is_active=True)

    title_hits = (
        base_queryset
        .filter(title_lookup)
        .order_by('page_id', 'chunk_index')
    )
    other_hits = (
        base_queryset
        .filter(text_lookup)
        .exclude(title_lookup)
        .order_by('page_id', 'chunk_index')
    )
    results = list(title_hits[:limit])
    if 'student_clubs' in facility_topics:
        for chunk in reversed(_dedupe_chunks(club_hits)):
            if chunk not in results:
                results.insert(0, chunk)
        results = _dedupe_chunks(results)
        if len(results) >= limit:
            return results[:limit]
    remaining = limit - len(results)
    if remaining > 0:
        results.extend(list(other_hits[:remaining]))
    if facility_topics & {'campus_life', 'student_clubs'}:
        results = [
            chunk
            for chunk in results
            if _get_chunk_metadata_value(chunk, 'source_group') not in {'scholarship', 'quota', 'tuition'}
        ]
    return results


def _is_chunk_protected_for_rerank(question: str, chunk: ContentChunk) -> bool:
    kind = _get_chunk_metadata_value(chunk, 'kind')
    record_type = _get_chunk_metadata_value(chunk, 'record_type')
    if _is_score_query(question) and kind == 'structured_admissions_score':
        return True
    if _is_score_query(question) and record_type == 'quota_row':
        return True
    if _is_fee_query(question) and kind == 'structured_admissions_fee':
        return True
    if _is_fee_query(question) and record_type == 'tuition_fee':
        return True
    if _is_staff_query(question) and kind == 'main_site_role_page':
        return True
    if _is_staff_query(question) and record_type in {'department_head_message', 'staff_role_assignment'}:
        return True
    if _is_program_list_query(question) and (
        record_type == 'bologna_program_overview'
        or (kind == 'bologna_program_page' and _get_chunk_metadata_value(chunk, 'program_title'))
    ):
        return True
    topics = _question_topics(question)
    if topics and _chunk_matches_question_topic(topics, chunk):
        return True
    return False


def _rerank_candidates(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    from scraper.embeddings import rerank_texts

    if not settings.RERANK_ENABLED:
        return chunks
    if len(chunks) <= settings.RERANK_OUTPUT_LIMIT:
        return chunks

    protected = [c for c in chunks if _is_chunk_protected_for_rerank(question, c)]
    unprotected = [c for c in chunks if not _is_chunk_protected_for_rerank(question, c)]

    if not unprotected:
        return chunks

    candidates = unprotected[:settings.RERANK_CANDIDATE_LIMIT]
    candidate_texts = [_truncate_context_text(c.text, 500) for c in candidates]

    try:
        reranked = rerank_texts(
            query=question,
            documents=candidate_texts,
            top_k=settings.RERANK_OUTPUT_LIMIT,
        )
    except Exception:
        logger.exception("Rerank failed, falling back to original order")
        return chunks

    reranked_chunks = []
    reranked_pks = set()
    for original_idx, score in reranked:
        if score >= settings.RERANK_MIN_SCORE:
            chunk = candidates[original_idx]
            reranked_chunks.append(chunk)
            reranked_pks.add(chunk.pk)

    remaining_unprotected = [
        c for c in unprotected if c.pk not in reranked_pks
    ]

    return protected + reranked_chunks + remaining_unprotected


def _retrieve_candidates(question: str, query_embedding: list[float]) -> list[ContentChunk]:
    candidate_limit = max(settings.RAG_RETRIEVE_LIMIT * CANDIDATE_LIMIT_MULTIPLIER, settings.RAG_RETRIEVE_LIMIT)
    candidate_per_page_limit = max(
        settings.RAG_PER_PAGE_LIMIT,
        settings.RAG_PER_PAGE_LIMIT * CANDIDATE_LIMIT_MULTIPLIER,
    )
    vector_chunks = retrieve_context(
        query_embedding,
        limit=candidate_limit,
        per_page_limit=candidate_per_page_limit,
        question=question,
    )
    keyword_chunks = retrieve_keyword_context(
        question,
        limit=candidate_limit,
        per_page_limit=candidate_per_page_limit,
    )
    hits = (
        _chunks_to_hits(vector_chunks, method='vector', weight=1.0)
        + _chunks_to_hits(keyword_chunks, method='keyword', weight=1.0)
        + _direct_candidate_hits(question, candidate_limit)
    )
    combined = _sort_candidate_hits(question, hits)
    combined = _filter_candidates_for_query(question, combined)
    combined = _rerank_candidates(question, combined)
    if _is_program_list_query(question):
        return _program_list_chunks_for_context(question, combined)
    if _is_course_query(question):
        combined = sorted(combined, key=lambda chunk: _course_sort_key(question, chunk))
    facility_topics = _question_topics(question) & {'library', 'sports', 'campus_life', 'student_clubs', 'dormitory', 'international', 'transport', 'prep', 'scholarships'}
    if facility_topics:
        return _limit_chunks(
            combined,
            limit=settings.RAG_RETRIEVE_LIMIT * 3,
            per_page_limit=settings.RAG_PER_PAGE_LIMIT * 3,
        )
    scoped_chunks, had_scope = _apply_scope_filter(question, combined)
    if had_scope and not scoped_chunks:
        scoped_chunks = combined
    if _is_course_query(question):
        return _limit_chunks(
            scoped_chunks,
            limit=settings.RAG_RETRIEVE_LIMIT * 3,
            per_page_limit=settings.RAG_PER_PAGE_LIMIT * 9,
        )
    return _limit_chunks(
        scoped_chunks,
        limit=settings.RAG_RETRIEVE_LIMIT,
        per_page_limit=settings.RAG_PER_PAGE_LIMIT,
    )


def _elapsed_ms(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000, 2)


def _log_chat_timing(
    *,
    question_hash: str,
    cached: bool,
    conversation_id: int,
    chunk_count: int,
    source_count: int,
    prompt_chars: int,
    timings: dict[str, float],
) -> None:
    logger.info(
        (
            'chat_timing question_hash=%s cached=%s conversation_id=%s chunk_count=%s '
            'source_count=%s prompt_chars=%s cache_ms=%s embed_ms=%s retrieve_ms=%s '
            'prompt_ms=%s llm_ms=%s persist_ms=%s total_ms=%s'
        ),
        question_hash,
        cached,
        conversation_id,
        chunk_count,
        source_count,
        prompt_chars,
        timings.get('cache_ms', 0.0),
        timings.get('embed_ms', 0.0),
        timings.get('retrieve_ms', 0.0),
        timings.get('prompt_ms', 0.0),
        timings.get('llm_ms', 0.0),
        timings.get('persist_ms', 0.0),
        timings.get('total_ms', 0.0),
    )


def _prepare_chat_context(question: str, conversation: Conversation) -> dict:
    resolved_question = _resolve_question_with_conversation(question, conversation)
    key = cache_key(resolved_question)
    timings: dict[str, float] = {}

    cache_start = perf_counter()
    cached_payload = cache.get(key)
    timings['cache_ms'] = _elapsed_ms(cache_start)
    if cached_payload is not None:
        sources = cached_payload.get('sources', [])
        return {
            'key': key,
            'timings': timings,
            'cached_payload': cached_payload,
            'chunks': [],
            'sources': sources,
            'prompt': '',
            'prompt_chars': 0,
            'resolved_question': resolved_question,
        }

    embed_start = perf_counter()
    query_embedding = embed_query(resolved_question)
    timings['embed_ms'] = _elapsed_ms(embed_start)

    semantic_topics = _semantic_question_topics(query_embedding)
    regex_topics = _regex_question_topics(resolved_question)
    combined_topics = regex_topics | semantic_topics
    _set_request_topics(combined_topics)

    try:
        retrieve_start = perf_counter()
        all_chunks = _retrieve_candidates(resolved_question, query_embedding)
        timings['retrieve_ms'] = _elapsed_ms(retrieve_start)
    finally:
        _clear_request_topics()
    prompt = ''
    prompt_chars = 0
    prompt_start = perf_counter()
    prompt_chunks: list[ContentChunk] = []
    if all_chunks:
        prompt, prompt_chunks = build_prompt(resolved_question, all_chunks)
        prompt_chars = len(prompt)
    timings['prompt_ms'] = _elapsed_ms(prompt_start)

    return {
        'key': key,
        'timings': timings,
        'cached_payload': None,
        'chunks': prompt_chunks,
        'retrieved_chunks': all_chunks,
        'sources': build_sources(prompt_chunks),
        'prompt': prompt,
        'prompt_chars': prompt_chars,
        'resolved_question': resolved_question,
    }


def _sse_event(event: str, payload: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _sse_done() -> str:
    return f'data: {SSE_DONE_SENTINEL}\n\n'


def _structured_sources(question: str, answer: str, chunks: list[ContentChunk]) -> list[dict] | None:
    if answer and _is_fee_query(question):
        fee_chunks = _fee_chunks_for_answer(question, chunks)
        if fee_chunks:
            return build_sources(fee_chunks)
    if answer and _is_course_query(question):
        course_chunks = _course_chunks_for_answer(question, chunks)
        if course_chunks:
            if (_requested_course_period_number(question) is None
                    and _requested_class_number(question) is None
                    and not _is_full_course_list_query(question)):
                overview_chunks = [
                    chunk
                    for chunk in course_chunks
                    if _get_chunk_metadata_value(chunk, 'chunk_level') == 'program_overview'
                ]
                if overview_chunks:
                    return build_sources(overview_chunks[:1])
            return build_sources(course_chunks)
    if answer and _question_topics(question) & {'library', 'sports', 'campus_life', 'student_clubs', 'transport', 'prep', 'dormitory', 'international', 'scholarships'}:
        return build_sources(chunks)
    return None


@transaction.atomic
def chat(question: str, conversation_id: int | None = None) -> dict:
    overall_start = perf_counter()
    conversation = get_conversation(conversation_id, question)

    greeting_response = _get_greeting_response(question)
    if greeting_response:
        answer = greeting_response
        _persist_exchange(conversation, question, answer)
        return {
            'answer': answer,
            'conversation_id': conversation.id,
            'sources': [],
            'cached': False,
            'busy': False,
        }

    context = _prepare_chat_context(question, conversation)
    resolved_question = context['resolved_question']
    question_hash = _question_hash(resolved_question)
    timings = context['timings']
    cached_payload = context['cached_payload']
    sources = context['sources']

    if cached_payload is not None:
        persist_start = perf_counter()
        _persist_exchange(conversation, question, cached_payload['answer'])
        timings['persist_ms'] = _elapsed_ms(persist_start)
        timings['total_ms'] = _elapsed_ms(overall_start)
        _log_chat_timing(
            question_hash=question_hash,
            cached=True,
            conversation_id=conversation.id,
            chunk_count=0,
            source_count=len(sources),
            prompt_chars=0,
            timings=timings,
        )
        return {
            'answer': cached_payload['answer'],
            'conversation_id': conversation.id,
            'sources': sources,
            'cached': True,
            'busy': False,
        }

    chunks = context['chunks']
    retrieved_chunks = context['retrieved_chunks']
    prompt = context['prompt']
    prompt_chars = context['prompt_chars']
    llm_busy = False
    facility_chunks: list[ContentChunk] = []
    structured_answer = ''
    answer_type = ''
    if not chunks:
        answer = NO_CONTEXT_ANSWER
        timings['llm_ms'] = 0.0
    else:
        structured_answer = _build_structured_staff_answer(resolved_question, retrieved_chunks)
        if structured_answer:
            answer_type = 'staff'
        if not structured_answer:
            structured_answer = _build_program_presence_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type = 'program'
        if not structured_answer:
            structured_answer = _build_structured_score_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type = 'score'
        if not structured_answer:
            structured_answer = _build_structured_fee_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type = 'fee'
        if not structured_answer:
            structured_answer = _build_structured_course_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type = 'course'
        if not structured_answer:
            structured_answer = _build_general_info_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type = 'general_info'
        if not structured_answer:
            structured_answer, facility_chunks = _build_facility_info_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type = 'facility'

        if structured_answer and _should_bypass_structured_answer(answer_type):
            structured_answer = ''

        source_chunks = facility_chunks if facility_chunks else retrieved_chunks
        if structured_answer:
            answer = structured_answer
            structured_sources = _structured_sources(resolved_question, answer, source_chunks)
            if structured_sources is not None:
                sources = structured_sources
            timings['llm_ms'] = 0.0
        else:
            llm_start = perf_counter()
            try:
                answer = _generate_answer_with_slot(prompt) or NO_CONTEXT_ANSWER
            except LLMBusyError:
                answer = LLM_BUSY_ANSWER
                sources = []
                llm_busy = True
            timings['llm_ms'] = _elapsed_ms(llm_start)

    persist_start = perf_counter()
    _persist_exchange(conversation, question, answer)
    payload = {
        'answer': answer,
        'conversation_id': conversation.id,
        'sources': sources,
        'cached': False,
        'busy': llm_busy,
    }
    if not llm_busy and not _is_llm_format_error_answer(answer):
        cache.set(
            context['key'],
            {'answer': answer, 'sources': sources},
            timeout=settings.CACHE_TTL,
        )
    timings['persist_ms'] = _elapsed_ms(persist_start)
    timings['total_ms'] = _elapsed_ms(overall_start)
    _log_chat_timing(
        question_hash=question_hash,
        cached=False,
        conversation_id=conversation.id,
        chunk_count=len(chunks),
        source_count=len(sources),
        prompt_chars=prompt_chars,
        timings=timings,
    )
    return payload


def chat_stream(question: str, conversation_id: int | None = None) -> Iterator[str]:
    overall_start = perf_counter()
    conversation = get_conversation(conversation_id, question)

    def _event_stream() -> Iterator[str]:
        yield _sse_event(
            'meta',
            {
                'conversation_id': conversation.id,
                'cached': False,
            },
        )

        greeting_response = _get_greeting_response(question)
        if greeting_response:
            answer = greeting_response
            yield _sse_event('token', {'text': answer})
            _persist_exchange(conversation, question, answer)
            yield _sse_event('sources', {'sources': []})
            yield _sse_done()
            return

        context = _prepare_chat_context(question, conversation)
        resolved_question = context['resolved_question']
        question_hash = _question_hash(resolved_question)
        timings = context['timings']
        cached_payload = context['cached_payload']
        sources = context['sources']

        if cached_payload is not None:
            yield _sse_event(
                'meta',
                {
                    'conversation_id': conversation.id,
                    'cached': True,
                },
            )
            yield _sse_event('token', {'text': cached_payload['answer']})

            persist_start = perf_counter()
            _persist_exchange(conversation, question, cached_payload['answer'])
            timings['persist_ms'] = _elapsed_ms(persist_start)
            timings['total_ms'] = _elapsed_ms(overall_start)
            _log_chat_timing(
                question_hash=question_hash,
                cached=True,
                conversation_id=conversation.id,
                chunk_count=0,
                source_count=len(sources),
                prompt_chars=0,
                timings=timings,
            )
            yield _sse_event('sources', {'sources': sources})
            yield _sse_done()
            return

        chunks = context['chunks']
        retrieved_chunks = context['retrieved_chunks']
        answer = NO_CONTEXT_ANSWER
        facility_chunks_stream: list[ContentChunk] = []
        structured_answer = ''
        answer_type_stream = ''
        llm_busy = False
        if not chunks:
            timings['llm_ms'] = 0.0
            yield _sse_event('token', {'text': answer})
        else:
            structured_answer = _build_structured_staff_answer(resolved_question, retrieved_chunks)
            if structured_answer:
                answer_type_stream = 'staff'
            if not structured_answer:
                structured_answer = _build_program_presence_answer(resolved_question, retrieved_chunks)
                if structured_answer:
                    answer_type_stream = 'program'
            if not structured_answer:
                structured_answer = _build_structured_score_answer(resolved_question, retrieved_chunks)
                if structured_answer:
                    answer_type_stream = 'score'
            if not structured_answer:
                structured_answer = _build_structured_fee_answer(resolved_question, retrieved_chunks)
                if structured_answer:
                    answer_type_stream = 'fee'
            if not structured_answer:
                structured_answer = _build_structured_course_answer(resolved_question, retrieved_chunks)
                if structured_answer:
                    answer_type_stream = 'course'
            if not structured_answer:
                structured_answer = _build_general_info_answer(resolved_question, retrieved_chunks)
                if structured_answer:
                    answer_type_stream = 'general_info'
            if not structured_answer:
                structured_answer, facility_chunks_stream = _build_facility_info_answer(resolved_question, retrieved_chunks)
                if structured_answer:
                    answer_type_stream = 'facility'

            if structured_answer and _should_bypass_structured_answer(answer_type_stream):
                structured_answer = ''

            source_chunks_stream = facility_chunks_stream if facility_chunks_stream else retrieved_chunks
            if structured_answer:
                answer = structured_answer
                structured_sources = _structured_sources(resolved_question, answer, source_chunks_stream)
                if structured_sources is not None:
                    sources = structured_sources
                timings['llm_ms'] = 0.0
                yield _sse_event('token', {'text': answer})
            else:
                llm_start = perf_counter()
                answer_parts: list[str] = []
                try:
                    for token in _generate_answer_stream_with_slot(context['prompt']):
                        answer_parts.append(token)
                        yield _sse_event('token', {'text': token})
                except LLMBusyError:
                    answer = LLM_BUSY_ANSWER
                    sources = []
                    llm_busy = True
                    yield _sse_event('token', {'text': answer})
                timings['llm_ms'] = _elapsed_ms(llm_start)

                if not llm_busy:
                    streamed_answer = ''.join(answer_parts).strip()
                    if streamed_answer:
                        answer = streamed_answer
                    else:
                        yield _sse_event('token', {'text': answer})

        persist_start = perf_counter()
        _persist_exchange(conversation, question, answer)
        if not llm_busy and not _is_llm_format_error_answer(answer):
            cache.set(
                context['key'],
                {'answer': answer, 'sources': sources},
                timeout=settings.CACHE_TTL,
            )
        timings['persist_ms'] = _elapsed_ms(persist_start)
        timings['total_ms'] = _elapsed_ms(overall_start)
        _log_chat_timing(
            question_hash=question_hash,
            cached=False,
            conversation_id=conversation.id,
            chunk_count=len(chunks),
            source_count=len(sources),
            prompt_chars=context['prompt_chars'],
            timings=timings,
        )
        yield _sse_event('sources', {'sources': sources})
        yield _sse_done()

    return _event_stream()
