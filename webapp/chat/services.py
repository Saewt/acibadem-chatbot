import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections.abc import Iterator
from time import perf_counter

import requests
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.core.cache import cache
from django.db import transaction
from django.db.models import Q
from openai import OpenAI
from pgvector.django import CosineDistance

from scraper.embeddings import embed_text
from scraper.models import ContentChunk

from .models import Conversation, Message

logger = logging.getLogger(__name__)
NO_CONTEXT_ANSWER = (
    'Bu konuda doğrulanmış üniversite kaynağı bulamadım. '
    'İstersen soruyu daha spesifik sorabilir veya başka bir resmi sayfayı hedefleyebilirsin.'
)
LLM_BUSY_ANSWER = (
    'Model şu anda başka bir yanıt üretiyor. '
    'Lütfen birkaç saniye sonra tekrar deneyin.'
)
QUESTION_HASH_LENGTH = 12
CACHE_KEY_VERSION = 'v7'
CANDIDATE_LIMIT_MULTIPLIER = 3
SSE_DONE_SENTINEL = '[DONE]'
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
PROGRAM_EXISTS_QUERY_PATTERN = re.compile(
    r'\b(var\s*m[ıi]|bulunuyor\s*mu|mevcut\s*mu|aç[ıi]k\s*m[ıi])\b'
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
    r'tavan\s*puan\w*'
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
SCHOLARSHIP_QUERY_PATTERN = re.compile(r'\b(burs\w*|scholarship\w*)\b')
DORMITORY_QUERY_PATTERN = re.compile(r'\b(yurt\w*|depozito\w*|konaklama\w*|dorm\w*)\b')
INTERNATIONAL_QUERY_PATTERN = re.compile(
    r'\b(uluslararası\w*|uluslararasi\w*|erasmus\w*|yurtdış\w*|yurtdis\w*|hareketlilik\w*)\b'
)
DOUBLE_MAJOR_MINOR_QUERY_PATTERN = re.compile(
    r'\b(çift\s*anadal\w*|cift\s*anadal\w*|yandal\w*|çap\w*|cap\w*|minor\w*|major\w*)\b'
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
    r'akts\w*|'
    r'ects\w*|'
    r'semester\w*'
    r')\b'
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
}


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
    return str(payload.get('message', {}).get('content') or '').strip()


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
    for candidate in _known_program_candidates():
        for alias in _build_scope_aliases(candidate):
            if _question_mentions_alias(normalized_text, alias):
                return _clean_display_text(candidate)
    return ''


def _extract_program_hint_from_text(text: str) -> str:
    for field, aliases in _extract_question_scope_hints(text):
        if field == 'program_title' and aliases:
            return max(aliases, key=len)
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
    return bool(_extract_known_program_from_text(question))


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


def _build_scope_constraint(question: str, chunks: list[ContentChunk]) -> dict | None:
    normalized_question = _normalize_lookup_text(question)
    if not normalized_question:
        return None

    for field in ('program_alias_text', 'placement_label', 'program_title', 'faculty'):
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
        if _build_scope_aliases(_get_chunk_metadata_value(chunk, scope['field']))
        & scope['aliases']
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


def _is_program_exists_query(question: str) -> bool:
    return bool(PROGRAM_EXISTS_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_dentistry_query(question: str) -> bool:
    return bool(DENTISTRY_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_score_query(question: str) -> bool:
    return bool(SCORE_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_fee_query(question: str) -> bool:
    return bool(FEE_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_rank_query(question: str) -> bool:
    return bool(RANK_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_points_query(question: str) -> bool:
    return bool(POINTS_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_quota_query(question: str) -> bool:
    return bool(QUOTA_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _is_course_query(question: str) -> bool:
    return bool(COURSE_QUERY_PATTERN.search(_normalize_lookup_text(question)))


def _question_topics(question: str) -> set[str]:
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
    return topics


def _chunk_matches_question_topic(question_topics: set[str], chunk: ContentChunk) -> bool:
    if not question_topics:
        return False

    topic = _get_chunk_metadata_value(chunk, 'topic')
    if topic in question_topics:
        return True

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
        for keyword in TOPIC_KEYWORDS.get(question_topic, ()):
            if _normalize_lookup_text(keyword) in searchable_text:
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
        if record_type == 'academic_staff_member':
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
        if chunk_level == 'semester_plan':
            return 0
        if chunk_level == 'program_overview':
            return 1
        if kind == 'bologna_program_page':
            return 2
        return 5

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


def _filter_candidates_for_query(question: str, chunks: list[ContentChunk]) -> list[ContentChunk]:
    if _is_staff_query(question):
        staff_chunks = [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'kind') in {'bologna_staff_page', 'main_site_staff_page'}
            or _get_chunk_metadata_value(chunk, 'record_type') in {'academic_staff_member', 'department_head_message'}
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
            return filtered

    question_topics = _question_topics(question)
    if question_topics:
        return [
            chunk
            for chunk in chunks
            if _chunk_matches_question_topic(question_topics, chunk)
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
    elif _question_topics(question):
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


def _build_structured_fee_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_fee_query(question):
        return ''

    fee_chunks = [
        chunk
        for chunk in chunks
        if _get_chunk_metadata_value(chunk, 'kind') == 'structured_admissions_fee'
        or _get_chunk_metadata_value(chunk, 'record_type') == 'tuition_fee'
    ]
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
    for chunk in sorted(
        fee_chunks,
        key=lambda item: (
            _get_chunk_metadata_value(item, 'program_title') or item.page.title,
            item.page.title,
        ),
    ):
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


def _build_structured_staff_answer(question: str, chunks: list[ContentChunk]) -> str:
    if not _is_staff_query(question):
        return ''

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

    program_title = _extract_known_program_from_text(question) or _extract_program_hint_from_text(question)
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


def retrieve_context(
    query_embedding: list[float], limit: int | None = None, per_page_limit: int | None = None
) -> list[ContentChunk]:
    limit = limit or settings.RAG_RETRIEVE_LIMIT
    per_page_limit = per_page_limit or settings.RAG_PER_PAGE_LIMIT
    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(embedding__isnull=False, page__is_active=True)
        .annotate(distance=CosineDistance('embedding', query_embedding))
        .order_by('distance')
    )
    selected: list[ContentChunk] = []
    for chunk in queryset[: limit * 4]:
        if chunk.distance is None or chunk.distance > 0.72:
            continue
        selected.append(chunk)
        if len(selected) >= limit * 2:
            break
    return _limit_chunks(selected, limit=limit, per_page_limit=per_page_limit)


def retrieve_keyword_context(
    question: str, limit: int | None = None, per_page_limit: int | None = None
) -> list[ContentChunk]:
    limit = limit or settings.RAG_RETRIEVE_LIMIT
    per_page_limit = per_page_limit or settings.RAG_PER_PAGE_LIMIT
    normalized_question = ' '.join(question.split())
    if not normalized_question:
        return []

    query = SearchQuery(normalized_question, search_type='plain', config='simple')
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
    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .annotate(rank=SearchRank(vector, query))
        .filter(rank__gt=0)
        .order_by('-rank', 'page_id', 'chunk_index')
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


def _select_prompt_chunks(chunks: list[ContentChunk]) -> list[tuple[ContentChunk, str]]:
    max_chunk_chars = settings.RAG_MAX_CHUNK_CHARS
    max_context_chars = settings.RAG_MAX_CONTEXT_CHARS
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
    prompt_chunks = _select_prompt_chunks(chunks)
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
        'Sen Acıbadem Üniversitesi için resmi kaynaklardan cevap veren bir asistansın.\n'
        'Yalnızca verilen bağlama dayan.\n'
        'Bağlamda ilgili bilgi varsa bu bilgiden yararlanarak cevap ver, eksik kısımları belirt.\n'
        'Bağlamda hiç ilgili bilgi yoksa bunu açıkça söyle.\n'
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
            'content': 'Resmi üniversite kaynağına dayalı, kısa ve net cevaplar ver.',
        },
        {'role': 'user', 'content': prompt},
    ]
    if _use_ollama_backend():
        return _ollama_chat(
            messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.1,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=messages,
    )
    return (response.choices[0].message.content or '').strip()


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
            'content': 'Resmi üniversite kaynağına dayalı, kısa ve net cevaplar ver.',
        },
        {'role': 'user', 'content': prompt},
    ]
    if _use_ollama_backend():
        yield from _ollama_chat_stream(
            messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return

    stream = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.1,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        yield from _iter_delta_content_text(delta)


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

    program_title = _extract_known_program_from_text(question) or _extract_program_hint_from_text(question)
    if not program_title:
        return []

    program_lookup = Q()
    for term in _program_lookup_terms(program_title):
        program_lookup |= (
            Q(metadata__program_title__icontains=term)
            | Q(metadata__unit_name__icontains=term)
            | Q(metadata__program_alias_text__icontains=term)
            | Q(page__title__icontains=term)
            | Q(text__icontains=term)
        )

    queryset = (
        ContentChunk.objects.select_related('page')
        .filter(page__is_active=True)
        .filter(
            Q(metadata__kind='main_site_staff_page')
            | Q(metadata__kind='bologna_staff_page')
            | Q(metadata__record_type='academic_staff_member')
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
        program_title = _extract_known_program_from_text(question) or _extract_program_hint_from_text(question)
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
    )
    keyword_chunks = retrieve_keyword_context(
        question,
        limit=candidate_limit,
        per_page_limit=candidate_per_page_limit,
    )
    direct_chunks = _retrieve_direct_staff_chunks(
        question, limit=candidate_limit * 2
    ) + _retrieve_direct_program_chunks(question, limit=candidate_limit)
    combined = _sort_candidate_chunks(question, direct_chunks + vector_chunks + keyword_chunks)
    combined = _filter_candidates_for_query(question, combined)
    scoped_chunks, had_scope = _apply_scope_filter(question, combined)
    if had_scope and not scoped_chunks:
        scoped_chunks = combined
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

    retrieve_start = perf_counter()
    chunks = _retrieve_candidates(resolved_question, query_embedding)
    timings['retrieve_ms'] = _elapsed_ms(retrieve_start)
    prompt = ''
    prompt_chars = 0
    prompt_start = perf_counter()
    if chunks:
        prompt, chunks = build_prompt(resolved_question, chunks)
        prompt_chars = len(prompt)
    timings['prompt_ms'] = _elapsed_ms(prompt_start)

    return {
        'key': key,
        'timings': timings,
        'cached_payload': None,
        'chunks': chunks,
        'sources': build_sources(chunks),
        'prompt': prompt,
        'prompt_chars': prompt_chars,
        'resolved_question': resolved_question,
    }


def _sse_event(event: str, payload: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _sse_done() -> str:
    return f'data: {SSE_DONE_SENTINEL}\n\n'


@transaction.atomic
def chat(question: str, conversation_id: int | None = None) -> dict:
    overall_start = perf_counter()
    conversation = get_conversation(conversation_id, question)
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
    prompt = context['prompt']
    prompt_chars = context['prompt_chars']
    llm_busy = False
    if not chunks:
        answer = NO_CONTEXT_ANSWER
        timings['llm_ms'] = 0.0
    else:
        structured_answer = _build_structured_staff_answer(resolved_question, chunks)
        if not structured_answer:
            structured_answer = _build_program_presence_answer(resolved_question, chunks)
        if not structured_answer:
            structured_answer = _build_structured_score_answer(resolved_question, chunks)
        if not structured_answer:
            structured_answer = _build_structured_fee_answer(resolved_question, chunks)
        if structured_answer:
            answer = structured_answer
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
    if not llm_busy:
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
        answer = NO_CONTEXT_ANSWER
        llm_busy = False
        if not chunks:
            timings['llm_ms'] = 0.0
            yield _sse_event('token', {'text': answer})
        else:
            structured_answer = _build_structured_staff_answer(resolved_question, chunks)
            if not structured_answer:
                structured_answer = _build_program_presence_answer(resolved_question, chunks)
            if not structured_answer:
                structured_answer = _build_structured_score_answer(resolved_question, chunks)
            if not structured_answer:
                structured_answer = _build_structured_fee_answer(resolved_question, chunks)
            if structured_answer:
                answer = structured_answer
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
        if not llm_busy:
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
