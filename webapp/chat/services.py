import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Iterator
from time import perf_counter

from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.core.cache import cache
from django.db import transaction
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
QUESTION_HASH_LENGTH = 12
CACHE_KEY_VERSION = 'v3'
CANDIDATE_LIMIT_MULTIPLIER = 3
SSE_DONE_SENTINEL = '[DONE]'
STAFF_QUERY_PATTERN = re.compile(
    r'\b('
    r'hoca\w*|'
    r'akademik\s*kadro\w*|'
    r'öğretim\s*üye\w*|'
    r'ogretim\s*uye\w*|'
    r'profesör\w*|'
    r'profesor\w*|'
    r'kimler|'
    r'listele\w*|'
    r'kaç|'
    r'kac'
    r')\b'
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


class ConversationNotFoundError(Exception):
    pass


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=settings.MODEL_RUNNER_BASE_URL,
        api_key=getattr(settings, 'MODEL_RUNNER_API_KEY', 'not-needed'),
        timeout=120.0,
    )


def warm_embedding_model() -> None:
    embed_text('warmup')


def warm_llm_model() -> None:
    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0,
        max_tokens=1,
        messages=[
            {
                'role': 'system',
                'content': 'Return a one-word acknowledgement.',
            },
            {'role': 'user', 'content': 'Ping'},
        ],
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
    tokens = [
        token
        for token in _normalize_lookup_text(value).split()
        if token and token not in {'kaç', 'kac'}
    ]
    if not tokens:
        return value

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


def _chunk_priority(question: str, chunk: ContentChunk) -> int:
    kind = _get_chunk_metadata_value(chunk, 'kind')
    staff_page_type = _get_chunk_metadata_value(chunk, 'staff_page_type')
    topic = _get_chunk_metadata_value(chunk, 'topic')
    staff_count_text = _get_chunk_metadata_value(chunk, 'staff_count')
    try:
        staff_count = int(staff_count_text) if staff_count_text else 0
    except ValueError:
        staff_count = 0

    if _is_staff_query(question):
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
        if kind == 'structured_admissions_score':
            return 0
        if kind == 'candidate_topic_page' and topic == 'admissions_scores':
            return 1
        if kind == 'structured_admissions_fee':
            return 4
        return 5

    if _is_fee_query(question):
        if kind == 'structured_admissions_fee':
            return 0
        if kind == 'candidate_topic_page' and topic == 'tuition':
            return 1
        if kind == 'structured_admissions_score':
            return 4
        return 5

    question_topics = _question_topics(question)
    if question_topics:
        if kind == 'candidate_topic_page' and topic in question_topics:
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
    if _is_score_query(question):
        return [
            chunk
            for chunk in chunks
            if _get_chunk_metadata_value(chunk, 'kind') == 'structured_admissions_score'
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
            or (
                _get_chunk_metadata_value(chunk, 'kind') == 'candidate_topic_page'
                and _get_chunk_metadata_value(chunk, 'topic') == 'tuition'
            )
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
    elif _is_fee_query(question):
        rules.append('Ücret sorularında yalnızca resmi öğrenim ücreti veya ilgili aday öğrenci kaynaklarını kullan.')
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
        _get_chunk_metadata_value(chunk, 'program_title')
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
        'Sen Acibadem Universitesi icin resmi kaynaklardan cevap veren bir asistansin.\n'
        'Yalnizca verilen baglama dayan.\n'
        'Baglam yetmiyorsa bunu acikca soyle ve tahmin yapma.\n'
        'Program veya fakulte odakli sorularda yalnizca metadata olarak eslesen kaynaklari kullan.\n'
        'Cevabi Turkce ver.\n'
        'Metin icinde kaynak numaralarini [1], [2] gibi kullan.\n\n'
        f'Ek kurallar:\n{rules_block}\n'
        f'Baglam:\n{context}\n\n'
        f'Kullanici sorusu: {question}'
    )
    return prompt, used_chunks


def generate_answer(prompt: str) -> str:
    response = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.1,
        messages=[
            {
                'role': 'system',
                'content': 'Resmi universite kaynagina dayali, kisa ve net cevaplar ver.',
            },
            {'role': 'user', 'content': prompt},
        ],
    )
    return (response.choices[0].message.content or '').strip()


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
    stream = get_llm_client().chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.1,
        messages=[
            {
                'role': 'system',
                'content': 'Resmi universite kaynagina dayali, kisa ve net cevaplar ver.',
            },
            {'role': 'user', 'content': prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        yield from _iter_delta_content_text(delta)


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
    combined = _sort_candidate_chunks(question, vector_chunks + keyword_chunks)
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


def _prepare_chat_context(question: str) -> dict:
    key = cache_key(question)
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
        }

    embed_start = perf_counter()
    query_embedding = embed_query(question)
    timings['embed_ms'] = _elapsed_ms(embed_start)

    retrieve_start = perf_counter()
    chunks = _retrieve_candidates(question, query_embedding)
    timings['retrieve_ms'] = _elapsed_ms(retrieve_start)
    scoped_chunks, had_scope = _apply_scope_filter(question, chunks)
    if had_scope:
        chunks = scoped_chunks

    prompt = ''
    prompt_chars = 0
    prompt_start = perf_counter()
    if chunks:
        prompt, chunks = build_prompt(question, chunks)
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
    }


def _sse_event(event: str, payload: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _sse_done() -> str:
    return f'data: {SSE_DONE_SENTINEL}\n\n'


@transaction.atomic
def chat(question: str, conversation_id: int | None = None) -> dict:
    overall_start = perf_counter()
    question_hash = _question_hash(question)
    conversation = get_conversation(conversation_id, question)
    context = _prepare_chat_context(question)
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
        }

    chunks = context['chunks']
    prompt = context['prompt']
    prompt_chars = context['prompt_chars']
    if not chunks:
        answer = NO_CONTEXT_ANSWER
        timings['llm_ms'] = 0.0
    else:
        structured_answer = _build_structured_score_answer(question, chunks)
        if structured_answer:
            answer = structured_answer
            timings['llm_ms'] = 0.0
        else:
            llm_start = perf_counter()
            answer = generate_answer(prompt) or NO_CONTEXT_ANSWER
            timings['llm_ms'] = _elapsed_ms(llm_start)

    persist_start = perf_counter()
    _persist_exchange(conversation, question, answer)
    payload = {
        'answer': answer,
        'conversation_id': conversation.id,
        'sources': sources,
        'cached': False,
    }
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
    question_hash = _question_hash(question)
    conversation = get_conversation(conversation_id, question)

    def _event_stream() -> Iterator[str]:
        yield _sse_event(
            'meta',
            {
                'conversation_id': conversation.id,
                'cached': False,
            },
        )

        context = _prepare_chat_context(question)
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
        if not chunks:
            timings['llm_ms'] = 0.0
            yield _sse_event('token', {'text': answer})
        else:
            structured_answer = _build_structured_score_answer(question, chunks)
            if structured_answer:
                answer = structured_answer
                timings['llm_ms'] = 0.0
                yield _sse_event('token', {'text': answer})
            else:
                llm_start = perf_counter()
                answer_parts: list[str] = []
                for token in generate_answer_stream(context['prompt']):
                    answer_parts.append(token)
                    yield _sse_event('token', {'text': token})
                timings['llm_ms'] = _elapsed_ms(llm_start)

                streamed_answer = ''.join(answer_parts).strip()
                if streamed_answer:
                    answer = streamed_answer
                else:
                    yield _sse_event('token', {'text': answer})

        persist_start = perf_counter()
        _persist_exchange(conversation, question, answer)
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
