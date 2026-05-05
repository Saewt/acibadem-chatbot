import json
import logging
from datetime import timedelta

from django.conf import settings
from django.core.management import call_command
from django.db import connection
from django.utils import timezone

from scraper.models import KnowledgeSyncState
from scraper.services import (
    ALL_BOLOGNA_UNIT_TYPES,
    DEFAULT_CANDIDATE_ROOT_URL,
    DEFAULT_CANDIDATE_TOPIC_URLS,
    DEFAULT_MAIN_SITE_SEEDS,
    _absolute_bologna_url,
    build_session,
    crawl_bologna,
    crawl_candidate_data,
    crawl_main_site,
    fetch_html,
    hash_content,
    normalize_whitespace,
)

logger = logging.getLogger(__name__)
KNOWLEDGE_REFRESH_LOCK_ID = 38204117


def _acquire_refresh_lock() -> bool:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_try_advisory_lock(%s)', [KNOWLEDGE_REFRESH_LOCK_ID])
        row = cursor.fetchone()
    return bool(row and row[0])


def _release_refresh_lock() -> None:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_advisory_unlock(%s)', [KNOWLEDGE_REFRESH_LOCK_ID])


def _watched_urls() -> list[str]:
    urls = list(DEFAULT_MAIN_SITE_SEEDS)
    urls.append(DEFAULT_CANDIDATE_ROOT_URL)
    urls.extend(DEFAULT_CANDIDATE_TOPIC_URLS)
    urls.append('https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=tr')
    urls.extend(
        _absolute_bologna_url(f'unitSelection.aspx?type={unit_type}&lang=tr')
        for unit_type in sorted(ALL_BOLOGNA_UNIT_TYPES)
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def build_remote_manifest(*, rate_limit_delay: float) -> dict[str, str]:
    manifest: dict[str, str] = {}
    client = build_session()
    try:
        for url in _watched_urls():
            html = fetch_html(client, url, rate_limit_delay=rate_limit_delay)
            if html is None:
                raise RuntimeError(f'Failed to fetch sync manifest source: {url}')
            manifest[url] = hash_content(normalize_whitespace(html))
    finally:
        client.close()
    return manifest


def manifest_hash(manifest: dict[str, str]) -> str:
    return hash_content(json.dumps(manifest, sort_keys=True))


def get_sync_state() -> KnowledgeSyncState:
    state, _created = KnowledgeSyncState.objects.get_or_create(
        key=settings.KNOWLEDGE_SYNC_KEY,
    )
    return state


def sync_due(state: KnowledgeSyncState | None = None) -> bool:
    state = state or get_sync_state()
    if state.last_checked_at is None:
        return settings.KNOWLEDGE_SYNC_RUN_ON_START
    due_after = timedelta(hours=settings.KNOWLEDGE_SYNC_INTERVAL_HOURS)
    return timezone.now() - state.last_checked_at >= due_after


def run_live_sync(
    *,
    force: bool = False,
    check_only: bool = False,
    rebuild_embeddings: bool = False,
    force_refresh: bool = False,
    rate_limit_delay: float = 1.0,
    max_pages: int = 500,
) -> dict:
    if not _acquire_refresh_lock():
        return {'changed': False, 'skipped': True, 'reason': 'refresh_lock_held'}

    try:
        state = get_sync_state()
        state.last_checked_at = timezone.now()
        state.last_status = 'checking'
        state.last_error = ''
        state.save(update_fields=['last_checked_at', 'last_status', 'last_error'])

        manifest = build_remote_manifest(rate_limit_delay=rate_limit_delay)
        current_manifest_hash = manifest_hash(manifest)
        changed = force or current_manifest_hash != state.last_manifest_hash

        if check_only:
            state.last_status = 'checked'
            state.save(update_fields=['last_status'])
            return {
                'changed': changed,
                'manifest_hash': current_manifest_hash,
                'last_manifest_hash': state.last_manifest_hash,
            }

        if not changed:
            state.last_status = 'idle'
            state.save(update_fields=['last_status'])
            return {
                'changed': False,
                'manifest_hash': current_manifest_hash,
                'last_manifest_hash': state.last_manifest_hash,
            }

        client = build_session()
        try:
            main_summary = crawl_main_site(
                client=client,
                seeds=list(DEFAULT_MAIN_SITE_SEEDS),
                max_pages=max_pages,
                force_refresh=force_refresh,
                rate_limit_delay=rate_limit_delay,
            )
            candidate_summary = crawl_candidate_data(
                client=client,
                force_refresh=force_refresh,
                rate_limit_delay=rate_limit_delay,
            )
            bologna_summary = crawl_bologna(
                client=client,
                unit_types=sorted(ALL_BOLOGNA_UNIT_TYPES),
                include_general_pages=True,
                force_refresh=force_refresh,
                rate_limit_delay=rate_limit_delay,
            )
        finally:
            client.close()

        call_command('generate_embeddings', rebuild=rebuild_embeddings)

        state.last_manifest_hash = current_manifest_hash
        state.last_success_at = timezone.now()
        state.last_status = 'success'
        state.last_error = ''
        state.save(
            update_fields=[
                'last_manifest_hash',
                'last_success_at',
                'last_status',
                'last_error',
            ]
        )
        return {
            'changed': True,
            'manifest_hash': current_manifest_hash,
            'main_site': main_summary,
            'candidate': candidate_summary,
            'bologna': bologna_summary,
        }
    finally:
        _release_refresh_lock()
