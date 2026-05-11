from django.core.management.base import BaseCommand

from scraper.models import ContentChunk, WebPage
from scraper.services import infer_general_topic_metadata


def _infer_and_apply(page: WebPage, force: bool = False) -> tuple[bool, int]:
    page_metadata = dict(page.metadata or {})
    if not force and page_metadata.get('topic'):
        return False, 0

    topic_metadata = infer_general_topic_metadata(page.url, page.title or '')
    if not topic_metadata:
        return False, 0

    page_changed = False
    for key, value in topic_metadata.items():
        if not page_metadata.get(key) or page_metadata.get(key) != value:
            page_metadata[key] = value
            page_changed = True
    if page_changed:
        page.metadata = page_metadata
        page.save(update_fields=['metadata'])

    chunk_updates = 0
    for chunk in page.chunks.all():
        chunk_metadata = dict(chunk.metadata or {})
        chunk_changed = False
        for key, value in topic_metadata.items():
            if not chunk_metadata.get(key) or chunk_metadata.get(key) != value:
                chunk_metadata[key] = value
                chunk_changed = True
        if chunk_changed:
            chunk.metadata = chunk_metadata
            chunk.save(update_fields=['metadata'])
            chunk_updates += 1

    return page_changed, chunk_updates


def repair_general_page_metadata(force: bool = False) -> tuple[int, int, int]:
    page_updates = 0
    chunk_updates = 0
    pages_checked = 0

    pages = WebPage.objects.filter(
        is_active=True,
    ).exclude(
        metadata__topic__isnull=False,
    ).prefetch_related('chunks')

    for page in pages.iterator(chunk_size=500):
        pages_checked += 1
        page_updated, chunks_updated = _infer_and_apply(page, force=force)
        if page_updated:
            page_updates += 1
        chunk_updates += chunks_updated

    return pages_checked, page_updates, chunk_updates


class Command(BaseCommand):
    help = 'Backfill topic metadata on imported general pages and chunks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Overwrite existing topic metadata',
        )

    def handle(self, *args, **options):
        force = options['force']
        pages_checked, page_updates, chunk_updates = repair_general_page_metadata(
            force=force,
        )
        self.stdout.write(
            self.style.SUCCESS(
                'repair_general_page_metadata completed '
                f'(checked={pages_checked}, pages={page_updates}, chunks={chunk_updates}).'
            )
        )
