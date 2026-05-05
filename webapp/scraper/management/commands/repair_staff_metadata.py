from django.core.management.base import BaseCommand

from scraper.models import ContentChunk, WebPage
from scraper.services import _build_program_alias_text


def _clean_text(value: str) -> str:
    return ' '.join(str(value or '').split()).strip()


def _program_title_from_page(page: WebPage) -> str:
    title = _clean_text(page.title)
    if ' - ' in title:
        title = title.split(' - ', 1)[0]
    if title and title.lower() not in {'akademik kadro', 'akademik personel'}:
        return title
    return ''


def _program_title_from_chunks(page: WebPage) -> str:
    values = (
        page.chunks.exclude(metadata__program_title='')
        .values_list('metadata__program_title', flat=True)
        .distinct()
    )
    for value in values:
        value = _clean_text(value)
        if value:
            return value

    values = (
        page.chunks.exclude(metadata__unit_name='')
        .values_list('metadata__unit_name', flat=True)
        .distinct()
    )
    for value in values:
        value = _clean_text(value)
        if value:
            return value
    return ''


def repair_staff_metadata() -> tuple[int, int]:
    page_updates = 0
    chunk_updates = 0
    pages = WebPage.objects.filter(
        is_active=True,
        metadata__kind='main_site_staff_page',
    ).prefetch_related('chunks')

    for page in pages:
        metadata = dict(page.metadata or {})
        program_title = (
            _clean_text(metadata.get('program_title', ''))
            or _program_title_from_chunks(page)
            or _program_title_from_page(page)
        )
        if not program_title:
            continue

        alias_text = _build_program_alias_text(program_title=program_title)
        changed = False
        for key, value in (
            ('program_title', program_title),
            ('program_alias_text', alias_text),
        ):
            if not metadata.get(key):
                metadata[key] = value
                changed = True
        if changed:
            page.metadata = metadata
            page.save(update_fields=['metadata'])
            page_updates += 1

        for chunk in page.chunks.all():
            chunk_metadata = dict(chunk.metadata or {})
            chunk_changed = False
            for key, value in (
                ('program_title', program_title),
                ('program_alias_text', alias_text),
            ):
                if not chunk_metadata.get(key):
                    chunk_metadata[key] = value
                    chunk_changed = True
            if chunk_changed:
                chunk.metadata = chunk_metadata
                chunk.save(update_fields=['metadata'])
                chunk_updates += 1

    return page_updates, chunk_updates


class Command(BaseCommand):
    help = 'Backfill missing program metadata on imported staff pages and chunks'

    def handle(self, *args, **options):
        page_updates, chunk_updates = repair_staff_metadata()
        self.stdout.write(
            self.style.SUCCESS(
                'repair_staff_metadata completed '
                f'(pages={page_updates}, chunks={chunk_updates}).'
            )
        )
