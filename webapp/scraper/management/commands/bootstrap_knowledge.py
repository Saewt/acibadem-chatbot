from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from scraper.models import WebPage

BOOTSTRAP_LOCK_ID = 38204117
REQUIRED_DATASET_FILES = (
    'acibadem_output/sources_clean.jsonl',
    'acibadem_output/chunks_clean.jsonl',
    'acibadem_output/records_clean.jsonl',
    'bologna_courses/sources.jsonl',
    'bologna_courses/records.jsonl',
    'bologna_courses/summary.json',
)


def _acquire_bootstrap_lock() -> bool:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_try_advisory_lock(%s)', [BOOTSTRAP_LOCK_ID])
        row = cursor.fetchone()
    return bool(row and row[0])


def _release_bootstrap_lock() -> None:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_advisory_unlock(%s)', [BOOTSTRAP_LOCK_ID])


def _missing_dataset_files(dataset_root: Path) -> list[Path]:
    return [path for relative_path in REQUIRED_DATASET_FILES if not (path := dataset_root / relative_path).exists()]


class Command(BaseCommand):
    help = 'Bootstrap the knowledge base from the mounted clean dataset when the database is empty'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-root',
            default=settings.ACIBADEM_DATASET_ROOT,
            help='Root directory containing the clean dataset snapshot.',
        )
        parser.add_argument(
            '--force-refresh',
            action='store_true',
            help='Recreate imported chunks even when page content hashes are unchanged.',
        )
        parser.add_argument(
            '--skip-embeddings',
            action='store_true',
            help='Import dataset rows without generating embeddings afterwards.',
        )
        parser.add_argument(
            '--rebuild-embeddings',
            action='store_true',
            help='Regenerate all embeddings after a successful import.',
        )

    def handle(self, *args, **options):
        if not settings.KNOWLEDGE_BOOTSTRAP_ENABLED:
            self.stdout.write(self.style.WARNING('bootstrap_knowledge skipped: disabled by settings.'))
            return

        if not _acquire_bootstrap_lock():
            self.stdout.write(
                self.style.WARNING('bootstrap_knowledge skipped: another process is already bootstrapping.')
            )
            return

        try:
            if WebPage.objects.filter(is_active=True).exists():
                self.stdout.write(
                    self.style.WARNING('bootstrap_knowledge no-op: active knowledge pages already exist.')
                )
                return

            dataset_root = Path(options['dataset_root'])
            missing_files = _missing_dataset_files(dataset_root)
            if missing_files:
                message = 'bootstrap_knowledge missing dataset files: ' + ', '.join(
                    str(path.relative_to(dataset_root)) for path in missing_files
                )
                if settings.KNOWLEDGE_BOOTSTRAP_FAIL_ON_MISSING_DATA:
                    raise CommandError(message)
                self.stdout.write(self.style.WARNING(message))
                return

            call_command(
                'import_acibadem_dataset',
                dataset_root=str(dataset_root),
                force_refresh=options['force_refresh'],
                skip_embeddings=options['skip_embeddings'],
                rebuild_embeddings=options['rebuild_embeddings'],
                stdout=self.stdout,
            )
        finally:
            _release_bootstrap_lock()
