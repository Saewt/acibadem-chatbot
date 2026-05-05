from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand

from scraper.dataset_import import import_acibadem_dataset


class Command(BaseCommand):
    help = 'Import the cleaned Acibadem dataset JSONL files into WebPage/ContentChunk tables'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-root',
            default=settings.ACIBADEM_DATASET_ROOT,
            help='Root directory containing the cleaned scraping outputs.',
        )
        parser.add_argument(
            '--force-refresh',
            action='store_true',
            help='Recreate page chunks even when the imported content hash is unchanged.',
        )
        parser.add_argument(
            '--skip-embeddings',
            action='store_true',
            help='Import chunks without running the embedding generation step afterwards.',
        )
        parser.add_argument(
            '--rebuild-embeddings',
            action='store_true',
            help='Regenerate embeddings for all chunks after the import finishes.',
        )

    def handle(self, *args, **options):
        summary = import_acibadem_dataset(
            dataset_root=options['dataset_root'],
            force_refresh=options['force_refresh'],
        )
        self.stdout.write(
            self.style.SUCCESS(
                'import_acibadem_dataset completed '
                f"(pages={summary['pages']}, chunks={summary['chunks']}, "
                f"main_site_pages={summary['main_site_pages']}, "
                f"structured_pages={summary['structured_pages']}, "
                f"bologna_pages={summary['bologna_pages']})"
            )
        )

        if options['skip_embeddings']:
            return

        call_command(
            'generate_embeddings',
            rebuild=options['rebuild_embeddings'],
            stdout=self.stdout,
        )
