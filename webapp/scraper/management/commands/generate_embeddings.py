from django.core.management.base import BaseCommand, CommandError

from scraper.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    iter_text_embedding_batches,
)
from scraper.models import ContentChunk
from scraper.services import build_chunk_embedding_text

DEFAULT_FETCH_SIZE = 128


class Command(BaseCommand):
    help = 'Generate embeddings for ContentChunks using sentence-transformers'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=DEFAULT_EMBEDDING_BATCH_SIZE,
            help='Number of chunks to embed per model batch.',
        )
        parser.add_argument(
            '--fetch-size',
            type=int,
            default=DEFAULT_FETCH_SIZE,
            help='Number of chunk rows to fetch from the database per iteration.',
        )
        parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Regenerate embeddings for all chunks, including existing ones.',
        )

    def _iter_chunk_windows(self, queryset, fetch_size: int):
        last_id = 0
        while True:
            rows = list(
                queryset.filter(id__gt=last_id)
                .order_by('id')
                .values_list('id', 'text', 'metadata')[:fetch_size]
            )
            if not rows:
                return
            yield rows
            last_id = rows[-1][0]

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        fetch_size = options['fetch_size']
        if batch_size < 1:
            raise CommandError('--batch-size must be greater than 0')
        if fetch_size < 1:
            raise CommandError('--fetch-size must be greater than 0')

        queryset = ContentChunk.objects.all()
        if not options['rebuild']:
            queryset = queryset.filter(embedding__isnull=True)
        total = queryset.count()
        if total == 0:
            self.stdout.write(self.style.WARNING('No chunks require embeddings.'))
            return

        processed = 0
        for rows in self._iter_chunk_windows(queryset, fetch_size):
            texts = [build_chunk_embedding_text(text, metadata) for _, text, metadata in rows]
            for start, vectors in iter_text_embedding_batches(texts, batch_size=batch_size):
                batch_rows = rows[start : start + len(vectors)]
                updates = [
                    ContentChunk(id=chunk_id, embedding=vector)
                    for (chunk_id, _, _), vector in zip(batch_rows, vectors, strict=True)
                ]
                ContentChunk.objects.bulk_update(updates, ['embedding'])
                processed += len(updates)
            self.stdout.write(f'Embedded {processed}/{total} chunks.')

        self.stdout.write(
            self.style.SUCCESS(
                'Generated embeddings for '
                f'{processed} chunks (batch_size={batch_size}, fetch_size={fetch_size}).'
            )
        )
