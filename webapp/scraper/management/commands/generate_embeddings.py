from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Generate embeddings for ContentChunks using Ollama nomic-embed-text'

    def handle(self, *args, **options):
        # TODO: Phase 2.3 — implement embedding generation with pgvector HNSW index
        self.stdout.write('generate_embeddings: not yet implemented')
