from django.conf import settings
from django.core.management.base import BaseCommand

from chat.services import warm_embedding_model, warm_llm_model


class Command(BaseCommand):
    help = 'Warm chat embedding and LLM models before serving traffic'

    def add_arguments(self, parser):
        parser.add_argument(
            '--llm',
            action='store_true',
            help='Warm the LLM even when LLM_WARMUP_ENABLED is false.',
        )

    def handle(self, *args, **options):
        self.stdout.write('Starting model warmup...')

        if settings.EMBEDDING_BACKEND == 'local':
            try:
                warm_embedding_model()
            except Exception as exc:
                self.stdout.write(self.style.WARNING(
                    f'Warmup failed for embedding_model: {exc}'
                ))
            else:
                self.stdout.write(self.style.SUCCESS(
                    'Warmup completed for embedding_model.'
                ))
        else:
            self.stdout.write('Embedding backend=api, skipping local warmup.')

        if not (settings.LLM_WARMUP_ENABLED or options['llm']):
            self.stdout.write('LLM warmup skipped: disabled by settings.')
            return

        try:
            warm_llm_model()
        except Exception as exc:
            self.stdout.write(self.style.WARNING(
                f'Warmup failed for llm_model: {exc}'
            ))
        else:
            self.stdout.write(self.style.SUCCESS('Warmup completed for llm_model.'))
