from django.core.management.base import BaseCommand

from chat.services import warm_embedding_model, warm_llm_model


class Command(BaseCommand):
    help = 'Warm chat embedding and LLM models before serving traffic'

    def handle(self, *args, **options):
        self.stdout.write('Starting model warmup...')
        for label, warmup in (
            ('embedding_model', warm_embedding_model),
            ('llm_model', warm_llm_model),
        ):
            try:
                warmup()
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f'Warmup failed for {label}: {exc}'))
            else:
                self.stdout.write(self.style.SUCCESS(f'Warmup completed for {label}.'))
