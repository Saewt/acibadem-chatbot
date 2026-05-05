import logging
import time

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone

from scraper.knowledge_sync import get_sync_state, sync_due

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run the periodic knowledge-sync scheduler loop'

    def add_arguments(self, parser):
        parser.add_argument(
            '--once',
            action='store_true',
            help='Evaluate the schedule once and exit.',
        )

    def handle(self, *args, **options):
        poll_seconds = max(settings.KNOWLEDGE_SCHEDULER_POLL_SECONDS, 1)
        run_once = options['once']

        while True:
            if settings.KNOWLEDGE_SYNC_ENABLED:
                state = get_sync_state()
                if sync_due(state):
                    self.stdout.write('Knowledge sync due; running sync_acibadem_knowledge.')
                    try:
                        call_command('sync_acibadem_knowledge', stdout=self.stdout)
                    except Exception as exc:  # pragma: no cover - defensive scheduler guard
                        logger.exception('knowledge_scheduler_failed error=%s', exc)
                else:
                    if (
                        state.last_checked_at is None
                        and not settings.KNOWLEDGE_SYNC_RUN_ON_START
                    ):
                        state.last_checked_at = timezone.now()
                        state.last_status = 'idle'
                        state.save(update_fields=['last_checked_at', 'last_status'])
                    self.stdout.write('Knowledge sync not due yet.')
            else:
                self.stdout.write('Knowledge sync disabled; sleeping.')

            if run_once:
                return
            time.sleep(poll_seconds)
