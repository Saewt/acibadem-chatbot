from django.core.management.base import BaseCommand

from scraper.services import build_session, crawl_candidate_data


class Command(BaseCommand):
    help = 'Scrape candidate-facing pages and persist normalized admissions data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force-refresh',
            action='store_true',
            help='Recreate chunks even when page content hash is unchanged.',
        )
        parser.add_argument(
            '--rate-limit-delay',
            type=float,
            default=1.0,
            help='Delay in seconds to apply before each request.',
        )

    def handle(self, *args, **options):
        client = build_session()
        try:
            summary = crawl_candidate_data(
                client=client,
                force_refresh=options['force_refresh'],
                rate_limit_delay=options['rate_limit_delay'],
            )
        finally:
            client.close()

        self.stdout.write(
            self.style.SUCCESS(
                'scrape_candidate_data completed '
                f"(seen={summary['seen']}, saved={summary['saved']}, "
                f"updated={summary['updated']}, structured_saved={summary['structured_saved']}, "
                f"deactivated={summary['deactivated']}, failed={summary['failed']})"
            )
        )
