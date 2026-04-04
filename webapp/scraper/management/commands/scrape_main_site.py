from django.core.management.base import BaseCommand

from scraper.services import DEFAULT_MAIN_SITE_SEEDS, build_session, crawl_candidate_data, crawl_main_site


class Command(BaseCommand):
    help = 'Scrape acibadem.edu.tr and persist cleaned content into pgvector-backed tables'

    def add_arguments(self, parser):
        parser.add_argument(
            '--seed',
            action='append',
            dest='seeds',
            default=[],
            help='Optional seed URL. Can be provided multiple times.',
        )
        parser.add_argument(
            '--max-pages',
            type=int,
            default=500,
            help='Maximum number of pages to crawl.',
        )
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
        seeds = options['seeds'] or list(DEFAULT_MAIN_SITE_SEEDS)
        client = build_session()
        try:
            summary = crawl_main_site(
                client=client,
                seeds=seeds,
                max_pages=options['max_pages'],
                force_refresh=options['force_refresh'],
                rate_limit_delay=options['rate_limit_delay'],
            )
            candidate_summary = crawl_candidate_data(
                client=client,
                force_refresh=options['force_refresh'],
                rate_limit_delay=options['rate_limit_delay'],
            )
        finally:
            client.close()
        self.stdout.write(
            self.style.SUCCESS(
                'scrape_main_site completed '
                f"(seen={summary['seen']}, saved={summary['saved']}, "
                f"updated={summary['updated']}, deactivated={summary['deactivated']}, "
                f"failed={summary['failed']}, candidate_seen={candidate_summary['seen']}, "
                f"candidate_saved={candidate_summary['saved']}, "
                f"candidate_structured_saved={candidate_summary['structured_saved']}, "
                f"candidate_failed={candidate_summary['failed']})"
            )
        )
