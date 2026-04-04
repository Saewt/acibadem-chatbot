from django.core.management.base import BaseCommand

from scraper.services import build_session, crawl_bologna


class Command(BaseCommand):
    help = 'Scrape obs.acibadem.edu.tr Bologna pages and persist cleaned content'

    def add_arguments(self, parser):
        parser.add_argument(
            '--unit-type',
            action='append',
            dest='unit_types',
            choices=['myo', 'lis', 'yls', 'dok'],
            default=[],
            help='Program type to scrape. Can be passed multiple times.',
        )
        parser.add_argument(
            '--skip-general-pages',
            action='store_true',
            help='Skip Bologna general content pages and ingest only program pages.',
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
            help='Delay in seconds to apply before each Bologna request.',
        )

    def handle(self, *args, **options):
        client = build_session()
        try:
            summary = crawl_bologna(
                client=client,
                unit_types=options['unit_types'] or ['myo', 'lis', 'yls', 'dok'],
                include_general_pages=not options['skip_general_pages'],
                force_refresh=options['force_refresh'],
                rate_limit_delay=options['rate_limit_delay'],
            )
        finally:
            client.close()
        self.stdout.write(
            self.style.SUCCESS(
                'scrape_bologna completed '
                f"(seen={summary['seen']}, saved={summary['saved']}, "
                f"updated={summary['updated']}, deactivated={summary['deactivated']}, "
                f"failed={summary['failed']})"
            )
        )
