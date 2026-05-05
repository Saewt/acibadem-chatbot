from django.core.management.base import BaseCommand

from scraper.knowledge_sync import run_live_sync


class Command(BaseCommand):
    help = 'Check upstream Acibadem sources and refresh the knowledge base when content changes'

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true', help='Sync even when the manifest is unchanged.')
        parser.add_argument('--check-only', action='store_true', help='Only compare upstream content hashes.')
        parser.add_argument(
            '--rebuild-embeddings',
            action='store_true',
            help='Regenerate all embeddings after a successful sync.',
        )
        parser.add_argument(
            '--force-refresh',
            action='store_true',
            help='Recreate chunks even when scraped page content hashes are unchanged.',
        )
        parser.add_argument(
            '--rate-limit-delay',
            type=float,
            default=1.0,
            help='Delay in seconds to apply before each upstream request.',
        )
        parser.add_argument(
            '--max-pages',
            type=int,
            default=500,
            help='Maximum number of main-site pages to crawl during a refresh.',
        )

    def handle(self, *args, **options):
        summary = run_live_sync(
            force=options['force'],
            check_only=options['check_only'],
            rebuild_embeddings=options['rebuild_embeddings'],
            force_refresh=options['force_refresh'],
            rate_limit_delay=options['rate_limit_delay'],
            max_pages=options['max_pages'],
        )
        if summary.get('skipped'):
            self.stdout.write(
                self.style.WARNING(
                    f"sync_acibadem_knowledge skipped reason={summary['reason']}"
                )
            )
            return

        if options['check_only']:
            status = 'changed' if summary['changed'] else 'unchanged'
            self.stdout.write(
                self.style.SUCCESS(
                    f"sync_acibadem_knowledge check completed status={status} "
                    f"manifest_hash={summary['manifest_hash']}"
                )
            )
            return

        if not summary['changed']:
            self.stdout.write(
                self.style.WARNING(
                    f"sync_acibadem_knowledge no-op manifest_hash={summary['manifest_hash']}"
                )
            )
            return

        self.stdout.write(
            self.style.SUCCESS(
                'sync_acibadem_knowledge completed '
                f"(main_site_saved={summary['main_site']['saved']}, "
                f"candidate_saved={summary['candidate']['saved']}, "
                f"bologna_saved={summary['bologna']['saved']}, "
                f"manifest_hash={summary['manifest_hash']})"
            )
        )
