from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Scrape obs.acibadem.edu.tr (Bologna system) using Playwright'

    def handle(self, *args, **options):
        # TODO: Phase 2.2 — implement Playwright scraper for Bologna system
        self.stdout.write('scrape_bologna: not yet implemented')
