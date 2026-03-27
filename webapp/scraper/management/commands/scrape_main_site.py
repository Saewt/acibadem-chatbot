from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Scrape acibadem.edu.tr using Scrapy'

    def handle(self, *args, **options):
        # TODO: Phase 2.1 — implement Scrapy spider for acibadem.edu.tr
        self.stdout.write('scrape_main_site: not yet implemented')
