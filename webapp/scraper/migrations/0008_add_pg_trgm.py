from django.contrib.postgres.operations import TrigramExtension
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0007_knowledgesyncstate'),
    ]

    operations = [
        TrigramExtension(),
        migrations.RunSQL(
            sql=(
                'CREATE INDEX IF NOT EXISTS trgm_content_chunk_text_idx '
                'ON scraper_contentchunk USING GIN (text gin_trgm_ops);'
            ),
            reverse_sql='DROP INDEX IF EXISTS trgm_content_chunk_text_idx;',
        ),
    ]
