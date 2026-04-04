from django.db import migrations
import pgvector.django


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0004_alter_webpage_options_alter_webpage_last_seen_at_and_more'),
    ]

    operations = [
        migrations.RunSQL(
            sql=(
                'CREATE INDEX IF NOT EXISTS contentchunk_embedding_hnsw_idx '
                'ON scraper_contentchunk USING hnsw (embedding vector_cosine_ops)'
            ),
            reverse_sql='DROP INDEX IF EXISTS contentchunk_embedding_hnsw_idx',
            state_operations=[
                migrations.AddIndex(
                    model_name='contentchunk',
                    index=pgvector.django.HnswIndex(
                        fields=['embedding'],
                        name='contentchunk_embedding_hnsw_idx',
                        opclasses=['vector_cosine_ops'],
                    ),
                )
            ],
        ),
    ]
