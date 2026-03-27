from django.db import migrations, models
import django.db.models.deletion
import pgvector.django


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.RunSQL(
            sql='CREATE EXTENSION IF NOT EXISTS vector',
            reverse_sql='DROP EXTENSION IF EXISTS vector',
        ),
        migrations.CreateModel(
            name='WebPage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.URLField(max_length=500, unique=True)),
                ('title', models.CharField(blank=True, max_length=500)),
                ('content_text', models.TextField(blank=True)),
                ('raw_html', models.TextField(blank=True)),
                ('scraped_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='ContentChunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('embedding', pgvector.django.VectorField(blank=True, dimensions=768, null=True)),
                ('chunk_index', models.IntegerField(default=0)),
                ('metadata', models.JSONField(blank=True, default=dict)),
                ('page', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='chunks',
                    to='scraper.webpage',
                )),
            ],
            options={
                'ordering': ['page', 'chunk_index'],
            },
        ),
    ]
