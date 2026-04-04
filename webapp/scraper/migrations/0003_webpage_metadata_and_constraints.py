from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0002_alter_contentchunk_embedding'),
    ]

    operations = [
        migrations.AddField(
            model_name='webpage',
            name='content_hash',
            field=models.CharField(blank=True, default='', max_length=64),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='webpage',
            name='is_active',
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name='webpage',
            name='language',
            field=models.CharField(default='tr', max_length=10),
        ),
        migrations.AddField(
            model_name='webpage',
            name='last_seen_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='webpage',
            name='metadata',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='webpage',
            name='source',
            field=models.CharField(
                choices=[('main_site', 'Main Site'), ('bologna', 'Bologna')],
                default='main_site',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='webpage',
            name='updated_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddConstraint(
            model_name='contentchunk',
            constraint=models.UniqueConstraint(
                fields=('page', 'chunk_index'), name='unique_chunk_per_page_position'
            ),
        ),
    ]
