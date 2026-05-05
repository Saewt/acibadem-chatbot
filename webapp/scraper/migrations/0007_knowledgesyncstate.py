from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0006_alter_webpage_source'),
    ]

    operations = [
        migrations.CreateModel(
            name='KnowledgeSyncState',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('key', models.CharField(max_length=100, unique=True)),
                ('last_checked_at', models.DateTimeField(blank=True, null=True)),
                ('last_success_at', models.DateTimeField(blank=True, null=True)),
                ('last_manifest_hash', models.CharField(blank=True, max_length=64)),
                ('last_status', models.CharField(default='idle', max_length=20)),
                ('last_error', models.TextField(blank=True)),
            ],
            options={
                'ordering': ['key'],
            },
        ),
    ]
