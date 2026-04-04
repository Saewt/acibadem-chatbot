from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0005_contentchunk_hnsw_index'),
    ]

    operations = [
        migrations.AlterField(
            model_name='webpage',
            name='source',
            field=models.CharField(
                choices=[
                    ('main_site', 'Main Site'),
                    ('bologna', 'Bologna'),
                    ('structured', 'Structured'),
                ],
                default='main_site',
                max_length=20,
            ),
        ),
    ]
