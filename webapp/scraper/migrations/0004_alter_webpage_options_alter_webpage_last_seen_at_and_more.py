from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0003_webpage_metadata_and_constraints'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='webpage',
            options={'ordering': ['title', 'url']},
        ),
        migrations.AlterField(
            model_name='webpage',
            name='last_seen_at',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AlterField(
            model_name='webpage',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
