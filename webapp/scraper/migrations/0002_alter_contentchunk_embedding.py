from django.db import migrations
import pgvector.django


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='contentchunk',
            name='embedding',
            field=pgvector.django.VectorField(blank=True, dimensions=384, null=True),
        ),
    ]
