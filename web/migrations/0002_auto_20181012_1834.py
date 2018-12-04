# Generated by Django 2.1.1 on 2018-10-12 10:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='snippet',
            name='image_pothole',
            field=models.ImageField(blank=True, upload_to='./web/media/pothole'),
        ),
        migrations.AddField(
            model_name='snippet',
            name='image_ref1',
            field=models.ImageField(blank=True, upload_to='./web/media/ref'),
        ),
        migrations.AddField(
            model_name='snippet',
            name='image_ref2',
            field=models.ImageField(blank=True, upload_to='./web/media/ref'),
        ),
    ]
