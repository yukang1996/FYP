# Generated by Django 2.1.1 on 2018-10-12 10:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0002_auto_20181012_1834'),
    ]

    operations = [
        migrations.AlterField(
            model_name='snippet',
            name='image_pothole',
            field=models.ImageField(blank=True, upload_to='pothole'),
        ),
        migrations.AlterField(
            model_name='snippet',
            name='image_ref1',
            field=models.ImageField(blank=True, upload_to='ref'),
        ),
        migrations.AlterField(
            model_name='snippet',
            name='image_ref2',
            field=models.ImageField(blank=True, upload_to='ref'),
        ),
    ]
