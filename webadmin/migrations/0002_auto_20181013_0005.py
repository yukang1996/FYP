# Generated by Django 2.1.1 on 2018-10-12 16:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webadmin', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='snippet',
            name='admin_name',
            field=models.CharField(default=0, max_length=100),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='snippet',
            name='file',
            field=models.FileField(blank=True, upload_to=''),
        ),
    ]
