# Generated by Django 4.2.20 on 2025-04-26 19:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('virus_scan', '0003_alter_request_engine_id_alter_request_engine_link_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='request',
            name='object_id',
            field=models.PositiveIntegerField(verbose_name='Attachment ID'),
        ),
    ]
