# Generated by Django 2.0.3 on 2018-03-25 22:44

import autoslug.fields
from django.db import migrations
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('institutions', '0015_auto_20170830_1408'),
    ]

    operations = [
        migrations.AlterField(
            model_name='institution',
            name='extra',
            field=jsonfield.fields.JSONField(blank=True, verbose_name='Unorganized additional information'),
        ),
        migrations.AlterField(
            model_name='institution',
            name='slug',
            field=autoslug.fields.AutoSlugField(editable=False, populate_from='name', unique=True, verbose_name='Slug'),
        ),
        migrations.AlterField(
            model_name='tag',
            name='slug',
            field=autoslug.fields.AutoSlugField(editable=False, populate_from='name', verbose_name='Slug'),
        ),
    ]
