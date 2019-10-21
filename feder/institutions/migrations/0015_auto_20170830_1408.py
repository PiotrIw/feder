# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-08-30 14:08
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [("institutions", "0014_auto_20170822_1403")]

    operations = [
        migrations.AlterField(
            model_name="institution",
            name="parents",
            field=models.ManyToManyField(
                blank=True,
                related_name="_institution_parents_+",
                to="institutions.Institution",
                verbose_name="Parent institutions",
            ),
        )
    ]
