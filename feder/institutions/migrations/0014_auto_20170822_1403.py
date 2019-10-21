# -*- coding: utf-8 -*-
# Generated by Django 1.10.7 on 2017-08-22 14:03
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [("institutions", "0013_auto_20170810_2118")]

    operations = [
        migrations.AlterField(
            model_name="tag",
            name="name",
            field=models.CharField(
                db_index=True, max_length=50, unique=True, verbose_name="Name"
            ),
        )
    ]
