# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cases', '0004_auto_20150803_1806'),
        ('letters', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='letter',
            name='case',
            field=models.ForeignKey(default=1, to='cases.Case'),
            preserve_default=False,
        ),
    ]
