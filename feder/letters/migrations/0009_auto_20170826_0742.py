# -*- coding: utf-8 -*-
# Generated by Django 1.10.7 on 2017-08-26 07:42
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [("letters", "0008_letter_message_id_field")]

    operations = [
        migrations.RenameField(
            model_name="letter",
            old_name="message_id_field",
            new_name="message_id_header",
        )
    ]
