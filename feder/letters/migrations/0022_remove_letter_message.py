# Generated by Django 1.11.16 on 2018-10-21 02:20

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("letters", "0021_auto_20180325_2244")]

    operations = [migrations.RemoveField(model_name="letter", name="message")]
