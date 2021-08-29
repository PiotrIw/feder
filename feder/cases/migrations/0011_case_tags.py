# Generated by Django 2.2.17 on 2021-01-28 17:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cases_tags", "0001_initial"),
        ("cases", "0010_auto_20200327_0040"),
    ]

    operations = [
        migrations.AddField(
            model_name="case",
            name="tags",
            field=models.ManyToManyField(
                blank=True, to="cases_tags.Tag", verbose_name="Tags"
            ),
        ),
    ]