# Generated by Django 2.2.10 on 2020-02-22 16:22

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("cases", "0007_auto_20180325_2244"),
    ]

    operations = [
        migrations.AddField(
            model_name="case",
            name="mass_assign",
            field=models.UUIDField(
                blank=True, editable=False, null=True, verbose_name="Mass assign ID"
            ),
        ),
    ]
