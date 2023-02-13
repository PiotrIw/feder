# Generated by Django 3.2.16 on 2023-02-03 17:27

from django.db import migrations, models
import django_extensions.db.fields


class Migration(migrations.Migration):
    dependencies = [
        ("letters", "0028_auto_20230202_2022"),
    ]

    operations = [
        migrations.CreateModel(
            name="LetterEmailDomain",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "created",
                    django_extensions.db.fields.CreationDateTimeField(
                        auto_now_add=True, verbose_name="created"
                    ),
                ),
                (
                    "modified",
                    django_extensions.db.fields.ModificationDateTimeField(
                        auto_now=True, verbose_name="modified"
                    ),
                ),
                (
                    "domain_name",
                    models.CharField(
                        blank=True,
                        max_length=100,
                        null=True,
                        verbose_name="Email address domain",
                    ),
                ),
                (
                    "is_monitoring_email_to_domain",
                    models.BooleanField(
                        default=False, verbose_name="Is monitoring Email To domain?"
                    ),
                ),
                (
                    "is_spammer_domain",
                    models.BooleanField(
                        default=False, verbose_name="Is spammer domain?"
                    ),
                ),
                (
                    "email_to_count",
                    models.IntegerField(
                        default=0, verbose_name="Email To addres counter"
                    ),
                ),
                (
                    "email_from_count",
                    models.IntegerField(
                        default=0, verbose_name="Email From addres counter"
                    ),
                ),
            ],
            options={
                "get_latest_by": "modified",
                "abstract": False,
            },
        ),
    ]
