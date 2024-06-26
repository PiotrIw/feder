# Generated by Django 4.2.13 on 2024-05-17 17:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('llm_evaluation', '0008_alter_llmmonthlycost_options'),
    ]

    operations = [
        migrations.AddField(
            model_name='llmletterrequest',
            name='args',
            field=models.JSONField(blank=True, null=True, verbose_name='Arguments'),
        ),
        migrations.AddField(
            model_name='llmletterrequest',
            name='name',
            field=models.CharField(blank=True, max_length=100, null=True, verbose_name='Name'),
        ),
        migrations.AddField(
            model_name='llmmonitoringrequest',
            name='args',
            field=models.JSONField(blank=True, null=True, verbose_name='Arguments'),
        ),
        migrations.AddField(
            model_name='llmmonitoringrequest',
            name='name',
            field=models.CharField(blank=True, max_length=100, null=True, verbose_name='Name'),
        ),
    ]
