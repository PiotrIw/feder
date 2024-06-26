# Generated by Django 4.2.13 on 2024-05-20 11:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('llm_evaluation', '0009_llmletterrequest_args_llmletterrequest_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='llmletterrequest',
            name='token_usage',
            field=models.JSONField(blank=True, null=True, verbose_name='LLM Engine token usage'),
        ),
        migrations.AlterField(
            model_name='llmmonitoringrequest',
            name='token_usage',
            field=models.JSONField(blank=True, null=True, verbose_name='LLM Engine token usage'),
        ),
    ]
