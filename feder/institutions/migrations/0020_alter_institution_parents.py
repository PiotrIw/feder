# Generated by Django 4.2.11 on 2024-03-21 10:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('institutions', '0019_alter_institution_tags'),
    ]

    operations = [
        migrations.AlterField(
            model_name='institution',
            name='parents',
            field=models.ManyToManyField(blank=True, to='institutions.institution', verbose_name='Parent institutions'),
        ),
    ]
