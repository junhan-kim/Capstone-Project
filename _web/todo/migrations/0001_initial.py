# Generated by Django 3.0.5 on 2020-05-02 18:17

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Todo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=5, verbose_name='Name')),
                ('todo', models.CharField(max_length=50, verbose_name='TODO')),
            ],
        ),
    ]