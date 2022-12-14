# Generated by Django 4.1.1 on 2022-09-23 09:49

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Endpoint',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=128)),
                ('owner', models.CharField(max_length=128)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Measurements',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('measurement_id', models.IntegerField()),
                ('timestamp', models.BigIntegerField()),
                ('measurement_type', models.CharField(max_length=1)),
                ('limp_type', models.CharField(max_length=1)),
                ('limp_side', models.CharField(max_length=1)),
                ('v1', models.DecimalField(decimal_places=3, max_digits=6)),
                ('v2', models.DecimalField(decimal_places=3, max_digits=6)),
                ('v3', models.DecimalField(decimal_places=3, max_digits=6)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
