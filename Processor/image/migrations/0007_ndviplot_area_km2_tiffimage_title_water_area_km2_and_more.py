# Generated by Django 5.1.1 on 2025-04-13 11:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image', '0006_water'),
    ]

    operations = [
        migrations.AddField(
            model_name='ndviplot',
            name='area_km2',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='tiffimage',
            name='title',
            field=models.CharField(default='Uploaded Image', max_length=200),
        ),
        migrations.AddField(
            model_name='water',
            name='area_km2',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='ndviplot',
            name='title',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='tiffimage',
            name='image',
            field=models.FileField(upload_to='tiff_images/'),
        ),
        migrations.AlterField(
            model_name='water',
            name='title',
            field=models.CharField(max_length=200),
        ),
    ]
