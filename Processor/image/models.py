# models.py
from django.db import models
class TIFFImage(models.Model):
    image = models.FileField(upload_to='tiff_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class NDVIPlot(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='ndvi_plots/')
    created_at = models.DateTimeField(auto_now_add=True)

class Water(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='water_plots/')
    created_at = models.DateTimeField(auto_now_add=True)
