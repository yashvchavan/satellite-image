# models.py
from django.db import models
from django.contrib.gis.db import models as gis_models

class AnalysisResult(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    start_date = models.DateField()
    end_date = models.DateField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Plots
    rgb_plot = models.ImageField(upload_to='results/rgb/')
    cir_plot = models.ImageField(upload_to='results/cir/')
    ndvi_plot = models.ImageField(upload_to='results/ndvi/')
    evi_plot = models.ImageField(upload_to='results/evi/')
    savi_plot = models.ImageField(upload_to='results/savi/')
    ndwi_plot = models.ImageField(upload_to='results/ndwi/')
    mndwi_plot = models.ImageField(upload_to='results/mndwi/')
    ndbi_plot = models.ImageField(upload_to='results/ndbi/')
    ui_plot = models.ImageField(upload_to='results/ui/')
    land_cover_plot = models.ImageField(upload_to='results/land_cover/')
    time_series_plot = models.ImageField(upload_to='results/timeseries/', null=True, blank=True)
    
    # Data
    statistics = models.JSONField()
    time_series_data = models.JSONField()
    
    # GeoDjango field for spatial queries
    location = gis_models.PointField(null=True, blank=True)
    
    def __str__(self):
        return f"Analysis at ({self.latitude}, {self.longitude}) on {self.timestamp}"
    
    def save(self, *args, **kwargs):
        from django.contrib.gis.geos import Point
        self.location = Point(self.longitude, self.latitude)
        super().save(*args, **kwargs)