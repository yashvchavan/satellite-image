# views.py
import os
import logging
from datetime import datetime, timedelta
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from .form import LocationForm
from .models import AnalysisResult
import ee
import numpy as np
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
import requests
import zipfile
from io import BytesIO
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
try:
    ee.Authenticate()
    ee.Initialize(project='ee-yashvchavan07')
except Exception as e:
    logger.error(f"Earth Engine initialization failed: {e}")

class SatelliteImageProcessor:
    def __init__(self, latitude, longitude, buffer_degrees=0.05):
        self.latitude = latitude
        self.longitude = longitude
        self.geometry = ee.Geometry.Rectangle([
            longitude - buffer_degrees, latitude - buffer_degrees,
            longitude + buffer_degrees, latitude + buffer_degrees
        ])
        self.temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def get_cloud_mask(self, image):
        """Improved cloud masking for Sentinel-2"""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
               .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)
    
    def get_sentinel2_collection(self, start_date, end_date, max_cloud=20):
        """Get processed Sentinel-2 collection"""
        collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(self.geometry) \
            .filterDate(start_date, end_date) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', max_cloud) \
            .map(self.get_cloud_mask) \
            .map(lambda img: img.multiply(0.0001))  # Scale reflectance
            
        return collection.median().clip(self.geometry)
    
    def download_image(self, image, bands, scale=10):
        """Download image data from Earth Engine"""
        url = image.select(bands).getDownloadURL({
            'scale': scale,
            'region': self.geometry,
            'fileFormat': 'GeoTIFF',
            'formatOptions': {'cloudOptimized': True}
        })
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(self.temp_dir)
                return [f for f in z.namelist() if f.endswith('.tif')]
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def calculate_indices(self, bands_data):
        """Calculate various vegetation and water indices"""
        red = bands_data.get('B4')
        green = bands_data.get('B3')
        blue = bands_data.get('B2')
        nir = bands_data.get('B8')
        swir1 = bands_data.get('B11', None)
        
        indices = {}
        
        # Vegetation indices
        indices['NDVI'] = (nir - red) / (nir + red + 1e-6)
        indices['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        indices['SAVI'] = 1.5 * (nir - red) / (nir + red + 0.5)
        
        # Water indices
        indices['NDWI'] = (green - nir) / (green + nir)
        indices['MNDWI'] = (green - swir1) / (green + swir1) if swir1 is not None else None
        
        # Urban/soil indices
        indices['NDBI'] = (swir1 - nir) / (swir1 + nir) if swir1 is not None else None
        indices['UI'] = (swir1 - nir) / (swir1 + nir) if swir1 is not None else None
        
        return indices
    
    def classify_land_cover(self, bands_data, n_clusters=5):
        """Perform unsupervised land cover classification"""
        # Stack all bands for classification
        bands = [bands_data[b] for b in ['B2', 'B3', 'B4', 'B8'] if b in bands_data]
        stacked = np.stack(bands, axis=-1)
        
        # Reshape for clustering
        h, w, d = stacked.shape
        X = stacked.reshape(-1, d)
        
        # Reduce dimensionality with PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        
        return labels.reshape(h, w)
    
    def generate_plots(self, bands_data, indices, classification):
        """Generate all analysis plots"""
        plots = {}
        
        # RGB True Color Image
        plots['rgb'] = self._plot_rgb(bands_data)
        
        # False Color Infrared
        plots['cir'] = self._plot_cir(bands_data)
        
        # Individual Indices
        for idx_name, idx_data in indices.items():
            if idx_data is not None:
                plots[idx_name] = self._plot_index(idx_data, idx_name)
        
        # Land Cover Classification
        plots['land_cover'] = self._plot_classification(classification)
        
        return plots
    
    def _plot_rgb(self, bands_data):
        """Plot RGB true color image"""
        rgb = np.stack([bands_data['B4'], bands_data['B3'], bands_data['B2']], axis=-1)
        rgb = np.clip(rgb * 3.5, 0, 1)  # Enhance brightness
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title('True Color (RGB)')
        plt.axis('off')
        return self._save_plot_to_buffer()
    
    def _plot_cir(self, bands_data):
        """Plot false color infrared"""
        cir = np.stack([bands_data['B8'], bands_data['B4'], bands_data['B3']], axis=-1)
        cir = np.clip(cir * 2.5, 0, 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cir)
        plt.title('False Color Infrared (CIR)')
        plt.axis('off')
        return self._save_plot_to_buffer()
    
    def _plot_index(self, index_data, title):
        """Plot a single index with custom colormap"""
        plt.figure(figsize=(10, 10))
        
        if 'NDVI' in title or 'EVI' in title or 'SAVI' in title:
            cmap = 'RdYlGn'
            vmin, vmax = -1, 1
        elif 'NDWI' in title or 'MNDWI' in title:
            cmap = 'Blues'
            vmin, vmax = -1, 1
        else:
            cmap = 'viridis'
            vmin, vmax = None, None
        
        plt.imshow(index_data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f'{title} Map')
        plt.axis('off')
        return self._save_plot_to_buffer()
    
    def _plot_classification(self, classification):
        """Plot land cover classification"""
        # Define colors and labels for classes
        class_colors = ['#00b300', '#006400', '#ffff00', '#ff0000', '#0000ff']
        class_labels = ['Forest', 'Dense Vegetation', 'Bare Soil', 'Urban', 'Water']
        
        # Create colormap
        cmap = colors.ListedColormap(class_colors)
        bounds = np.arange(len(class_colors) + 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(classification, cmap=cmap, norm=norm)
        
        # Create legend
        patches = [Patch(color=color, label=label) 
                  for color, label in zip(class_colors, class_labels)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Land Cover Classification')
        plt.axis('off')
        return self._save_plot_to_buffer()
    
    def _save_plot_to_buffer(self):
        """Save matplotlib plot to in-memory buffer"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        return buffer
    
    def calculate_statistics(self, indices):
        """Calculate basic statistics for each index"""
        stats = {}
        for name, data in indices.items():
            if data is not None:
                stats[name] = {
                    'mean': float(np.nanmean(data)),
                    'std': float(np.nanstd(data)),
                    'min': float(np.nanmin(data)),
                    'max': float(np.nanmax(data)),
                    'median': float(np.nanmedian(data))
                }
        return stats
    
    def process_time_series(self, start_date, end_date, interval_days=30):
        """Process time series data"""
        dates = pd.date_range(start_date, end_date, freq=f'{interval_days}D')
        time_series = []
        
        for i in range(len(dates) - 1):
            current_start = dates[i].strftime('%Y-%m-%d')
            current_end = dates[i+1].strftime('%Y-%m-%d')
            
            try:
                image = self.get_sentinel2_collection(current_start, current_end)
                bands = ['B2', 'B3', 'B4', 'B8', 'B11']
                files = self.download_image(image, bands)
                
                bands_data = {}
                for band in bands:
                    band_file = next(f for f in files if band in f)
                    with rio.open(os.path.join(self.temp_dir, band_file)) as src:
                        bands_data[band] = src.read(1)
                
                indices = self.calculate_indices(bands_data)
                stats = {k: float(np.nanmean(v)) for k, v in indices.items() if v is not None}
                stats['date'] = current_start
                time_series.append(stats)
                
            except Exception as e:
                logger.warning(f"Failed to process {current_start} to {current_end}: {e}")
                continue
        
        return time_series
    
    def generate_time_series_plot(self, time_series):
        """Generate time series plot of vegetation indices"""
        if not time_series:
            return None
            
        df = pd.DataFrame(time_series)
        df['date'] = pd.to_datetime(df['date'])
        
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            if col not in ['date', 'NDBI', 'UI'] and not pd.api.types.is_numeric_dtype(df[col]):
                plt.plot(df['date'], df[col], label=col)
        
        plt.title('Vegetation Index Time Series')
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.legend()
        plt.grid(True)
        
        return self._save_plot_to_buffer()

def index(request):
    if request.method == 'POST':
        form = LocationForm(request.POST)
        if form.is_valid():
            latitude = form.cleaned_data['latitude']
            longitude = form.cleaned_data['longitude']
            start_date = form.cleaned_data.get('start_date', '2022-01-01')
            end_date = form.cleaned_data.get('end_date', '2022-12-31')
            
            try:
                processor = SatelliteImageProcessor(latitude, longitude)
                
                # Process main image
                image = processor.get_sentinel2_collection(start_date, end_date)
                bands = ['B2', 'B3', 'B4', 'B8', 'B11']  # Blue, Green, Red, NIR, SWIR
                files = processor.download_image(image, bands)
                
                # Read bands
                bands_data = {}
                for band in bands:
                    band_file = next(f for f in files if band in f)
                    with rio.open(os.path.join(processor.temp_dir, band_file)) as src:
                        bands_data[band] = src.read(1)
                
                # Calculate indices
                indices = processor.calculate_indices(bands_data)
                
                # Classify land cover
                classification = processor.classify_land_cover(bands_data)
                
                # Generate plots
                plots = processor.generate_plots(bands_data, indices, classification)
                
                # Calculate statistics
                stats = processor.calculate_statistics(indices)
                
                # Process time series
                time_series = processor.process_time_series(start_date, end_date)
                time_series_plot = processor.generate_time_series_plot(time_series)
                
                # Save results to database
                result = AnalysisResult(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    statistics=json.dumps(stats),
                    time_series_data=json.dumps(time_series)
                )
                
                # Save all plots
                for name, buffer in plots.items():
                    if buffer:
                        getattr(result, f'{name}_plot').save(
                            f'{name}_{latitude}_{longitude}.png',
                            ContentFile(buffer.getvalue()),
                            save=False
                        )
                
                if time_series_plot:
                    result.time_series_plot.save(
                        f'timeseries_{latitude}_{longitude}.png',
                        ContentFile(time_series_plot.getvalue()),
                        save=False
                    )
                
                result.save()
                
                # Prepare context for template
                context = {
                    'form': form,
                    'result': result,
                    'statistics': stats,
                    'time_series': time_series[:10],  # First 10 entries for display
                    'plots': {name: getattr(result, f'{name}_plot').url 
                            for name in plots.keys()},
                    'time_series_plot': result.time_series_plot.url if time_series_plot else None
                }
                
                return render(request, 'index.html', context)
                
            except Exception as e:
                logger.error(f"Processing failed: {e}")
                return render(request, 'index.html', {
                    'form': form,
                    'error': f"Processing failed: {str(e)}"
                })
    
    else:
        # Set default dates (last 6 months)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        form = LocationForm(initial={'start_date': start_date, 'end_date': end_date})
    
    return render(request, 'index.html', {'form': form})

def get_time_series_data(request):
    """API endpoint for time series data"""
    result_id = request.GET.get('result_id')
    try:
        result = AnalysisResult.objects.get(id=result_id)
        return JsonResponse({
            'time_series': json.loads(result.time_series_data),
            'status': 'success'
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})