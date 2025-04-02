from django.shortcuts import render, redirect
from .form import TIFFUploadForm
from .models import TIFFImage, NDVIPlot, Water
import rasterio as rio
import numpy as np
from io import BytesIO
from rasterio.plot import reshape_as_image
import matplotlib
import matplotlib.pyplot as plt
from django.core.files.base import ContentFile

# Use 'Agg' backend for Matplotlib
matplotlib.use('Agg')

def index(request):
    if request.method == 'POST':
        form = TIFFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            

            
            # Save the uploaded TIFF file
            uploaded_image = form.save()

            try:
                # Open the uploaded TIFF file
                with rio.open(uploaded_image.image.path) as src:
                    img_data = src.read()
                    tiff_image = reshape_as_image(img_data)

                # Extract RGB and NIR channels
                red, green, blue = tiff_image[:, :, 2], tiff_image[:, :, 1], tiff_image[:, :, 0]
                if tiff_image.shape[2] > 3:
                    nir = tiff_image[:, :, 3]
                    NDVI = (nir - red) / (nir + red + 1e-6)

                    # Save NDVI plot to buffer
                    plt.figure(figsize=(10, 10))
                    plt.imshow(NDVI, cmap='RdYlGn_r')
                    plt.axis('off')
                    plt.title('NDVI Map')

                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)

                    # Save plot to NDVIPlot model
                    plot = NDVIPlot(title='NDVI Plot')
                    plot.image.save('ndvi_plot.png', ContentFile(buffer.read()), save=True)
                    buffer.close()




                    ####WATER BODEIS
                    NDWI = (nir - green) / (nir + green + 1e-6)

                    ####WATER BODEIS
                    water_mask = NDVI > 0
                    rgb_water = np.zeros((*water_mask.shape, 3))
                    rgb_water[water_mask, 2] = 1  # Blue for water

                    # Plot the water detection map
                    plt.figure(figsize=(10, 10))
                    plt.imshow(rgb_water)
                    plt.axis('off')
                    plt.title('Detected Water Bodies')
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)

                    # Save plot to Water model
                    water = Water(title='NDVI Plot')
                    water.image.save('water_plot.png', ContentFile(buffer.read()), save=True)

                    buffer.close()
                    plt.close()
                    return render(request, 'index.html', {'form': form, 'water': water , 'plot':plot})

            except Exception as e:
                print(f"Error processing TIFF file: {e}")

    else:
        form = TIFFUploadForm()

    return render(request, 'index.html', {'form': form})