# Processor Django Project

This repository contains a Django-based web application for processing and visualizing geospatial data, including NDVI and water body analysis using satellite imagery.

## Features
- Upload and process satellite TIFF images
- Generate NDVI and water body plots
- Visualize results with interactive charts
- User-friendly web interface

## Tech Stack
- Python 3.12+
- Django
- Earth Engine API
- Rasterio
- NumPy, Pandas, Matplotlib, scikit-learn
- Pillow

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd Processor
```

### 2. Install dependencies
```bash
pip install -r requirement.txt
```

### 3. Apply migrations
```bash
python manage.py migrate
```

### 4. Run the development server
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser to access the application.

## Project Structure
```
Processor/
├── credentials.json
├── db.sqlite3
├── manage.py
├── requirement.txt
├── downloaded_image/
├── image/
│   ├── ...
├── media/
│   ├── img/
│   ├── ndvi_plots/
│   ├── plots/
│   ├── tiff_files/
│   ├── tiff_images/
│   └── water_plots/
├── static/
│   └── style.css
│   └── images/
├── templates/
│   ├── index.html
│   └── statistics.html
└── ...
```

## License
This project is licensed under the MIT License.

## Acknowledgements
- [Google Earth Engine](https://earthengine.google.com/)
- [Django](https://www.djangoproject.com/)
- [Rasterio](https://rasterio.readthedocs.io/)
