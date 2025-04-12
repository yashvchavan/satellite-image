# form.py
from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import date

class LocationForm(forms.Form):
    latitude = forms.FloatField(
        label='Latitude',
        validators=[MinValueValidator(-90), MaxValueValidator(90)],
        widget=forms.NumberInput(attrs={
            'step': '0.000001',
            'placeholder': 'e.g., 19.0760'
        })
    )
    
    longitude = forms.FloatField(
        label='Longitude',
        validators=[MinValueValidator(-180), MaxValueValidator(180)],
        widget=forms.NumberInput(attrs={
            'step': '0.000001',
            'placeholder': 'e.g., 72.8777'
        })
    )
    
    start_date = forms.DateField(
        label='Start Date',
        widget=forms.DateInput(attrs={'type': 'date'}),
        initial=date.today().replace(month=1, day=1)
    )
    
    end_date = forms.DateField(
        label='End Date',
        widget=forms.DateInput(attrs={'type': 'date'}),
        initial=date.today()
    )
    
    buffer_size = forms.FloatField(
        label='Area Size (km)',
        initial=5,
        validators=[MinValueValidator(1), MaxValueValidator(50)],
        help_text='Size of area to analyze in kilometers'
    )
    
    cloud_cover = forms.IntegerField(
        label='Max Cloud Cover (%)',
        initial=20,
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )