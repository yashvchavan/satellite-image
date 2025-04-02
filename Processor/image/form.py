# forms.py
from django import forms
from .models import TIFFImage

class TIFFUploadForm(forms.ModelForm):
    class Meta:
        model = TIFFImage
        fields = ['image']


class LocationForm(forms.Form):
    latitude = forms.FloatField(label='Latitude', required=True)
    longitude = forms.FloatField(label='Longitude', required=True)