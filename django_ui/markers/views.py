from django.shortcuts import render
from django.views.generic.base import TemplateView

# Create your views here.
class MarkersMapView(TemplateView):
    template_name = "map.html"
