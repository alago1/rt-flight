from django.shortcuts import render
from django.views.generic.base import TemplateView

from markers.models import Marker

# Create your views here.
class MarkersMapView(TemplateView):
    queryset = Marker.objects.all()
    template_name = "map.html"
