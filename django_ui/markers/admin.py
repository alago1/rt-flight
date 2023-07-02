from django.contrib.gis import admin
from markers.models import Marker

# Register your models here.
@admin.register(Marker)
class MarkerAdmin(admin.GISModelAdmin):
    list_display = ("name", "location", "confidence", "radius")
