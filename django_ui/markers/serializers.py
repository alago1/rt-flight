from rest_framework_gis import serializers
from markers.models import Marker

class MarkerSerializer(serializers.GeoFeatureModelSerializer):
    class Meta:
        fields = ("id", "name", "confidence", "radius")
        geo_field = "location"
        model = Marker
