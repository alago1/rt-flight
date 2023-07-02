from django.urls import path
from markers.views import MarkersMapView

app_name = "markers"

urlpatterns = [
    path('map/', MarkersMapView.as_view()),
]
