from rest_framework import routers
from markers.viewsets import MarkerViewSet

routers = routers.DefaultRouter()
routers.register(r"markers", MarkerViewSet)

urlpatterns = routers.urls
