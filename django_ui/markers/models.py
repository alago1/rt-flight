from django.contrib.gis.db import models


class Marker(models.Model):
    name = models.CharField(max_length=255)
    location = models.PointField()
    confidence = models.FloatField(default=0.0)
    radius = models.FloatField(default=1e-2)

    def __str__(self):
        return self.name
