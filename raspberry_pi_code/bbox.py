from dataclasses import dataclass
from typing import List

@dataclass
class BBox:
    latitude: float
    longitude: float 
    radius: float
    confidence: float

    def __init__(self, latitude: float, longitude: float, radius: float, confidence: float):
        self.latitude = latitude
        self.longitude = longitude 
        self.radius = radius
        self.confidence = confidence

@dataclass
class BBoxes:
    bboxes: List[BBox]