from dataclasses import dataclass
from typing import List

@dataclass
class BBox:
    latitude: float
    longitude: float 
    radius: float
    confidence: float


@dataclass
class BBoxes:
    bboxes: List[BBox]