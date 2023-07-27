from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class BBox:
    latitude: float
    longitude: float 
    radius: float
    confidence: float


@dataclass(slots=True)
class BBoxes:
    bboxes: List[BBox]
