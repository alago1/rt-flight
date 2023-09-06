from typing import Tuple

import numpy as np

from .layer import PipelineLayer
from ..models.header_metadata import HeaderMetadata
from ..util.gps import bbox_gps_center_and_radius_in_meters


class GPSTranslationLayer(PipelineLayer):
    def run(self, metadata: HeaderMetadata, bboxes: np.ndarray, **kwargs):
        """
        Returns a list of tuples in the form (latitude, longitude, radius, confidence)
        """

        return [
            (bbox_gps_center_and_radius_in_meters(metadata, bbox[:4], **kwargs) + (bbox[4],))
            for bbox in bboxes
        ]
