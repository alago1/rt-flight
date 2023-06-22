import numpy as np

from layers.layer import PipelineLayer
from models.header_metadata import HeaderMetadata
from util.gps import bbox_gps_center_and_radius_in_meters


class GPSTranslationLayer(PipelineLayer):
    def run(self, metadata: HeaderMetadata, bboxes: np.ndarray):
        return [
            (bbox_gps_center_and_radius_in_meters(metadata, bbox[:4]) + (bbox[4],))
            for bbox in bboxes
        ]