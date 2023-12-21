import logging
from typing import Sequence, Tuple

import numpy as np
import PIL.Image

from .layer import PipelineLayer
from ..engines.engine import EngineLoader
from ..util.logging import log_time

import server.util.yolo_util as YoloUtil


class DetectionLayer(PipelineLayer):
    def __init__(self, model_path: str, min_confidence: float = 0.3, *args, **kwargs):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.net = self.load_model(*args, **kwargs)


    @log_time
    def load_model(self, *args, **kwargs):
        return EngineLoader.load(self.model_path, *args, **kwargs)


    @log_time
    def _get_bboxes_pixels(self, img_path: str) -> Sequence[Tuple[int, int, int, int, int]]:
        """
        Returns a numpy array of bounding boxes in the format of x0, x1, y0, y1, confidence
        where x0, x1 are columns and y0, y1 are rows

        img_path: path to image as a string
        """

        input_shape_hw = self.net.get_input_shape()

        # load and preprocess image
        img_pil = PIL.Image.open(img_path)
        img = YoloUtil.preprocess_image(img_pil, input_shape_hw)

        predictions = self.net(img)
        boxes, _, scores = YoloUtil.postprocess_net_output(
            predictions,
            input_shape_hw,
            img_pil.size[::-1],
            confidence=self.min_confidence,
        )

        # append confidence to each data point
        boxes = np.concatenate((boxes, 100 * scores.reshape(-1, 1)), axis=1).astype(int)

        # follows the format of x0, x1, y0, y1, confidence
        # where x0, x1 are columns and y0, y1 are rows
        return boxes


    def run(self, img_path: str):
        logging.info(f"Running object detection on image: {img_path}")
        output = self._get_bboxes_pixels(img_path)
        logging.info(f"Number of detections found: {len(output)}")
        return output