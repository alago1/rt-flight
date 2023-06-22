import logging
import traceback
from typing import List

import exiftool
import zmq

from layers.detector import DetectionLayer
from layers.gps_translator import GPSTranslationLayer
from layers.header_reader import HeaderReader
from layers.parallel import ParallelLayer
from models.bbox import BBox
from models.error import DetectionError, HeaderError
from util.logging import setup_logger


def GetBoundingBoxes(path: str) -> List[BBox]:
    try:
        header_data, bboxes_pixels = parallel_layer.run((path,), share_input=True)

        if len(bboxes_pixels) == 0:
            logging.info("No detections found")
            return []

        output = gps_translation_layer.run(header_data, bboxes_pixels)

        bbox_list = [
            BBox(
                latitude=float(bbox[0]),
                longitude=float(bbox[1]),
                radius=float(bbox[2]),
                confidence=float(bbox[3])
            )
            for bbox in output
        ]
        logging.info(f"Returning {len(bbox_list)} bboxes")

    except Exception as e:
        logging.error(traceback.format_exc())
        raise e

    return bbox_list


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:5555")

    setup_logger()

    model_path = "../yolo/yolov3-416.onnx"
    # model_path = "../yolo/yolov3-416.trt"

    parallel_layer = ParallelLayer([
        HeaderReader(),
        DetectionLayer(model_path, engine="onnx", providers=[("CUDAExecutionProvider")])
    ])
    gps_translation_layer = GPSTranslationLayer()

    try:
        while True:
            message = socket.recv()  # Should obtain message which is path of new image
            print('Received a message')

            try:
                bboxes = GetBoundingBoxes(message.decode("utf-8"))
                socket.send_pyobj(bboxes)
            except UnicodeDecodeError:
                socket.send_pyobj(DetectionError("User cannot decode path message. Is it using UTF-8?"))
            except FileNotFoundError:
                socket.send_pyobj(DetectionError(f"File '{message.decode('utf-8')}' not found"))
            except exiftool.exceptions.ExifToolException:
                # TODO: there are other cases where exif data can be provided but is not valid. These should be handled
                socket.send_pyobj(HeaderError(f"File '{message.decode('utf-8')}' does not have valid EXIF data"))
    except KeyboardInterrupt:
        pass
