import logging
import traceback
from typing import List

import exiftool
import zmq

from server.layers.detector import DetectionLayer
from server.layers.gps_translator import GPSTranslationLayer
from server.layers.header_reader import HeaderReader
from server.layers.parallel import ParallelLayer
from server.models.bbox import BBox
from server.models.error import DetectionError, HeaderError
from server.models.header_metadata import HeaderMissingError
from server.util.logging import setup_logger


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

    model_path = "yolo/yolov3-416.onnx"
    # model_path = "yolo/yolov3-416.trt"

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
                bboxes = GetBoundingBoxes(message.decode())
                from pprint import pprint
                pprint(bboxes)
                socket.send_pyobj(bboxes)
            except HeaderMissingError as e:
                logging.info(f"Header missing from image '{message.decode()}'")
                socket.send_pyobj(HeaderError(f"Header missing from image '{message.decode()}': {str(e)}"))
            except UnicodeDecodeError:
                socket.send_pyobj(DetectionError("User cannot decode path message. Is it using UTF-8?"))
            except FileNotFoundError:
                socket.send_pyobj(DetectionError(f"File '{message.decode()}' not found"))
            except exiftool.exceptions.ExifToolException:
                socket.send_pyobj(HeaderError(f"File '{message.decode()}' does not have valid EXIF data"))
            except Exception as e:
                logging.error(traceback.format_exc())
                socket.send_pyobj(DetectionError(f"Unknown error: {str(e)}"))
    except KeyboardInterrupt:
        pass
