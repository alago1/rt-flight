import logging
import traceback
from typing import List
from pprint import pprint
import ntpath

import exiftool
import zmq

from server.layers.detector import DetectionLayer
from server.layers.gps_translator import GPSTranslationLayer
from server.layers.header_reader import HeaderReader
from server.layers.parallel import ParallelLayer
from server.models.bbox import BBox, BBoxes
from server.models.error import DetectionError, HeaderError
from server.models.header_metadata import HeaderMissingError
from server.util.logging import setup_logger


def GetBoundingBoxes(path: str) -> List[BBox]:
    try:
        header_data, bboxes_pixels = parallel_layer.run((path,), share_input=True)
        filename = ntpath.split(path)[1] or ntpath.basename(ntpath.split(path)[0])

        if len(bboxes_pixels) == 0:
            logging.info("No detections found")
            return BBoxes([], filename)

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

    return BBoxes(bbox_list, filename)


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:5555")

    setup_logger()

    model_path = "yolo/yolov3-aerial.onnx"

    kwargs = {'providers': [('CPUExecutionProvider')]} \
        if model_path.endswith('.onnx') else {}

    parallel_layer = ParallelLayer([
        HeaderReader(),
        DetectionLayer(model_path, engine="auto", **kwargs),
    ], use_threads=True)
    gps_translation_layer = GPSTranslationLayer()

    try:
        while True:
            message = socket.recv()  # Should obtain message which is path of new image
            print('Received a message')

            try:
                bboxes = GetBoundingBoxes(message.decode())
                pprint(bboxes)
                socket.send_pyobj(bboxes)
            except HeaderMissingError as e:
                logging.info(f"Header missing from image '{message.decode()}'")
                socket.send_pyobj(HeaderError(f"Header missing from image '{message.decode()}': {str(e)}"))
            except UnicodeDecodeError:
                logging.error("Cannot decode path message.")
                socket.send_pyobj(DetectionError("User cannot decode path message. Is it using UTF-8?"))
            except FileNotFoundError:
                logging.error(f"File '{message.decode()}' not found")
                socket.send_pyobj(DetectionError(f"File '{message.decode()}' not found"))
            except exiftool.exceptions.ExifToolException:
                logging.error(f"Invalid exif data found in file {message.decode()}")
                socket.send_pyobj(HeaderError(f"File '{message.decode()}' does not have valid EXIF data"))
            except Exception as e:
                logging.error(traceback.format_exc())
                socket.send_pyobj(DetectionError(f"Unknown error: {str(e)}"))
    except KeyboardInterrupt:
        pass
