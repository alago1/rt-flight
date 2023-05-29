import zmq
import exiftool

from layers.detector import DetectionLayer
from layers.gps_translator import GPSTranslationLayer
from layers.header_reader import HeaderReader
from models.bbox import BBox
from models.error import DetectionError, HeaderError

import logging
import traceback
import multiprocessing as mp
from typing import List


logger = logging.getLogger()

def get_header_data(q, header_layer, path):
    q.put(header_layer.run(path))


def GetBoundingBoxes(path: str) -> List[BBox]:
    try:
        q = mp.Queue()
        p = mp.Process(target=get_header_data, args=(q, header_layer, path))
        p.start()

        bboxes_pixels = obj_layer.run(path)

        p.join()
        header_data = q.get()

        if len(bboxes_pixels) == 0:
            print("No detections found")
            return []

        output = gps_translation_layer.run(header_data, bboxes_pixels)

        bbox_list = [
            BBox(
                latitude=bbox[0],
                longitude=bbox[1],
                radius=bbox[2],
                confidence=bbox[3]
            )
            for bbox in output
        ]
        print(f"Returning {len(bbox_list)} bboxes")

    except Exception as e:
        print(traceback.format_exc())
        logger.error(traceback.format_exc())
        raise e

    return bbox_list


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:5555")

    model_path = "../yolo/yolov3-416.onnx"
    # model_path = "../yolo/yolov3-416.trt"

    obj_layer = DetectionLayer(model_path=model_path, engine="onnx", providers=[("CUDAExecutionProvider")])
    header_layer = HeaderReader()
    gps_translation_layer = GPSTranslationLayer()

    try:
        while True:
            message = socket.recv()  # Should obtain message which is path of new image

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
