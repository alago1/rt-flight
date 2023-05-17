import time
from concurrent import futures
from contextlib import redirect_stdout
import traceback

import tflite_runtime.interpreter as tflite
import exiftool
import geopy
import geopy.distance

import matplotlib.pyplot as plt
import numpy as np
import PIL
import smopy

# from dataclasses import dataclass
import zmq
from bbox import BBox

import yolo_util as YoloUtil


def log(message):
    global log_path
    with open(log_path, "a") as f:
        with redirect_stdout(f):
            print(message)


class ObjectDetectionLayer:
    def __init__(self, model_path=None, min_confidence=0.3):
        self.model_path = model_path

        self.net = self._load_model()
        self.min_confidence = min_confidence

        self.classes = ["car", "truck", "bus", "minibus", "cyclist"]

    def _load_model(self):
        model = tflite.Interpreter(self.model_path, num_threads=4)
        return model

    def _get_bboxes_pixels(self, img_path):
        self.net.allocate_tensors()
        _, height, width, _ = self.net.get_input_details()[0]["shape"]

        # load and preprocess image
        img_pil = PIL.Image.open(img_path)
        img = YoloUtil.preprocess_image(img_pil, (height, width))

        tensor_index = self.net.get_input_details()[0]["index"]
        self.net.set_tensor(tensor_index, img)

        self.net.invoke()
        output_details = self.net.get_output_details()
        predictions = [
            self.net.get_tensor(output_details[i]["index"])
            for i in range(len(output_details))
        ]

        predictions.sort(key=lambda x: x.shape[1])

        boxes = []
        for i in range(len(predictions)):
            # boxes.append(self._decode_yolo_boxes(predictions[i][0], anchors[i], (height, width), img_pil.size))
            boxes.append(
                YoloUtil.decode_net_output(
                    predictions[i],
                    YoloUtil.v3_anchors[YoloUtil.v3_anchor_mask[i]],
                    20,
                    (height, width),
                )
            )

        boxes = np.concatenate(boxes, axis=1)
        boxes = YoloUtil.correct_boxes(boxes, img_pil.size[::-1], (height, width))
        boxes, _, scores = YoloUtil.handle_predictions(
            boxes, 20, confidence=self.min_confidence
        )

        # changes format from (x, y, w, h) to (x0, y0, x1, y1)
        boxes = YoloUtil.adjust_boxes(boxes, img_pil.size[::-1])

        # change to (x0, x1, y0, y1)
        boxes = boxes[:, [0, 2, 1, 3]]

        # append confidence to each data point
        boxes = np.concatenate((boxes, 100 * scores.reshape(-1, 1)), axis=1).astype(int)

        # follows the format of x0, x1, y0, y1, confidence
        # where x0, x1 are columns and y0, y1 are rows
        return boxes

    def run(self, img_path):
        log("*" * 75)
        log(f"Running object detection on image: {img_path}")
        output = self._get_bboxes_pixels(img_path)
        log(f"Number of detections found: {len(output)}")
        log("*" * 75)
        return output


class GPSTranslocationLayer:
    latitude = None
    longitude = None
    altitude = None
    heading = None

    half_image_width_meters = None
    half_image_height_meters = None

    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    image_width = None
    image_height = None

    def _load_metadata(self, image_path):
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(image_path)[0]

        log(f"Metadata: {metadata}")

        self.image_width = (
            dict.get(metadata, "EXIF:ExifImageWidth", None)
            or dict.get(metadata, "EXIF:ImageWidth", None)
            or dict.get(metadata, "File:ImageWidth")
        )
        self.image_height = (
            dict.get(metadata, "EXIF:ExifImageHeight", None)
            or dict.get(metadata, "EXIF:ImageHeight", None)
            or dict.get(metadata, "File:ImageHeight")
        )

        self.latitude = metadata["EXIF:GPSLatitude"]
        self.longitude = metadata["EXIF:GPSLongitude"]
        self.altitude = metadata["EXIF:GPSAltitude"]
        self.heading = dict.get(metadata, "EXIF:GPSImgDirection", 0)

        if self.heading == 0:
            print(
                "WARNING: Heading defaulted to 0. The program will continute to run, but this may cause issues."
            )

        if metadata["EXIF:GPSLatitudeRef"] == "S":
            assert self.latitude >= 0, "Latitude is negative but ref is S"
            self.latitude *= -1

        if metadata["EXIF:GPSLongitudeRef"] == "W":
            assert self.longitude >= 0, "Longitude is negative but ref is W"
            self.longitude *= -1

        if metadata["EXIF:GPSImgDirectionRef"] == "M":
            assert (
                np.abs(self.heading) > 2 * np.pi
            ), "Heading is in radians but we assume degrees. Please fix"
            self.heading -= 8.0  # subtract 8deg to account for magnetic declination

        units_to_meter_conversion_factors = [
            None,  # this is the default value
            0.0254,  # inches
            1e-2,  # cm
            1e-3,  # mm
            1e-6,  # um
        ]
        unit_index = dict.get(metadata, "EXIF:FocalPlaneResolutionUnit", 1) - 1
        resolution_conversion_factor = units_to_meter_conversion_factors[unit_index]

        assert (
            resolution_conversion_factor is not None
        ), "FocalPlaneResolutionUnit is None"

        focal_length = metadata["EXIF:FocalLength"] * resolution_conversion_factor
        sensor_width = (
            metadata["EXIF:FocalPlaneXResolution"] * resolution_conversion_factor
        )
        sensor_height = (
            metadata["EXIF:FocalPlaneYResolution"] * resolution_conversion_factor
        )

        self.half_image_width_meters = self.altitude * sensor_width / focal_length
        self.half_image_height_meters = self.altitude * sensor_height / focal_length

        log("*" * 75)
        log(f"Loaded metadata from image: {image_path}")
        log(f"Image dimensions:{self.image_width}, {self.image_height}")
        log(f"GPS coordinates: {self.latitude}, {self.longitude}")
        log(f"Altitude: {self.altitude}")
        log(f"Heading: {self.heading}")
        log(f"Half image width in meters: {self.half_image_width_meters}")
        log(f"Half image height in meters: {self.half_image_height_meters}")
        log("*" * 75)

    def _destination_point(self, start_lat, start_lon, bearing, distance):
        start_point = geopy.Point(start_lat, start_lon)
        distance = geopy.distance.distance(meters=distance)
        destination_point = distance.destination(point=start_point, bearing=bearing)
        return destination_point.latitude, destination_point.longitude

    def _plot_corners_on_map(self, zoom=14):
        # Create a Smopy map using the bounding box
        _map = smopy.Map((self.latitude, self.longitude), z=14)

        x_top_left, y_top_left = _map.to_pixels(*self.top_left)
        x_top_right, y_top_right = _map.to_pixels(*self.top_right)
        x_bottom_left, y_bottom_left = _map.to_pixels(*self.bottom_left)
        x_bottom_right, y_bottom_right = _map.to_pixels(*self.bottom_right)

        plt.figure(figsize=(10, 10))
        plt.imshow(_map.img)
        plt.scatter(
            [x_top_left, x_top_right, x_bottom_right, x_bottom_left, x_top_left],
            [y_top_left, y_top_right, y_bottom_right, y_bottom_left, y_top_left],
        )
        plt.show()

    def _plot_corners_on_map_with_detection(self, det, radius, zoom=14):
        # Create a Smopy map using the bounding box
        _map = smopy.Map((self.latitude, self.longitude), z=14)

        x_top_left, y_top_left = _map.to_pixels(*self.top_left)
        x_top_right, y_top_right = _map.to_pixels(*self.top_right)
        x_bottom_left, y_bottom_left = _map.to_pixels(*self.bottom_left)
        x_bottom_right, y_bottom_right = _map.to_pixels(*self.bottom_right)

        det_lat, det_lon = det
        det_lat, det_lon = _map.to_pixels(det_lat, det_lon)

        plt.figure(figsize=(10, 10))
        plt.imshow(_map.img)
        plt.scatter(
            [x_top_left, x_top_right, x_bottom_right, x_bottom_left],
            [y_top_left, y_top_right, y_bottom_right, y_bottom_left],
        )
        plt.scatter(
            [det_lat, det_lat],
            [det_lon, det_lon],
            s=(self.image_width / 2) / self.half_image_width_meters * radius,
        )
        plt.show()

    def _get_corner_coordinates(self):
        # Calculate the distances to the corners
        distance_to_corner = np.sqrt(
            (self.half_image_width_meters) ** 2 + (self.half_image_height_meters) ** 2
        )

        # Calculate the bearings from the center to the corners
        bearing_top_right = (
            self.heading
            + np.degrees(
                np.arctan2(self.half_image_height_meters, self.half_image_width_meters)
            )
        ) % 360
        bearing_bottom_right = (
            self.heading
            + np.degrees(
                np.arctan2(self.half_image_height_meters, -self.half_image_width_meters)
            )
        ) % 360
        bearing_top_left = (
            self.heading
            + np.degrees(
                np.arctan2(-self.half_image_height_meters, self.half_image_width_meters)
            )
        ) % 360
        bearing_bottom_left = (
            self.heading
            + np.degrees(
                np.arctan2(
                    -self.half_image_height_meters, -self.half_image_width_meters
                )
            )
        ) % 360

        # Calculate the GPS coordinates of the corners
        self.top_right = self._destination_point(
            self.latitude, self.longitude, bearing_top_right, distance_to_corner
        )
        self.top_left = self._destination_point(
            self.latitude, self.longitude, bearing_top_left, distance_to_corner
        )
        self.bottom_right = self._destination_point(
            self.latitude, self.longitude, bearing_bottom_right, distance_to_corner
        )
        self.bottom_left = self._destination_point(
            self.latitude, self.longitude, bearing_bottom_left, distance_to_corner
        )

    def _pixel_to_gps(self, pixel):
        x, y = pixel

        center_relative_position_pixel = (
            x - self.image_width / 2,
            y - self.image_height / 2,
        )

        pixel_heading = (
            self.heading
            + np.degrees(np.arctan2(*center_relative_position_pixel[::-1]))
            + 90  # this may be wrong :shrug:
        )

        displacement_x_meters = (
            (center_relative_position_pixel[0] / self.image_width)
            * 2
            * self.half_image_height_meters
        )
        displacement_y_meters = (
            (center_relative_position_pixel[1] / self.image_height)
            * 2
            * self.half_image_width_meters
        )

        distance_meters = np.sqrt(
            displacement_x_meters**2 + displacement_y_meters**2
        )

        return self._destination_point(
            self.latitude, self.longitude, pixel_heading, distance_meters
        )

    def _bbox_pixels_to_center_gps(self, bbox_pixels):
        y_min, y_max, x_min, x_max = bbox_pixels  # x: cols, y: rows

        bbox_center = (y_min + y_max) / 2, (x_min + x_max) / 2
        return self._pixel_to_gps(bbox_center)

    def _get_radius_of_bbox_in_meters(self, bbox_pixels):
        y_min, y_max, x_min, x_max = bbox_pixels  # x: cols, y: rows
        axis_length_pixels = (y_max - y_min) / 2, (x_max - x_min) / 2
        axis_length_meters = (
            axis_length_pixels[0]
            / self.image_height
            * 2
            * self.half_image_height_meters,
            axis_length_pixels[1] / self.image_width * 2 * self.half_image_width_meters,
        )

        return np.sqrt(axis_length_meters[0] ** 2 + axis_length_meters[1] ** 2)

    def _bbox_gps_center_and_radius_in_meters(self, bbox_pixels):
        center = self._bbox_pixels_to_center_gps(bbox_pixels)
        radius = self._get_radius_of_bbox_in_meters(bbox_pixels)
        log(f"Detection for center: {center}, radius: {radius}")
        log("*" * 75)
        return center[0], center[1], radius

    def run(self, image_path, bboxes):
        self._load_metadata(image_path)
        self._get_corner_coordinates()

        return [
            (self._bbox_gps_center_and_radius_in_meters(bbox[:4]) + (bbox[4],))
            for bbox in bboxes
        ]


def GetBoundingBoxes(path):
    try:
        bboxes_pixels = obj_layer.run(path)

        if len(bboxes_pixels) == 0:
            print("No detections found")
            return []

        output = gps_translation_layer.run(path, bboxes_pixels)

        bbox_list = [
            BBox(
                latitude=bbox[0], longitude=bbox[1], radius=bbox[2], confidence=bbox[3]
            )
            for bbox in output
        ]
        print(f"Returning {len(bbox_list)} bboxes")

    except Exception as e:
        print(traceback.format_exc())
        log(traceback.format_exc())
        raise e

    return bbox_list

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    global log_path
    log_path = "logs.txt"

    global obj_layer, gsp_translation_layer

    model_path = "../yolo/yolov3-aerial-latest.tflite"

    obj_layer = ObjectDetectionLayer(model_path=model_path)
    log(f"Using detection model loaded from {model_path}")

    gps_translation_layer = GPSTranslocationLayer()

    try:
        while True:
            message = socket.recv()  # Should obtain message which is path of new image
            
            # TODO: Check if path is a valid image
            try:
                bboxes = GetBoundingBoxes(message.decode("utf-8"))

                socket.send_pyobj(bboxes)
            except:
                pass

    except KeyboardInterrupt:
        pass
