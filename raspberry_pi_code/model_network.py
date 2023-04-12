import os
import sys
import time
from concurrent import futures
from contextlib import redirect_stdout

import cv2
import exiftool
import geopy
import geopy.distance
import grpc
import matplotlib.pyplot as plt
import numpy as np
import PIL
import smopy

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))

import messaging_pb2
import messaging_pb2_grpc


def log(message):
    global log_path
    with open(log_path, "a") as f:
        with redirect_stdout(f):
            print(message)


class ObjectDetectionLayer:
    def __init__(self, weights_file=None, config_file=None, min_confidence=0.3):
        self.weights_file = weights_file
        self.config_file = config_file

        self.net = self._load_model()
        self.min_confidence = min_confidence

        self.classes = ["car", "truck", "bus", "minibus", "cyclist"]

    def _load_model(self):
        return cv2.dnn.readNet(self.weights_file, self.config_file)

    def _get_output_layers(self, net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except Exception:
            output_layers = [
                layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()
            ]

        return output_layers

    def _get_bboxes_pixels(self, img_path):
        img = PIL.Image.open(img_path)
        img = np.array(img)

        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        Height, Width = image.shape[:2]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False
        )

        self.net.setInput(blob)

        outs = self.net.forward(self._get_output_layers(self.net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.min_confidence:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        bboxes_with_confidence = []
        for i in indices:
            try:
                box = boxes[i]
            except:
                box = boxes[i[0]]

            x, y, w, h = [max(v, 0) for v in box[:4]]  # model outputs can be negative

            bboxes_with_confidence.append(
                np.array((x, x + w, y, y + h, 100 * confidences[i]))
            )

        # follows the format of x0, x1, y0, y1, confidence
        return np.array(bboxes_with_confidence).astype(int)

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
            [x_top_left, x_top_right, x_bottom_right, x_bottom_left, x_top_left],
            [y_top_left, y_top_right, y_bottom_right, y_bottom_left, y_top_left],
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
            - 180
            + np.degrees(
                np.arctan2(self.half_image_height_meters, self.half_image_width_meters)
            )
        ) % 360
        bearing_top_left = (
            self.heading
            - 180
            + np.degrees(
                np.arctan2(self.half_image_height_meters, -self.half_image_width_meters)
            )
        ) % 360
        bearing_bottom_right = (
            self.heading
            - 180
            + np.degrees(
                np.arctan2(-self.half_image_height_meters, self.half_image_width_meters)
            )
        ) % 360
        bearing_bottom_left = (
            self.heading
            - 180
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

        center_relative_position_pixel = (x - self.image_width, y - self.image_height)

        pixel_heading = (
            self.heading
            + np.degrees(np.arctan2(*center_relative_position_pixel))
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
        x_min, x_max, y_min, y_max = bbox_pixels  # x: rows, y: cols

        bbox_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        return self._pixel_to_gps(bbox_center)

    def _get_radius_of_bbox_in_meters(self, bbox_pixels):
        x_min, x_max, y_min, y_max = bbox_pixels  # x: rows, y: cols
        axis_length_pixels = (x_max - x_min) / 2, (y_max - y_min) / 2
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


class MessagingService(messaging_pb2_grpc.MessagingServiceServicer):
    def __init__(self):
        self.buffer = []

    def GetBoundingBoxes(self, request, context):
        bboxes_pixels = obj_layer.run(request.path)

        if len(bboxes_pixels) == 0:
            print("No detections found")
            return messaging_pb2.BBoxes(bboxes=[])

        output = gps_translation_layer.run(request.path, bboxes_pixels)

        bbox_list = [
            messaging_pb2.BBox(
                latitude=bbox[0],
                longitude=bbox[1],
                radius=bbox[2],
                confidence=bbox[3],
            )
            for bbox in output
        ]
        print(f"Returning {len(bbox_list)} bboxes")

        return messaging_pb2.BBoxes(bboxes=bbox_list)


def serve(port_num):
    global obj_layer, gps_translation_layer

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    using_tiny = False

    if using_tiny:
        weights_file = "yolo/yolov3-tiny.weights"
        config_file = "yolo/yolov3-tiny.cfg"
    else:
        weights_file = "yolo/yolov3-aerial.weights"
        config_file = "yolo/yolov3-aerial.cfg"

    log(
        f"Using detection model with weights from {weights_file} and config from {config_file}"
    )

    obj_layer = ObjectDetectionLayer(config_file=config_file, weights_file=weights_file)

    gps_translation_layer = GPSTranslocationLayer()

    messaging_pb2_grpc.add_MessagingServiceServicer_to_server(
        MessagingService(), server
    )
    server.add_insecure_port(f"[::]:{port_num}")
    server.start()
    log(f"Server started on port {port_num}...")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    global log_path
    port_arg = sys.argv[1] if len(sys.argv) > 1 else 50951
    log_path = sys.argv[2] if len(sys.argv) > 2 else "logs.txt"
    serve(port_num=port_arg)
