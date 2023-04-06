import json
import os
import sys
import time
from concurrent import futures

import cv2
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
        bboxes_pixels = self._get_bboxes_pixels(img_path)

        return ([], []) if len(bboxes_pixels) == 0 else bboxes_pixels


class GPSTranslocationLayer:
    latitude = None
    longitude = None
    altitude = None
    heading = None

    sensorw = None
    sensorh = None

    focal_length = None

    imagew = None
    imageh = None

    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    v_fov = None
    h_fov = None

    v_dist = None
    h_dist = None

    aspect_ratio = None

    def _get_data_from_json(self, json_data):
        data = json.loads(json_data)

        self.altitude = data["altitude"]
        self.latitude = data["latitude"]
        self.longitude = data["longitude"]
        self.heading = data["heading"]
        self.sensorh = data["sensorh"]
        self.sensorw = data["sensorw"]
        self.focal_length = data["focal_length"]
        self.imagew = data["imagew"]
        self.imageh = data["imageh"]

    def _destination_point(self, start_lat, start_lon, heading, distance):
        start_point = geopy.Point(start_lat, start_lon)
        distance = geopy.distance.distance(meters=distance)
        destination_point = distance.destination(
            point=start_point, bearing=heading
        )  # i have no idea if this uses true north or magnetic north or grid north. geography sucks
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
            s=(self.imagew / 2) / self.h_dist * radius,
        )
        plt.show()

    def _get_corner_coordinates(self):
        self.h_fov = 2 * np.degrees(np.atan(self.sensorw / (2 * self.focal_length)))
        self.v_fov = 2 * np.degrees(np.atan(self.sensorh / (2 * self.focal_length)))

        self.h_dist = self.altitude * np.tan(np.radians(self.h_fov / 2))
        self.v_dist = self.altitude * np.tan(np.radians(self.h_fov / 2))

        self.aspect_ratio = self.imagew / self.imageh

        # Calculate the distances to the top-right and bottom-left corners
        d_top_right = np.sqrt((self.h_dist) ** 2 + (self.v_dist) ** 2)
        d_bottom_left = d_top_right

        # Calculate the distances to the top-left and bottom-right corners
        d_top_left = np.sqrt(
            (self.h_dist) ** 2 + (self.v_dist * self.aspect_ratio) ** 2
        )
        d_bottom_right = d_top_left

        # Calculate the bearings from the center to the corners
        bearing_top_right = (
            self.heading - 180 + np.degrees(np.atan2(self.h_dist, self.v_dist))
        ) % 360
        bearing_top_left = (
            self.heading - 180 + np.degrees(np.atan2(-self.h_dist, self.v_dist))
        ) % 360
        bearing_bottom_right = (
            self.heading - 180 + np.degrees(np.atan2(self.h_dist, -self.v_dist))
        ) % 360
        bearing_bottom_left = (
            self.heading - 180 + np.degrees(np.atan2(-self.h_dist, -self.v_dist))
        ) % 360

        # Calculate the GPS coordinates of the corners
        self.top_right = self._destination_point(
            self.latitude, self.longitude, bearing_top_right, d_top_right
        )
        self.top_left = self._destination_point(
            self.latitude, self.longitude, bearing_top_left, d_top_left
        )
        self.bottom_right = self._destination_point(
            self.latitude, self.longitude, bearing_bottom_right, d_bottom_right
        )
        self.bottom_left = self._destination_point(
            self.latitude, self.longitude, bearing_bottom_left, d_bottom_left
        )

    def _pixel_to_gps(self, pixel):
        x, y = pixel
        mid_x = self.imagew / 2
        mid_y = self.imageh / 2

        pixel_heading = self.heading + np.degrees(np.atan2(x - mid_x, y - mid_y))

        loc_x = (x - mid_x) / mid_x
        loc_y = (y - mid_y) / mid_y

        dist_loc_x = loc_x * self.h_dist
        dist_loc_y = loc_y * self.v_dist

        distance = np.sqrt(dist_loc_x**2 + dist_loc_y**2)

        return self._destination_point(
            self.latitude, self.longitude, pixel_heading, distance
        )

    def _bbox_pixels_to_gps(self, bbox_pixels):
        x_min, y_min, x_max, y_max = bbox_pixels

        coord_1 = self._pixel_to_gps((x_min, y_min))
        coord_2 = self._pixel_to_gps((x_max, y_max))

        return coord_1, coord_2

    def _get_center_of_bbox(self, coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        return (lat1 + lat2) / 2, (lon1 + lon2) / 2

    def _get_radius_of_bbox_in_meters(self, coord1, coord2):
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        return geopy.distance.distance((lat1, lon1), (lat2, lon2)).m / 2

    def _bbox_gps_center_and_radius_in_meters(self, bbox_pixels):
        coord_1, coord_2 = self._bbox_pixels_to_gps(bbox_pixels)
        center = self._get_center_of_bbox(coord_1, coord_2)
        radius = self._get_radius_of_bbox_in_meters(coord_1, coord_2)
        return center, radius

    def run(self, image_path, json_data, bboxes):
        self._get_data_from_json(json_data)
        self._get_corner_coordinates()
        out = None
        for bbox in bboxes:
            out = self._bbox_gps_center_and_radius_in_meters(bbox)
        self._plot_corners_on_map_with_detection(*out)


class MessagingService(messaging_pb2_grpc.MessagingServiceServicer):
    def __init__(self):
        self.buffer = []

    def GetBoundingBoxes(self, request, context):
        bboxes_pixels = obj_layer.run(request.path)
        print(bboxes_pixels)

        if bboxes_pixels == ([], []):
            return messaging_pb2.BBoxes(jsondata="No objects detected")

        gps_translation_layer.run(request.path, request.jsondata, bboxes_pixels)

        return messaging_pb2.BBoxes(jsondata="Data received and added to buffer")


def serve():
    global obj_layer, gps_translation_layer

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    weights_file = "weights/yolov3-aerial.weights"
    config_file = "weights/yolov3-aerial.cfg"

    obj_layer = ObjectDetectionLayer(config_file=config_file, weights_file=weights_file)

    gps_translation_layer = GPSTranslocationLayer()

    messaging_pb2_grpc.add_MessagingServiceServicer_to_server(
        MessagingService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
