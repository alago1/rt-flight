# pylint: disable=syntax-error
from exif import Image
from math import radians, degrees, sin, cos, tan, atan2, sqrt, atan, asin
import json
import smopy
import matplotlib.pyplot as plt
import numpy as np
from geopy import distance

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
            
    def _destination_point(self, lat1, lon1, heading, distance):
        # Convert latitude and longitude from degrees to radians
        lat1 = radians(lat1)
        lon1 = radians(lon1)

        # Convert the bearing to radians
        bearing = radians(heading)

        # Earth's mean radius in meters (6,371,000 meters)
        earth_radius = 6371000

        # Calculate the angular distance (in radians) covered on Earth's surface
        angular_distance = distance / earth_radius

        # Calculate the destination point's latitude
        lat2 = asin(sin(lat1) * cos(angular_distance) +
                        cos(lat1) * sin(angular_distance) * cos(bearing))

        # Calculate the destination point's longitude
        lon2 = lon1 + atan2(sin(bearing) * sin(angular_distance) * cos(lat1),
                                cos(angular_distance) - sin(lat1) * sin(lat2))

        # Convert the latitude and longitude from radians back to degrees
        lat2 = degrees(lat2)
        lon2 = degrees(lon2)

        return lon2, lat2

    def _plot_corners_on_map(self, zoom=14):

        # Create a Smopy map using the bounding box
        _map = smopy.Map((self.latitude, self.longitude), z=14)

        x_top_left, y_top_left = _map.to_pixels(self.top_left[1], self.top_left[0])
        x_top_right, y_top_right = _map.to_pixels(self.top_right[1], self.top_right[0])
        x_bottom_left, y_bottom_left = _map.to_pixels(self.bottom_left[1], self.bottom_left[0])
        x_bottom_right, y_bottom_right = _map.to_pixels(self.bottom_right[1], self.bottom_right[0])

        plt.figure(figsize=(10, 10))
        plt.imshow(_map.img)
        plt.scatter([x_top_left, x_top_right, x_bottom_right, x_bottom_left, x_top_left],
                    [y_top_left, y_top_right, y_bottom_right, y_bottom_left, y_top_left])
        plt.show()

    def _plot_corners_on_map_with_detection(self, det, radius, zoom=14):
        
        # Create a Smopy map using the bounding box
        _map = smopy.Map((self.latitude, self.longitude), z=14)

        x_top_left, y_top_left = _map.to_pixels(self.top_left[1], self.top_left[0])
        x_top_right, y_top_right = _map.to_pixels(self.top_right[1], self.top_right[0])
        x_bottom_left, y_bottom_left = _map.to_pixels(self.bottom_left[1], self.bottom_left[0])
        x_bottom_right, y_bottom_right = _map.to_pixels(self.bottom_right[1], self.bottom_right[0])
        
        det_lat, det_lon = det
        det_lon, det_lat = _map.to_pixels(det_lon, det_lat)

        plt.figure(figsize=(10, 10))
        plt.imshow(_map.img)
        plt.scatter([x_top_left, x_top_right, x_bottom_right, x_bottom_left, x_top_left],
                    [y_top_left, y_top_right, y_bottom_right, y_bottom_left, y_top_left])
        plt.scatter([det_lon, det_lon], [det_lat, det_lat], s=(self.imagew/2)/self.h_dist * radius)
        plt.show()

    def _get_corner_coordinates(self):
        self.h_fov = 2 * degrees(atan(self.sensorw / (2 * self.focal_length)))
        self.v_fov = 2 * degrees(atan(self.sensorh / (2 * self.focal_length)))

        self.h_dist = self.altitude * tan(radians(self.h_fov/2))
        self.v_dist = self.altitude * tan(radians(self.h_fov/2))
        
        self.aspect_ratio = self.imagew / self.imageh

        # Calculate the distances to the top-right and bottom-left corners
        d_top_right = sqrt((self.h_dist) ** 2 + (self.v_dist) ** 2)
        d_bottom_left = d_top_right

        # Calculate the distances to the top-left and bottom-right corners
        d_top_left = sqrt((self.h_dist) ** 2 + (self.v_dist * self.aspect_ratio) ** 2)
        d_bottom_right = d_top_left

        # Calculate the bearings from the center to the corners
        bearing_top_right = (self.heading - 180 + degrees(atan2(self.h_dist, self.v_dist))) % 360
        bearing_top_left = (self.heading - 180 + degrees(atan2(-self.h_dist, self.v_dist))) % 360
        bearing_bottom_right = (self.heading - 180 + degrees(atan2(self.h_dist, -self.v_dist))) % 360
        bearing_bottom_left = (self.heading - 180 + degrees(atan2(-self.h_dist, -self.v_dist))) % 360

        # Calculate the GPS coordinates of the corners
        self.top_right = self._destination_point(self.latitude, self.longitude, bearing_top_right, d_top_right)
        self.top_left = self._destination_point(self.latitude, self.longitude, bearing_top_left, d_top_left)
        self.bottom_right = self._destination_point(self.latitude, self.longitude, bearing_bottom_right, d_bottom_right)
        self.bottom_left = self._destination_point(self.latitude, self.longitude, bearing_bottom_left, d_bottom_left)

    def _pixel_to_gps(self, pixel):
        x, y = pixel
        mid_x = self.imagew/2
        mid_y = self.imageh/2

        pixel_heading = self.heading + degrees(atan2(x - mid_x, y - mid_y))
        
        #calculate the distance between the pixel and the center of the image using self.h_dist and self.v_dist
        loc_x = (x - mid_x) / mid_x
        loc_y = (y - mid_y) / mid_y
        
        dist_loc_x = loc_x * self.h_dist
        dist_loc_y = loc_y * self.v_dist
        
        distance = sqrt(dist_loc_x ** 2 + dist_loc_y ** 2)

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
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        return distance.distance(coord1, coord2).m * 2

    def _bbox_gps_center_and_radius_in_meters(self, bbox_pixels):
        coord_1, coord_2 = self._bbox_pixels_to_gps(bbox_pixels)
        center = self._get_center_of_bbox(coord_1, coord_2)
        radius = self._get_radius_of_bbox_in_meters(coord_1, coord_2)
        return center, radius


    def run(self, image_path, json_data, bboxes):
        with open(img_path, 'rb') as src:
            img = Image(src)        
            
        self._get_data_from_json(json_data)
        self._get_corner_coordinates()
        out = None
        for bbox in bboxes:
            out = self._bbox_gps_center_and_radius_in_meters(bbox)
        self._plot_corners_on_map_with_detection(*out)
        

img_path =  "../realtime_ui/data/IMG_0064.jpg"

data = {
    "longitude": -82.89811490277778,
    "latitude": 30.30491903277778,
    "altitude": 96.91,
    "heading": 318.5,
    "sensorw": 6.17,
    "sensorh": 4.55,
    "focal_length": 7.2,
    "imagew": 5184,
    "imageh": 3888
}

bbox_pixels = [(4000, 3000, 4500, 3500)]
json_data = json.dumps(data)

obj = GPSTranslocationLayer()
obj.run(img_path, json_data, bbox_pixels)