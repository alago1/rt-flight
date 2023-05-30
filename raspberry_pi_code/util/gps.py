from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopy
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import smopy

if TYPE_CHECKING:
    from models.header_metadata import HeaderMetadata

def destination_point(start_lat, start_lon, bearing, distance):
        start_point = geopy.Point(start_lat, start_lon)
        distance = geopy.distance.distance(meters=distance)
        destination_point = distance.destination(point=start_point, bearing=bearing)
        return destination_point.latitude, destination_point.longitude


def pixel_to_gps(metadata: HeaderMetadata, pixel):
    x, y = pixel

    center_relative_position_pixel = (
        x - metadata.image_width / 2,
        y - metadata.image_height / 2,
    )

    pixel_heading = (
        metadata.heading
        + np.degrees(np.arctan2(*center_relative_position_pixel[::-1]))
        + 90  # this may be wrong :shrug:
    )

    displacement_x_meters = (
        (center_relative_position_pixel[0] / metadata.image_width)
        * 2
        * metadata.half_image_height_meters
    )
    displacement_y_meters = (
        (center_relative_position_pixel[1] / metadata.image_height)
        * 2
        * metadata.half_image_width_meters
    )

    distance_meters = np.sqrt(
        displacement_x_meters**2 + displacement_y_meters**2
    )

    return destination_point(
        metadata.latitude, metadata.longitude, pixel_heading, distance_meters
    )


def bbox_pixels_to_center_gps(metadata: HeaderMetadata, bbox_pixels):
    y_min, y_max, x_min, x_max = bbox_pixels  # x: cols, y: rows

    bbox_center = (y_min + y_max) / 2, (x_min + x_max) / 2
    return pixel_to_gps(metadata, bbox_center)


def get_radius_of_bbox_in_meters(metadata: HeaderMetadata, bbox_pixels):
    y_min, y_max, x_min, x_max = bbox_pixels  # x: cols, y: rows
    axis_length_pixels = (y_max - y_min) / 2, (x_max - x_min) / 2
    axis_length_meters = (
        axis_length_pixels[0]
        / metadata.image_height
        * 2
        * metadata.half_image_height_meters,
        axis_length_pixels[1] / metadata.image_width * 2 * metadata.half_image_width_meters,
    )

    return np.sqrt(axis_length_meters[0] ** 2 + axis_length_meters[1] ** 2)


def bbox_gps_center_and_radius_in_meters(metadata: HeaderMetadata, bbox_pixels):
    center = bbox_pixels_to_center_gps(metadata, bbox_pixels)
    radius = get_radius_of_bbox_in_meters(metadata, bbox_pixels)
    logging.debug(f"Detection for center: {center}, radius: {radius}")
    return center[0], center[1], radius


def get_corner_coordinates(metadata: HeaderMetadata):
    # Calculate the distances to the corners
    distance_to_corner = np.sqrt(
        (metadata.half_image_width_meters) ** 2 + (metadata.half_image_height_meters) ** 2
    )

    # Calculate the bearings from the center to the corners
    bearing_top_right = (
        metadata.heading
        + np.degrees(
            np.arctan2(metadata.half_image_height_meters, metadata.half_image_width_meters)
        )
    ) % 360
    bearing_bottom_right = (
        metadata.heading
        + np.degrees(
            np.arctan2(metadata.half_image_height_meters, -metadata.half_image_width_meters)
        )
    ) % 360
    bearing_top_left = (
        metadata.heading
        + np.degrees(
            np.arctan2(-metadata.half_image_height_meters, metadata.half_image_width_meters)
        )
    ) % 360
    bearing_bottom_left = (
        metadata.heading
        + np.degrees(
            np.arctan2(
                -metadata.half_image_height_meters, -metadata.half_image_width_meters
            )
        )
    ) % 360

    # Calculate the GPS coordinates of the corners
    top_right = destination_point(
        metadata.latitude, metadata.longitude, bearing_top_right, distance_to_corner
    )
    top_left = destination_point(
        metadata.latitude, metadata.longitude, bearing_top_left, distance_to_corner
    )
    bottom_right = destination_point(
        metadata.latitude, metadata.longitude, bearing_bottom_right, distance_to_corner
    )
    bottom_left = destination_point(
        metadata.latitude, metadata.longitude, bearing_bottom_left, distance_to_corner
    )

    return top_right, top_left, bottom_right, bottom_left


def plot_corners_on_map(metadata: HeaderMetadata, zoom=14):
    # Create a Smopy map using the bounding box
    map = smopy.Map((metadata.latitude, metadata.longitude), z=zoom)

    x_top_left, y_top_left = map.to_pixels(*metadata.top_left)
    x_top_right, y_top_right = map.to_pixels(*metadata.top_right)
    x_bottom_left, y_bottom_left = map.to_pixels(*metadata.bottom_left)
    x_bottom_right, y_bottom_right = map.to_pixels(*metadata.bottom_right)

    plt.figure(figsize=(10, 10))
    plt.imshow(map.img)
    plt.scatter(
        [x_top_left, x_top_right, x_bottom_right, x_bottom_left, x_top_left],
        [y_top_left, y_top_right, y_bottom_right, y_bottom_left, y_top_left],
    )
    plt.show()


def plot_corners_on_map_with_detection(metadata: HeaderMetadata, det, radius, zoom=14):
    # Create a Smopy map using the bounding box
    map = smopy.Map((metadata.latitude, metadata.longitude), z=zoom)

    x_top_left, y_top_left = map.to_pixels(*metadata.top_left)
    x_top_right, y_top_right = map.to_pixels(*metadata.top_right)
    x_bottom_left, y_bottom_left = map.to_pixels(*metadata.bottom_left)
    x_bottom_right, y_bottom_right = map.to_pixels(*metadata.bottom_right)

    det_lat, det_lon = det
    det_lat, det_lon = map.to_pixels(det_lat, det_lon)

    plt.figure(figsize=(10, 10))
    plt.imshow(map.img)
    plt.scatter(
        [x_top_left, x_top_right, x_bottom_right, x_bottom_left],
        [y_top_left, y_top_right, y_bottom_right, y_bottom_left],
    )
    plt.scatter(
        [det_lat, det_lat],
        [det_lon, det_lon],
        s=(metadata.image_width / 2) / metadata.half_image_width_meters * radius,
    )
    plt.show()
