from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopy
import geopy.distance
import numpy as np

if TYPE_CHECKING:
    from models.header_metadata import HeaderMetadata

def destination_point(start_lat, start_lon, bearing, distance):
        start_point = geopy.Point(start_lat, start_lon)
        distance = geopy.distance.distance(meters=distance)
        destination_point = distance.destination(point=start_point, bearing=bearing)
        return destination_point.latitude, destination_point.longitude


def pixel_to_gps(metadata: HeaderMetadata, pixel, backend="gdal"):
    if backend not in ("geopy", "gdal"):
        raise NotImplementedError("Only geopy and gdal backends are supported")
    
    if backend == "gdal":
        from ..util.gps_gdal import pixel_to_wgs84
        corner_gps = (metadata.top_left, metadata.top_right, metadata.bottom_right, metadata.bottom_left)
        return pixel_to_wgs84(metadata.image_path, pixel, corner_gps)

    y, x = pixel  # x: cols, y: rows

    # center relative pixel position
    relative_y, relative_x = (
        metadata.image_height / 2 - y,  # y-axis is flipped
        x - metadata.image_width / 2,
    )

    height_meters = 2 * metadata.half_image_height_meters
    width_meters = 2 * metadata.half_image_width_meters

    height_meters_per_pixel = height_meters / metadata.image_height
    width_meters_per_pixel = width_meters / metadata.image_width

    displacement_y_meters = relative_y * height_meters_per_pixel
    displacement_x_meters = relative_x * width_meters_per_pixel

    distance_meters = np.sqrt(
        displacement_x_meters**2 + displacement_y_meters**2
    )
    
    # angle from center of image to pixel clockwise from y-axis
    in_frame_angle = 90 - np.degrees(np.arctan2(relative_y, relative_x))

    # add 90 degrees to account for the fact that 0 degrees is north
    pixel_heading = (metadata.heading + in_frame_angle) % 360

    return destination_point(
        metadata.latitude, metadata.longitude, pixel_heading, distance_meters
    )


def bbox_pixels_to_center_gps(metadata: HeaderMetadata, bbox_pixels):
    x_min, x_max, y_min, y_max = bbox_pixels  # x: cols, y: rows

    bbox_center = (y_min + y_max) / 2, (x_min + x_max) / 2
    return pixel_to_gps(metadata, bbox_center)


def get_radius_of_bbox_in_meters(metadata: HeaderMetadata, bbox_pixels):
    x_min, x_max, y_min, y_max = bbox_pixels  # x: cols, y: rows
    semiaxis_length_pixels = (y_max - y_min) / 2, (x_max - x_min) / 2

    height_meters = 2 * metadata.half_image_height_meters
    width_meters = 2 * metadata.half_image_width_meters

    height_meters_per_pixel = height_meters / metadata.image_height
    width_meters_per_pixel = width_meters / metadata.image_width

    semiaxis_length_meters = (
        semiaxis_length_pixels[0] * height_meters_per_pixel,
        semiaxis_length_pixels[1] * width_meters_per_pixel,
    )

    return np.sqrt(semiaxis_length_meters[0] ** 2 + semiaxis_length_meters[1] ** 2)


def bbox_gps_center_and_radius_in_meters(metadata: HeaderMetadata, bbox_pixels):
    """
    Returns the center of the bbox in gps coordinates (lat, lon) and the radius of the bbox in meters

    metadata: HeaderMetadata
    bbox_pixels: (x_min, x_max, y_min, y_max) in pixels where x: cols, y: rows
    """

    center = bbox_pixels_to_center_gps(metadata, bbox_pixels)
    radius = get_radius_of_bbox_in_meters(metadata, bbox_pixels)
    logging.debug(f"Detection for center: {center}, radius: {radius}")
    return center[0], center[1], radius


def get_corner_coordinates(metadata: HeaderMetadata):
    """
    Given the metadata, calculates the gps coordinates of the corners of the image.

    :param metadata: HeaderMetatada
    :returns gps coordinates in the order of top-left, top-right, bottom-right, bottom-left
    """

    # Calculate the distances to the corners
    distance_to_corner = np.sqrt(
        (metadata.half_image_width_meters) ** 2 + (metadata.half_image_height_meters) ** 2
    )

    # angle to the top right corner from y-axis clockwise
    top_right_angle_cw = 90 - np.degrees(
        np.arctan2(metadata.half_image_height_meters, metadata.half_image_width_meters)
    )

    # Calculate the bearings from the center to the corners
    bearing_top_left = (metadata.heading - top_right_angle_cw) % 360
    bearing_top_right = (metadata.heading + top_right_angle_cw) % 360
    bearing_bottom_right = (metadata.heading + 180 - top_right_angle_cw) % 360
    bearing_bottom_left = (metadata.heading + 180 + top_right_angle_cw) % 360

    # Calculate the GPS coordinates of the corners
    top_left = destination_point(
        metadata.latitude, metadata.longitude, bearing_top_left, distance_to_corner
    )
    top_right = destination_point(
        metadata.latitude, metadata.longitude, bearing_top_right, distance_to_corner
    )
    bottom_right = destination_point(
        metadata.latitude, metadata.longitude, bearing_bottom_right, distance_to_corner
    )
    bottom_left = destination_point(
        metadata.latitude, metadata.longitude, bearing_bottom_left, distance_to_corner
    )

    return top_left, top_right, bottom_right, bottom_left
