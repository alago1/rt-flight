from dataclasses import dataclass
from typing import Optional
import logging

import exiftool
import numpy as np

from util.gps import get_corner_coordinates

@dataclass
class HeaderMetadata:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    heading: Optional[float] = None

    half_image_width_meters: Optional[float] = None
    half_image_height_meters: Optional[float] = None

    image_width: Optional[float] = None
    image_height: Optional[float] = None

    # these are for debugging
    top_left: Optional[float] = None
    top_right: Optional[float] = None
    bottom_left: Optional[float] = None
    bottom_right: Optional[float] = None


    @staticmethod
    def read(image_path: str):
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(image_path)[0]

        new_metadata = HeaderMetadata()

        logging.info(f"Metadata: {metadata}")

        new_metadata.image_width = (
            dict.get(metadata, "EXIF:ExifImageWidth", None)
            or dict.get(metadata, "EXIF:ImageWidth", None)
            or dict.get(metadata, "File:ImageWidth")
        )
        new_metadata.image_height = (
            dict.get(metadata, "EXIF:ExifImageHeight", None)
            or dict.get(metadata, "EXIF:ImageHeight", None)
            or dict.get(metadata, "File:ImageHeight")
        )

        new_metadata.latitude = metadata["EXIF:GPSLatitude"]
        new_metadata.longitude = metadata["EXIF:GPSLongitude"]
        new_metadata.altitude = metadata["EXIF:GPSAltitude"]
        new_metadata.heading = dict.get(metadata, "EXIF:GPSImgDirection", 0)

        if new_metadata.heading == 0 and logger:
            logger.warning(
                "WARNING: Heading defaulted to 0. The program will continute to run, but this may cause issues."
            )

        if metadata["EXIF:GPSLatitudeRef"] == "S":
            assert new_metadata.latitude >= 0, "Latitude is negative but ref is S"
            new_metadata.latitude *= -1

        if metadata["EXIF:GPSLongitudeRef"] == "W":
            assert new_metadata.longitude >= 0, "Longitude is negative but ref is W"
            new_metadata.longitude *= -1

        if metadata["EXIF:GPSImgDirectionRef"] == "M":
            assert (
                np.abs(new_metadata.heading) > 2 * np.pi
            ), "Heading is in radians but we assume degrees. Please fix"
            new_metadata.heading -= 8.0  # subtract 8deg to account for magnetic declination

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

        new_metadata.half_image_width_meters = new_metadata.altitude * sensor_width / focal_length
        new_metadata.half_image_height_meters = new_metadata.altitude * sensor_height / focal_length

        new_metadata.top_left, new_metadata.top_right, new_metadata.bottom_left, new_metadata.bottom_right = get_corner_coordinates(new_metadata)

        return new_metadata
