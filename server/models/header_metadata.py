import logging
from dataclasses import dataclass
from typing import Optional

import exiftool
import numpy as np

from ..util.gps import get_corner_coordinates

class HeaderMissingError(Exception):
    pass

@dataclass
class HeaderMetadata:
    image_path: Optional[str] = None

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
        new_metadata.image_path = image_path

        logging.info(f"Metadata: {metadata}")

        new_metadata.image_width = (
            dict.get(metadata, "EXIF:ExifImageWidth")
            or dict.get(metadata, "EXIF:ImageWidth")
            or dict.get(metadata, "File:ImageWidth")
        )
        new_metadata.image_height = (
            dict.get(metadata, "EXIF:ExifImageHeight")
            or dict.get(metadata, "EXIF:ImageHeight")
            or dict.get(metadata, "File:ImageHeight")
        )

        if new_metadata.image_width is None or new_metadata.image_height is None:
            raise HeaderMissingError("Image width and height are missing")

        try:
            new_metadata.latitude = metadata["EXIF:GPSLatitude"]
            new_metadata.longitude = metadata["EXIF:GPSLongitude"]
            new_metadata.altitude = metadata["EXIF:GPSAltitude"]

            new_metadata.heading = (
                dict.get(metadata, "EXIF:GPSImgDirection", 0)
                or dict.get(metadata, "MakerNotes:CameraYaw", 0)
            )
        except KeyError as e:
            raise HeaderMissingError("GPS data is missing") from e

        if new_metadata.heading == 0:
            logging.warning(
                "WARNING: Heading defaulted to 0. The program will continute to run, but this may cause issues."
            )

        try:
            if metadata["EXIF:GPSLatitudeRef"] == "S":
                assert new_metadata.latitude >= 0, "Latitude is negative but ref is S"
                new_metadata.latitude *= -1

            if metadata["EXIF:GPSLongitudeRef"] == "W":
                assert new_metadata.longitude >= 0, "Longitude is negative but ref is W"
                new_metadata.longitude *= -1

            if dict.get(metadata, "EXIF:GPSImgDirectionRef", "T") == "M":
                assert (
                    np.abs(new_metadata.heading) > 2 * np.pi
                ), "Heading is in radians but we assume degrees. Please fix"
                new_metadata.heading -= 8.0  # subtract 8deg to account for magnetic declination
        except KeyError as e:
            raise HeaderMissingError("GPS Direction Ref is missing") from e

        units_to_meter_conversion_factors = [
            None,  # this is the default value
            0.0254,  # inches
            1e-2,  # cm
            1e-3,  # mm
            1e-6,  # um
        ]
        unit_index = dict.get(metadata, "EXIF:FocalPlaneResolutionUnit", 1) - 1
        resolution_conversion_factor = units_to_meter_conversion_factors[unit_index]

        if resolution_conversion_factor is None:
            raise HeaderMissingError("FocalPlaneResolutionUnit is missing")

        try:
            focal_length = metadata["EXIF:FocalLength"] * resolution_conversion_factor
            sensor_width = (
                metadata["EXIF:FocalPlaneXResolution"] * resolution_conversion_factor
            )
            sensor_height = (
                metadata["EXIF:FocalPlaneYResolution"] * resolution_conversion_factor
            )
        except KeyError as e:
            raise HeaderMissingError("FocalLength, FocalPlaneXResolution, or FocalPlaneYResolution is missing") from e

        new_metadata.half_image_width_meters = new_metadata.altitude * sensor_width / focal_length / 2
        new_metadata.half_image_height_meters = new_metadata.altitude * sensor_height / focal_length / 2


        logging.debug(f"Image width, height meters: {2 * new_metadata.half_image_width_meters}, {2 * new_metadata.half_image_height_meters}")

        new_metadata.top_left, new_metadata.top_right, new_metadata.bottom_right, new_metadata.bottom_left = get_corner_coordinates(new_metadata)

        return new_metadata
