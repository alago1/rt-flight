import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from exiftool import ExifToolHelper
from geopy.distance import distance as geopy_distance


def mock_flight_path(
        large_orthomosaic_path,
        num_gcps,
        corner_gps_coordinates,
        altitude,
        mock_output_dim=(600, 600),
        seed=None):
    """
    Mocks the flight path of a UAV by cropping a large orthomosaic
    along a random path and returning each cropped image in sequence.

    :param large_orthomosaic_path: path to large orthomosaic image
    :param num_gcps: number of ground control points (GCPs) on the random path
    :param corner_gps_coordinates: 2D array of GPS coordinates of the corners of the orthomosaic image
    :param mock_output_dim: 2D array of width-height dimensions of individual mock output images
    :param seed: seed for random number generator
    """
    if seed is not None:
        np.random.seed(seed)

    orthomosaic = np.asarray(Image.open(large_orthomosaic_path))[:, :, :3]
    diag_len = np.sqrt(mock_output_dim[0]**2 + mock_output_dim[1]**2)
    gcps = random_ground_control_points(
        orthomosaic.shape,
        num_gcps,
        padding=(int(diag_len), int(diag_len)),
    )

    pixel_path = build_path_pixels(gcps)

    for x, y, angle in pixel_path:
        yield create_cropped_image(
            orthomosaic,
            (x, y),
            angle,
            mock_output_dim,
            corner_gps_coordinates,
            altitude,
            large_orthomosaic_path
        )


def random_ground_control_points(orthomosaic_shape, num_gcps, padding=(0, 0)):
    """
    Generates a list of random ground control points (GCPs) for a given orthomosaic image.

    :param orthomosaic_shape: 2D array of width-height dimensions of orthomosaic image
    :param num_gcps: number of GCPs to generate
    :param padding: 2D array of width-height padding to apply to orthomosaic image
    :return: list of GCPs
    """
    return np.random.randint(
        padding,
        high=(
            orthomosaic_shape[0] - padding[0],
            orthomosaic_shape[1] - padding[1],
        ),
        size=(num_gcps, 2),
    )


def build_path_pixels(gcps, step_size=100):
    """
    Computes a path between a list of ground control points (GCPs) in pixels.
    Interpolates linearly between each pair of GCPs.
    Interpolates angle at each GCP.

    :param gcps: list of GCPs
    :return: list of pixel locations and direction along the path
    """
    delta = np.diff(gcps, axis=0)
    directions = delta / np.linalg.norm(delta, axis=1).reshape(-1, 1)
    angles = -np.arctan2(directions.T[1], directions.T[0]) * 180 / np.pi
    delta_angles = np.append(np.diff(angles), 0)

    path = []

    for t1, t2, angle, delta_angle in zip(gcps, gcps[1:], angles, delta_angles):
        steps = np.linalg.norm(t2 - t1) / step_size
        line = np.linspace(t1, t2, steps.astype("uint32"), dtype="uint32")
        path.extend([np.array([x, y, angle]) for x, y in line])

        if delta_angle == 0 or len(line) == 0:
            continue

        interpolated_angles = np.linspace(angle, angle + delta_angle, 3)
        path.extend([
            np.array([line[-1][0], line[-1][1], theta])
            for theta in interpolated_angles
        ])

    return path


def create_cropped_image(image, pixel_center, angle, cropped_shape, corner_gps_coordinates, altitude, img_path):
    """
    Creates a cropped image file with EXIF metadata based on a large orthomosaic image.

    :param image: large orthomosaic image
    :param pixel_center: 2D array of pixel location
    :param angle: angle of UAV at pixel location
    :param cropped_shape: 2D array of width-height dimensions of cropped image
    :param corner_gps_coordinates: 2D array of GPS coordinates of the corners of the orthomosaic image
        from top-left to bottom-left in clockwise order
    :param altitude: altitude of UAV
    :return: path to cropped image file
    """
    cropped_diag = np.sqrt(cropped_shape[0]**2 + cropped_shape[1]**2)
    sample = _crop_around(image, pixel_center, (cropped_diag, cropped_diag))
    rotated_sample = _center_crop(rotate(sample, -angle, reshape=False), cropped_shape)
    
    tf = tempfile.NamedTemporaryFile(prefix='rtflight_', suffix=".jpg", delete=False)
    Image.fromarray(rotated_sample).save(tf.name)
    
    meta = build_metadata(pixel_center, angle, image.shape, corner_gps_coordinates, altitude)

    with ExifToolHelper() as et:
        et.set_tags([tf.name], tags=meta, params=["-P", "-overwrite_original"])

    return tf.name


def _crop_around(image, center, dim):
    dim = np.array(dim).astype("uint32")
    x = int(center[1] - dim[1] // 2)
    y = int(center[0] - dim[0] // 2)
    return image[y : y + dim[0], x : x + dim[1]]


def _center_crop(image, dim):
    return image[
        image.shape[0] // 2 - dim[0] // 2 : image.shape[0] // 2 + dim[0] // 2,
        image.shape[1] // 2 - dim[1] // 2 : image.shape[1] // 2 + dim[1] // 2,
    ]


def approx_gps(pixel_center, orthomosaic_shape, corner_gps_coordinates):
    """
    Compute the approximate GPS coordinates of a pixel location in an orthomosaic image
    using the GPS coordinates of the corners of the orthomosaic image.

    :param pixel_center: 2D array of pixel location
    :param orthomosaic_shape: 2D array of width-height dimensions of orthomosaic image
    :param corner_gps_coordinates: 2D array of GPS coordinates of the corners of the orthomosaic image
        in the order [top-left, top-right, bottom-right, bottom-left]ic (used on gdal backend only)

    :return: 2D array of GPS coordinates of pixel location
    """
    right_vec = corner_gps_coordinates[1] - corner_gps_coordinates[0]
    down_vec = corner_gps_coordinates[3] - corner_gps_coordinates[0]
    top_left_relative_loc = right_vec * pixel_center[1] / orthomosaic_shape[1] + down_vec * pixel_center[0] / orthomosaic_shape[0]
    gps_linear = corner_gps_coordinates[0] + top_left_relative_loc
    return gps_linear

def build_metadata(
        pixel_center,
        direction_deg,
        orthomosaic_shape,
        corner_gps_coordinates,
        altitude):
    image_width_meters = geopy_distance(corner_gps_coordinates[0], corner_gps_coordinates[1]).meters
    image_height_meters = geopy_distance(corner_gps_coordinates[0], corner_gps_coordinates[3]).meters

    focal_length = 5e-3  # 5mm
    sensor_width = image_width_meters / altitude * focal_length
    sensor_height = image_height_meters / altitude * focal_length

    lat, lng = approx_gps(pixel_center, orthomosaic_shape, corner_gps_coordinates)

    orthomosaic_bearing = get_bearing(corner_gps_coordinates[2], corner_gps_coordinates[1])
    image_bearing = (orthomosaic_bearing + direction_deg) % 360

    return {
        "GPSLatitude": np.abs(lat),
        "GPSLatitudeRef": "N" if lat >= 0 else "S",
        "GPSLongitude": np.abs(lng),
        "GPSLongitudeRef": "E" if lng >= 0 else "W",
        "GPSAltitude": altitude,
        "GPSImgDirection": image_bearing,
        "GPSImgDirectionRef": "T",
        "FocalLength": focal_length * 1e3,
        "FocalPlaneResolutionUnit": 4,  # mm
        "FocalPlaneXResolution": sensor_width * 1e3,
        "FocalPlaneYResolution": sensor_height * 1e3,
    }


def get_bearing(pos1, pos2):
    lat1, lng1 = pos1
    lat2, lng2 = pos2
    dLon = lng2 - lng1
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    brng = np.rad2deg(np.arctan2(y, x))
    if brng < 0: brng+= 360
    return brng
