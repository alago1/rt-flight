from osgeo import gdal

gdal.UseExceptions()
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer


def pixel_to_wgs84(image_shape, pixel, corner_gps_coordinates):
    """
    Converts pixel location in image to GPS coordinates in wgs84.

    :param image_shape: tuple of (height, width) of image
    :param pixel: tuple of (row, column) coordinates
    :param corner_gps_coordinates: list of tuples of (latitude, longitude) coordinates
        in the order of top_left, top_right, bottom_right, bottom_left

    :return: tuple of (latitude, longitude) coordinates
    """
    height, width = image_shape
    corners = [(0, 0), (0, width), (height, width), (height, 0)]

    gcps = [
        GroundControlPoint(row=row, col=col, x=lat, y=lon)
        for (row, col), (lat, lon) in zip(corners, corner_gps_coordinates)
    ]

    transformer = GCPTransformer(gcps)
    return transformer.xy(*pixel)
