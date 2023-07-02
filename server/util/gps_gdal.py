import tempfile
from functools import lru_cache

from osgeo import gdal, osr, ogr

# @lru_cache(maxsize=1)
def _gdal_dataset(dataset_path, corner_gps_coordinates=None):
    ds = gdal.Open(dataset_path, gdal.GA_ReadOnly)
    InSR = osr.SpatialReference()

    if corner_gps_coordinates is None:
        InSR.ImportFromWkt(ds.GetProjectionRef())
        return ds, InSR
    
    # if ds.GetProjectionRef() == "":
    #     InSR.ImportFromEPSG(4326)  # assumes WGS84
    #     ds.SetProjection(InSR.ExportToWkt())
    # else:
    #     InSR.ImportFromWkt(ds.GetProjectionRef())

    gcp_list = [
        gdal.GCP(*corner_gps_coordinates[0][::-1], 0, 0, 0),
        gdal.GCP(*corner_gps_coordinates[1][::-1], 0, ds.RasterXSize, 0),
        gdal.GCP(*corner_gps_coordinates[2][::-1], 0, ds.RasterXSize, ds.RasterYSize),
        gdal.GCP(*corner_gps_coordinates[3][::-1], 0, 0, ds.RasterYSize),
    ]

    tf = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)

    ds = gdal.Translate(tf.name, ds, GCPs=gcp_list)
    ds = gdal.Warp(tf.name, ds, dstSRS='EPSG:4326')
    InSR.ImportFromWkt(ds.GetProjectionRef())

    return ds, InSR


def _pixel_to_proj_coordinates(gt, column, row):
    a, b, c, d, e, f = gt
    x = a + column * b + row * c
    y = d + column * e + row * f
    
    return x, y

def pixel_to_wgs84(image_path, pixel, corner_gps_coordinates):
    """
    Converts a pixel coordinate to a GPS coordinate.

    :param dataset_path: path to the image
    :param pixel: tuple of (row, column) coordinates
    :param corner_gps_coordinates: list of tuples of (latitude, longitude) coordinates
        in the order of top_left, top_right, bottom_right, bottom_left
    
    :return: tuple of (latitude, longitude) coordinates
    """

    ds, InSR = _gdal_dataset(image_path, corner_gps_coordinates)

    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(4326)

    gt = ds.GetGeoTransform()
    column, row = pixel
    x, y = _pixel_to_proj_coordinates(gt, column, row)

    transform = osr.CoordinateTransformation(InSR, OutSR)

    latlong = transform.TransformPoint(x, y)
    latitude, longitude = latlong[:2][::-1]

    return latitude, longitude
