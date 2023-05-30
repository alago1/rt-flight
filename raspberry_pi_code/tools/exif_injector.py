import numpy as np
import PIL.Image as Image
from exif import GpsAltitudeRef
from exif import Image as ExifImage
from geopy import Point
from geopy.distance import geodesic

image_path = "data.jpg"

# Load the image with the exif library
with open(image_path, 'rb') as image_file:
    exif_img = ExifImage(image_file)

DATASET_TOP_LEFT_GPS = np.array((12.86308254761559, 77.5151947517078))
DATASET_TOP_RIGHT_GPS = np.array((12.863010715187013, 77.52267023737696))
DATASET_BOT_LEFT_GPS = np.array((12.859008245256549, 77.5151541499705))
DATASET_BOT_RIGHT_GPS = np.array((12.858936436333265, 77.52262951527761))

# Image dimensions
image_width = 14180
image_height = 7877

# Pixel location
pixel_x, pixel_y = 7058, 2866

# Calculate scaling factors
longitude_left = DATASET_TOP_LEFT_GPS[1]
longitude_right = DATASET_TOP_RIGHT_GPS[1]
latitude_top = DATASET_TOP_LEFT_GPS[0]
latitude_bottom = DATASET_BOT_LEFT_GPS[0]

x_scaling_factor = (longitude_right - longitude_left) / image_width
y_scaling_factor = (latitude_top - latitude_bottom) / image_height

# Calculate GPS coordinates for the given pixel location
longitude = longitude_left + x_scaling_factor * pixel_x
latitude = latitude_top - y_scaling_factor * pixel_y

latitude  = 12.8614666
longitude = 77.51842268

next_x = 6734.0
next_y = 3165.0

next_long = longitude_left + x_scaling_factor * next_x
next_lat = latitude_top - y_scaling_factor * next_y

print(next_long, next_lat)

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the initial compass bearing (forward azimuth) between two points.
    The result is the angle from pointA to pointB in degrees, with respect to true north.
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = np.radians(pointA[0])
    lat2 = np.radians(pointB[0])

    diff_long = np.radians(pointB[1] - pointA[1])

    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))

    initial_bearing = np.arctan2(x, y)

    # Normalize the result
    initial_bearing = np.degrees(initial_bearing)
    return (initial_bearing + 360) % 360


heading = calculate_initial_compass_bearing((latitude, longitude), (next_lat, next_long))
print(f"Heading from point A to point B: {heading} degrees")


def decimal_to_dms(decimal_degrees):
    degrees = int(decimal_degrees)
    minutes_decimal = abs(decimal_degrees - degrees) * 60
    minutes = int(minutes_decimal)
    seconds_decimal = (minutes_decimal - minutes) * 60
    seconds = round(seconds_decimal, 6)
    return (degrees, minutes, seconds)

# GPS coordinates in decimal format
decimal_latitude = latitude
decimal_longitude = longitude

# Convert decimal coordinates to DMS format
dms_latitude = decimal_to_dms(decimal_latitude)
dms_longitude = decimal_to_dms(decimal_longitude)

print(latitude, longitude)
print(dms_latitude, dms_longitude)

# # Create the EXIF data
exif_img.gps_latitude = dms_latitude
exif_img.gps_latitude_ref = "N"
exif_img.gps_longitude = dms_longitude
exif_img.gps_longitude_ref = "E"
exif_img.gps_altitude = 50
exif_img.gps_altitude_ref = 2
exif_img.gps_img_direction = 222.79
exif_img.gps_img_direction_ref = "T"
exif_img.focal_length = 7.2
exif_img.lens_specification = (6.17, 4.55)
exif_img.pixel_x_dimension = 1000
exif_img.pixel_y_dimension = 1000

# Save the image with the new EXIF data
with open('../output_image.jpg', 'wb') as output_file:
    output_file.write(exif_img.get_file())