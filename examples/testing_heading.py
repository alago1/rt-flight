import math
import exiftool

def main():

    IMAGE = 607

    IMAGE_PATH1 = "../data_ignore/sequencial_data/DJI_" + str(IMAGE - 1).zfill(4) + ".JPG"
    IMAGE_PATH2 = "../data_ignore/sequencial_data/DJI_" + str(IMAGE).zfill(4) + ".JPG"
    LATITUDE_KEY = "EXIF:GPSLatitude"
    LONGITUDE_KEY = "EXIF:GPSLongitude"
    FLIGHT_YAW_KEY = "XMP:FlightYawDegree"
    GIMBAL_YAW_KEY = "XMP:GimbalYawDegree"

    with exiftool.ExifToolHelper() as et:
        metadata1 = et.get_metadata(IMAGE_PATH1)[0]
    
    with exiftool.ExifToolHelper() as et:
        metadata2 = et.get_metadata(IMAGE_PATH2)[0]

    origin = [float(metadata1.get(LATITUDE_KEY)), float(metadata1.get(LONGITUDE_KEY))]
    dest = [float(metadata2.get(LATITUDE_KEY)), float(metadata2.get(LONGITUDE_KEY))]

    calculated_heading = calculate_heading(*origin, *dest)

    flight_yaw = (float(metadata2.get(FLIGHT_YAW_KEY)) + 360) % 360
    gimbal_yaw = (float(metadata2.get(GIMBAL_YAW_KEY)) + 360) % 360

    print("For Image: " + str(IMAGE))
    print("Calculated Heading: " + str(calculated_heading))
    print("Flight Yaw: " + str(flight_yaw))
    print("Gimbal Yaw: " + str(gimbal_yaw))
    print(origin)

def calculate_heading(lat1, lon1, lat2, lon2):
    """
    Calculates the heading in degrees between two GPS coordinates using the Haversine formula.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate the difference in longitude and latitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the bearing
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)

    # Convert to degrees
    heading = math.degrees(bearing)

    # Ensure heading is between 0 and 360 degrees
    if heading < 0:
        heading += 360

    return heading

if __name__ == "__main__":
    main()