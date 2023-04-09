import PIL.Image as Image
from exif import Image as ExifImage

image_path = "2023-04-08_01-56-52_RGB.jpg"

with open(image_path, 'rb') as image_file:
    exif_img = ExifImage(image_file)
    
#print(exif_img.get_all())

print(exif_img.gps_latitude)
print(exif_img.gps_latitude_ref)
print(exif_img.gps_longitude)
print(exif_img.gps_longitude_ref)
print(exif_img.gps_altitude)
print(exif_img.gps_img_direction)
print(exif_img.gps_img_direction_ref)
print(exif_img.focal_length)
print(exif_img.focal_plane_x_resolution)
print(exif_img.focal_plane_y_resolution)

img = Image.open(image_path)

print(img.width)
print(img.height)