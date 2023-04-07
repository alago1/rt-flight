import PIL.Image as Image
from exif import Image as ExifImage

image_path = "../output_image.jpg"

with open(image_path, 'rb') as image_file:
    exif_img = ExifImage(image_file)
    
print(exif_img.get_all())