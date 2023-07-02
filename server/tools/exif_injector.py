import argparse
import os
import shutil
import pathlib

from exiftool import ExifToolHelper

FP_UNITS = ('in', 'cm', 'mm', 'um')

def parse_args():
    desc = 'Inject Exif metadata into an image'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i', '--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('-o', '--output', type=str, help='Output image or directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite original image')
    parser.add_argument('-la', '--latitude', type=float, help='Latitude')
    parser.add_argument('-lo', '--longitude', type=float, help='Longitude')
    parser.add_argument('-a', '--altitude', type=float, help='Altitude')
    parser.add_argument('-he', '--heading', type=float, help='Heading')

    parser.add_argument('-fpu', '--focal-plane-unit', type=str, choices=FP_UNITS, help='Focal plane unit', default='mm')
    parser.add_argument('-fl', '--focal-length', type=float, help='Focal length of sensor')
    parser.add_argument('-fpx', '--focal-plane-x-resolution', type=float, help='Focal plane x resolution')
    parser.add_argument('-fpy', '--focal-plane-y-resolution', type=float, help='Focal plane y resolution')
    
    args = parser.parse_args()
    return args


def inject_metadata(image_path: str, args):
    tags = dict()

    if args.latitude:
        tags['GPSLatitude'] = args.latitude
        tags['GPSLatitudeRef'] = 'N' if args.latitude >= 0 else 'S'
    
    if args.longitude:
        tags['GPSLongitude'] = args.longitude
        tags['GPSLongitudeRef'] = 'E' if args.longitude >= 0 else 'W'

    if args.altitude:
        tags['GPSAltitude'] = args.altitude
        tags['GPSAltitudeRef'] = 0

    if args.heading:
        tags['GPSImgDirection'] = args.heading
        tags['GPSImgDirectionRef'] = 'T'

    if args.focal_length:
        tags['FocalLength'] = args.focal_length
    
    if args.focal_plane_x_resolution:
        tags['FocalPlaneXResolution'] = args.focal_plane_x_resolution
    
    if args.focal_plane_y_resolution:
        tags['FocalPlaneYResolution'] = args.focal_plane_y_resolution

    if args.focal_plane_unit:
        tags['FocalPlaneResolutionUnit'] = FP_UNITS.index(args.focal_plane_unit) + 2

    with ExifToolHelper() as et:
        et.set_tags(
            [image_path],
            tags=tags,
            params=["-P", "-overwrite_original"],
        )


def is_valid_file(filename):
    prefix, ext = os.path.splitext(filename)

    if '_injected' in os.path.basename(prefix):
        return False

    return ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']


def format_filename(folder_path, filename):
    prefix, ext = os.path.splitext(filename)
    new_filename = f"{prefix}_injected{ext}"
    return os.path.join(folder_path, new_filename)


def get_output_path(args, filename):
    if args.output is None:
        return format_filename(os.path.dirname(args.input), filename)

    if os.path.isdir(args.output):
        filename = os.path.basename(filename)
        return os.path.join(args.output, filename)

    if os.path.isfile(args.output):
        return args.output
    

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        raise SystemExit('Input file or directory does not exist')

    if os.path.isdir(args.input) and os.path.isfile(args.output):
        raise SystemExit('Input is a directory but output is an existing file')
    
    if args.overwrite and args.output is not None and args.overwrite != args.output:
        raise SystemExit('Cannot specify both --overwrite and --output')


    if args.output is None:
        print("Output not specified. Writing to new 'injected' file. If you want to overwrite the original file, use the --overwrite flag.")
    elif not os.path.exists(args.output):
        if os.path.isdir(args.input):
            os.makedirs(args.output)
        else:
            pathlib.Path(args.output).touch()


    if os.path.isdir(args.input):
        files = [os.path.join(args.input, file) for file in os.listdir(args.input) if is_valid_file(file)]
    else:
        files = [args.input]


    for file in files:
        if not os.path.exists(file):
            raise SystemExit(f'File {file} does not exist')

        if args.overwrite:
            print('Overwriting {file} with new exif metadata')
        else:
            output_path = get_output_path(args, file)
            shutil.copy(file, output_path)
            print(f'Writing new exif metadata to {output_path}')        

        inject_metadata(file, args)


if __name__ == '__main__':
    main()
