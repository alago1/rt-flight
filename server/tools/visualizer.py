import argparse
import os

import cv2

from server.layers.detector import DetectionLayer
from server.layers.gps_translator import GPSTranslationLayer
from server.layers.header_reader import HeaderReader
from server.layers.parallel import ParallelLayer
from server.models.bbox import BBox


def parse_args():
    """Parse command-line arguments"""
    desc = 'Run object detection and output image with bounding boxes'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i', '--image', type=str, required=True, help='Input image for object detection')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-o', '--output', type=str, help='Output file with marked bounding boxes')

    args = parser.parse_args()
    return args


def pipeline(image_path: str, model_path: str):
    kwargs = {'providers': [('CUDAExecutionProvider', 'TensorRTExecutionProvider', 'CPUExecutionProvider')]} \
        if model_path.endswith('.onnx') else {}
    
    image_processing_layer = ParallelLayer([
        HeaderReader(),
        DetectionLayer(model_path, engine="auto", **kwargs),
    ])
    gps_translation_layer = GPSTranslationLayer()

    header, bboxes = image_processing_layer.run((image_path,), share_input=True)

    if len(bboxes) == 0:
        return [], []
    
    gps_output = [
        BBox(
            latitude=float(bbox[0]),
            longitude=float(bbox[1]),
            radius=float(bbox[2]),
            confidence=float(bbox[3])
        )
        for bbox in gps_translation_layer.run(header, bboxes)
    ]

    return bboxes, gps_output


def draw_label(img,
              text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0),
              bg_padding=(0, 5)
            ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (pos[0] - bg_padding[0], pos[1] - bg_padding[1]), (x + text_w + bg_padding[0], y + text_h + bg_padding[1]), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"Error: file {args.image} not found.")
    
    if args.output is not None and os.path.isdir(args.output):
        raise SystemExit(f"Error: file {args.output} exists and is a directory.")

    output = args.output
    if output is None:
        output = os.path.splitext(args.image)[0] + '_annotated.jpg'
        print(f'Output path not provided. Writing to {output}')

    img = cv2.imread(args.image)

    bboxes, gps_output = pipeline(args.image, args.model)

    for (x0, x1, y0, y1, _), info in zip(bboxes, gps_output):
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 5)
        label = f'({info.latitude:.6f}, {info.longitude:.6f}). {int(info.confidence)}%. {info.radius:.2f}m'
        draw_label(img, label, pos=(x0, y0 - 10))

    cv2.imwrite(output, img)


if __name__ == "__main__":
    main()
