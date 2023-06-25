import argparse
import os

import cv2

from layers.detector import DetectionLayer


def parse_args():
    """Parse command-line arguments"""
    desc = 'Run object detection and output image with bounding boxes'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i', '--image', type=str, required=True, help='Input image for object detection')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-o', '--output', type=str, help='Output file with marked bounding boxes')

    args = parser.parse_args()
    return args


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

    kwargs = {'providers': [('CUDAExecutionProvider', 'TensorRTExecutionProvider', 'CPUExecutionProvider')]} \
        if args.model.endswith('.onnx') else {}
    
    detection_layer = DetectionLayer(args.model, engine="auto", **kwargs)
    bboxes = detection_layer.run(args.image)


    for x0, x1, y0, y1, _ in bboxes:
        img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 5)
    
    cv2.imwrite(output, img)


if __name__ == "__main__":
    main()