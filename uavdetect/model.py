import json
import numpy as np
import cv2
import keras
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle

JSON_PATH = "../uavdetect_config.json"
READY = False
DEBUG = False
CONFIG_PATH = None
WEIGHTS_PATH = None
CLASSES_PATH = None

weights_file, cfg_file, classes_file = None
jdata = None
model = None
classes = None

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.get_score

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def initialize(READY=READY):
    if READY == True:
        return 0
    with open(JSON_PATH, "r") as fjson:
        jdata = json.load(fjson)
        CONFIG_PATH = jdata["config_path"]
        WEIGHTS_PATH = jdata["weights_path"]
        CLASSES_PATH = jdata["classes_path"]
    cfg_file = CONFIG_PATH
    weights_file = WEIGHTS_PATH
    classes_file = CLASSES_PATH
    model = keras.models.load_model(WEIGHTS_PATH, compile=False)

    READY = True


# draw all results
def draw_boxes(image, v_boxes, v_labels, v_scores):
    #     # load the image
    #     data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(image)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=1)
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='red')
    # show the plot
    pyplot.show()

# returns bounding boxes detected in the image
def detect(imgpath):
    image = cv2.imread(imgpath)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_file, cfg_file)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    bound_boxes = []
    for i in indices:
        try:
            box = boxes[i]

        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        bound_box = BoundBox(x, y, x + w, y + h, confidences[i], class_ids[i])
        bound_boxes.append(bound_box)

    v_boxes = bound_boxes
    v_labels = class_ids
    v_scores = confidences

    if DEBUG == True:
        draw_boxes(image, v_boxes, v_labels, v_scores)
    return v_boxes

if __name__ == "__main__":
    initialize()