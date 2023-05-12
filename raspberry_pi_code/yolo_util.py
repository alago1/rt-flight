# Based on implementation from: https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/yolo_postprocess_np.py

import numpy as np
import PIL

import copy

# yolov3 anchors
# v3_anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
v3_anchor_mask = np.array([[6,7,8], [3,4,5], [0,1,2]])
v3_anchors = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def preprocess_image(image_pil, net_input_shape):
    img_width, img_height = image_pil.size
    net_input_height, net_input_width = net_input_shape[:2]

    scale = min(net_input_width / img_width, net_input_height / img_height)
    padding_width = int(img_width * scale)
    padding_height = int(img_height * scale)
    
    dx = (net_input_width - padding_width) // 2
    dy = (net_input_height - padding_height) // 2
    offset = (dx, dy)

    img = image_pil.resize((padding_width, padding_height), resample=PIL.Image.BICUBIC)
    new_image = PIL.Image.new("RGB", net_input_shape[:2], (128, 128, 128))
    new_image.paste(img, offset)
    img = new_image

    img = np.asarray(img).astype(np.float32) / 255.
    img = np.expand_dims(img, axis=0)
    return img


def decode_net_output(prediction, anchors, num_classes, input_shape):
    '''Decode final layer features to bounding box parameters.'''
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_shape = np.shape(prediction)[1:3]
    #check if stride on height & width are same
    assert input_shape[0]//grid_shape[0] == input_shape[1]//grid_shape[1], 'model stride mismatch.'
    stride = input_shape[0] // grid_shape[0]

    prediction = np.reshape(prediction,
                            (batch_size, grid_shape[0] * grid_shape[1] * num_anchors, num_classes + 5))

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_shape[0])
    grid_x = np.arange(grid_shape[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    ################################

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_shape[0] * grid_shape[1], 1))
    anchors = np.expand_dims(anchors, 0)

    box_xy = (_sigmoid(prediction[..., :2]) + x_y_offset) / np.array(grid_shape)[::-1]
    box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_shape)[::-1]

    # Sigmoid objectness scores
    objectness = _sigmoid(prediction[..., 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1

    # Sigmoid class scores
    class_scores = _sigmoid(prediction[..., 5:])

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def correct_boxes(predictions, img_shape, model_input_shape):
    '''rescale predicition boxes back to original image shape'''
    box_xy = predictions[..., :2]
    box_wh = predictions[..., 2:4]
    objectness = np.expand_dims(predictions[..., 4], -1)
    class_scores = predictions[..., 5:]

    # model_input_shape & image_shape should be (height, width) format
    model_input_shape = np.array(model_input_shape, dtype='float32')
    image_shape = np.array(img_shape, dtype='float32')

    new_shape = np.round(image_shape * np.min(model_input_shape/image_shape))
    offset = (model_input_shape-new_shape)/2./model_input_shape
    scale = model_input_shape/new_shape
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    # Convert centroids to top left coordinates
    box_xy -= box_wh / 2

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    box_xy *= image_wh
    box_wh *= image_wh

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def handle_predictions(predictions, num_classes, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    # check if only 1 class for different score
    if num_classes == 1:
        box_scores = box_confidences
    else:
        box_scores = box_confidences * box_class_probs

    # filter boxes with score threshold
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    # Boxes, Classes and Scores returned from NMS
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold, confidence=confidence)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype('int32')
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)

        return boxes, classes, scores

    else:
        return [], [], []
    

def nms_boxes(boxes, classes, scores, iou_threshold, confidence=0.1, is_soft=False, use_exp=False, sigma=0.5):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and first line
            b_nms[[i,0],:] = b_nms[[0,i],:]
            c_nms[[i,0]] = c_nms[[0,i]]
            s_nms[[i,0]] = s_nms[[0,i]]

            iou = box_diou(b_nms)

            # drop the last line since it has been record
            b_nms = b_nms[1:]
            c_nms = c_nms[1:]
            s_nms = s_nms[1:]

            # normal Hard-NMS
            keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores


def box_diou(boxes):
    """
    Calculate DIoU value of 1st box with other boxes of a box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    diou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)

    # box center distance
    x_center = x + w/2
    y_center = y + h/2
    center_distance = np.power(x_center[1:] - x_center[0], 2) + np.power(y_center[1:] - y_center[0], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x[1:], x[0])
    enclose_ymin = np.minimum(y[1:], y[0])
    enclose_xmax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_ymax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + np.finfo(float).eps)

    return diou


def filter_boxes(boxes, classes, scores, max_boxes):
    '''
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    '''
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]

    return nboxes, nclasses, nscores


def adjust_boxes(boxes, img_shape):
    '''
    change box format from (x,y,w,h) top left coordinate to
    (xmin,ymin,xmax,ymax) format
    '''
    if boxes is None or len(boxes) == 0:
        return []

    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape

    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin,ymin,xmax,ymax])

    return np.array(adjusted_boxes,dtype=np.int32)