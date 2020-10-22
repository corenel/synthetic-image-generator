import imgaug as ia
import os
import numpy as np
import setting

from PIL import Image

ia.seed(1)


def write_yolo_annotations(outpath, annotations, image_width, image_height):
    with open(outpath, 'w') as anno_fp:
        for anno_object in annotations:
            label_id = anno_object['label']
            x = anno_object['coordinates']['x'] / image_width
            y = anno_object['coordinates']['y'] / image_height
            w = anno_object['coordinates']['width'] / image_width
            h = anno_object['coordinates']['height'] / image_height
            anno_fp.write('{} {} {} {} {}\n'.format(label_id, x, y, w, h))


def get_box(obj_w, obj_h, min_x, min_y, max_x, max_y):
    x1, y1 = np.random.randint(min_x, max_x, 1), np.random.randint(min_x, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def get_group_object_positions(object_group, image_background, image_files_object):
    bkg_w, bkg_h = image_background.size
    boxes = []
    objs = [Image.open(image_files_object[i]) for i in object_group]
    obj_sizes = [tuple([int(dim) for dim in obj.size]) for obj in objs]
    for w, h in obj_sizes:
        # set background image boundaries
        min_x, min_y = 2 * w, 2 * h
        max_x, max_y = bkg_w - 2 * w, bkg_h - 2 * h
        # get new box coordinates for the obj on the bkg
        while True:
            new_box = get_box(w, h, min_x, min_y, max_x, max_y)
            for box in boxes:
                res = intersects(box, new_box)
                if res:
                    break
            else:
                break  # only executed if the inner loop did NOT break
            continue  # only executed if the inner loop DID break
        # append our new box
        boxes.append(new_box)
    return obj_sizes, boxes


def get_label_id(filename):
    basename = os.path.basename(filename)
    filename = os.path.splitext(basename)[0]
    return setting.LABELS.index(filename)
