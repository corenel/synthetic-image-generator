import os

import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

import setting

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
    x1, y1 = np.random.randint(min_x, max_x, 1), np.random.randint(min_y, max_y, 1)
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
    obj_sizes = [tuple([int(setting.OBJECT_SCALE_FACTOR * dim) for dim in obj.size]) for obj in objs]
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


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


def build_augment_sequence_for_object():
    return iaa.Sequential([
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
            rotate=(-5, 5),  # rotate by -45 to +45 degrees
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            # if mode is constant, use a cval between 0 and 255
            cval=(0, 255),
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            mode=ia.ALL
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
                   [
                       iaa.OneOf([
                           # blur images with a sigma between 0 and 3.0
                           iaa.GaussianBlur((0, 1.0)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.AverageBlur(k=(2, 5)),
                           # blur image using local medians with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 7)),
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(
                           0.75, 1.5)),  # sharpen images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       # iaa.SimplexNoiseAlpha(iaa.OneOf([
                       #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                       #     iaa.DirectedEdgeDetect(
                       #         alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       # ])),
                       # add gaussian noise to images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(
                           0.0, 0.05 * 255), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Add((-10, 10), per_channel=0.5),
                   ]),
        # move pixels locally around (with random strengths)
        sometimes(iaa.ElasticTransformation(
            alpha=(0.5, 3.5), sigma=0.25)),
        # sometimes move parts of the image around
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=False))
    ],
        random_order=True)


def build_augment_sequence_for_background():
    return iaa.Sequential([
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
            # translate by -20 to +20 percent (per axis)
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            # if mode is constant, use a cval between 0 and 255
            cval=(0, 255),
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            mode=ia.ALL
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
                   [
                       iaa.OneOf([
                           # blur images with a sigma between 0 and 3.0
                           iaa.GaussianBlur((0, 3.0)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local medians with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(
                           0.75, 1.5)),  # sharpen images
                       # add gaussian noise to images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(
                           0.0, 0.05 * 255), per_channel=0.5),
                       iaa.OneOf([
                           # randomly remove up to 10% of the pixels
                           iaa.Dropout((0.01, 0.015), per_channel=0.1),
                           iaa.CoarseDropout((0.01, 0.015), size_percent=(
                               0.02, 0.05), per_channel=0.1),
                       ]),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Add((-10, 10), per_channel=0.5),
                   ]),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03), keep_size=False))
    ],
        random_order=True)
