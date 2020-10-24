import os

import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import setting

ia.seed(1)


def read_yolo_annotations(inpath, image_width, image_height):
    """
    Read annotations (in YOLO format) form file

    :param inpath: filepath to annotation file
    :type inpath: str
    :param image_width: width of image
    :type image_width: int
    :param image_height: height of image
    :type image_height: int
    :return: parsed bounding box annotations
    :rtype: BoundingBoxesOnImage
    """
    with open(inpath, 'r') as fp:
        lines = fp.readlines()
    bb_list = []
    for line in lines:
        items = line.split(' ')
        if len(items) < 5:
            print('Invalid anno line: {}'.format(line))
        label_id, x, y, w, h = items
        x = float(x) * image_width
        y = float(y) * image_height
        w = float(w) * image_width
        h = float(h) * image_height
        label_id = int(label_id)
        bb_list.append(
            BoundingBox(x1=x - w / 2,
                        y1=y - h / 2,
                        x2=x + w / 2,
                        y2=y + w / 2,
                        label=label_id))
    bbs = BoundingBoxesOnImage(bounding_boxes=bb_list,
                               shape=(image_height, image_width))
    return bbs


def write_yolo_annotations(outpath, annotations, image_width, image_height):
    """
    Write annotations into file following the YOLO format

    :param outpath: filepath to save
    :type outpath: str
    :param annotations: annotations of bounding boxes
    :type annotations: list
    :param image_width: width of image
    :type image_width: int
    :param image_height: height of image
    :type image_height: int
    """
    with open(outpath, 'w') as anno_fp:
        for anno_object in annotations:
            label_id = anno_object['label']
            x = anno_object['coordinates']['x'] / image_width
            y = anno_object['coordinates']['y'] / image_height
            w = anno_object['coordinates']['width'] / image_width
            h = anno_object['coordinates']['height'] / image_height
            anno_fp.write('{} {} {} {} {}\n'.format(label_id, x, y, w, h))


def get_box(obj_w, obj_h, min_x, min_y, max_x, max_y):
    """
    Generate a random bounding box for object to paste

    :param obj_w: width of object
    :type obj_w: int
    :param obj_h: height of object
    :type obj_h: int
    :param min_x: minimum value of position x
    :type min_x: int
    :param min_y: minimum value of position y
    :type min_y: int
    :param max_x: maximum value of position x
    :type max_x: int
    :param max_y: maximum value of position y
    :type max_y: int
    :return: generated bboxes
    :rtype: list[int]
    """
    x1, y1 = np.random.randint(min_x, max_x,
                               1), np.random.randint(min_y, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


def intersects(box, new_box):
    """
    Check whether two bounding boxes are intersected

    :param box: one bounding box
    :type box: list[int]
    :param new_box: another bounding box
    :type new_box: list[int]
    :return: whether two bounding boxes are intersected
    :rtype: bool
    """
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def get_group_object_positions(object_group, image_background, dataset_object):
    """
    Generate positions for grouped object to paste on background image

    :param object_group: group of objects to appear
    :type object_group: list[int]
    :param image_background: background image
    :type image_background: numpy.array
    :param dataset_object: dataset of object images
    :type dataset_object: dataset.ObjectImageFolderDataset
    :return: size and bounding oxes of grouped objects
    """
    bkg_w, bkg_h = image_background.size
    boxes = []
    objs = []
    labels = []
    obj_sizes = []
    for i in object_group:
        obj, label = dataset_object[i]
        objs.append(obj)
        labels.append(label)
        factor = min([
            (setting.OBJECT_SCALE_FACTOR * image_background.size[dim]) /
            obj.size[dim] for dim in range(len(obj.size))
        ])
        obj_sizes.append(
            tuple(int(obj.size[dim] * factor) for dim in range(len(obj.size))))

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
    return objs, labels, obj_sizes, boxes


def get_label_id(filename):
    """
    Convert filename to label index

    :param filename: input object filename
    :type filename: str
    :return: label index
    :rtype: int
    """
    basename = os.path.basename(filename)
    filename = os.path.splitext(basename)[0]
    return setting.LABELS.index(filename)


def sometimes(aug):
    """
    Return a shortcut for iaa.Sometimes

    :param aug: augmentation method
    :type aug: iaa.meta.Augmenter
    :return: wrapped augmentation method
    :rtype: iaa.meta.Augmenter
    """
    return iaa.Sometimes(0.5, aug)


def build_augment_sequence_for_object():
    """
    Build augmentation sequence for object

    :return: aug for object
    :rtype: iaa.Sequential
    """
    return iaa.Sequential(
        [
            # crop images by -5% to 10% of their height/width
            sometimes(
                iaa.CropAndPad(
                    percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))),
            sometimes(
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={
                        'x': (0.9, 1.1),
                        'y': (0.9, 1.1)
                    },
                    rotate=(-5, 5),  # rotate by -45 to +45 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode=ia.ALL)),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf(
                (0, 3),
                [
                    iaa.OneOf([
                        # blur images with a sigma between 0 and 3.0
                        iaa.GaussianBlur((0, 1.0)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.AverageBlur(k=(2, 5)),
                        # blur image using local medians with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 7)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0),
                                lightness=(0.75, 1.5)),  # sharpen images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    # iaa.SimplexNoiseAlpha(iaa.OneOf([
                    #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    #     iaa.DirectedEdgeDetect(
                    #         alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    # ])),
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # change brightness of images (by -10 to 10 of original value)
                    iaa.Add((-10, 10), per_channel=0.5),
                ]),
            # move pixels locally around (with random strengths)
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
            # sometimes move parts of the image around
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            sometimes(
                iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=False))
        ],
        random_order=True)


def build_augment_sequence_for_background():
    """
    Build augmentation sequence for background

    :return: aug for background
    :rtype: iaa.Sequential
    """
    return iaa.Sequential(
        [
            # crop images by -5% to 10% of their height/width
            sometimes(
                iaa.CropAndPad(
                    percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))),
            sometimes(
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={
                        'x': (0.8, 1.2),
                        'y': (0.8, 1.2)
                    },
                    # translate by -20 to +20 percent (per axis)
                    translate_percent={
                        'x': (-0.05, 0.05),
                        'y': (-0.05, 0.05)
                    },
                    rotate=(-10, 10),  # rotate by -45 to +45 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode=ia.ALL)),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf(
                (0, 3),
                [
                    iaa.OneOf([
                        # blur images with a sigma between 0 and 3.0
                        iaa.GaussianBlur((0, 3.0)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.AverageBlur(k=(2, 7)),
                        # blur image using local medians with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0),
                                lightness=(0.75, 1.5)),  # sharpen images
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.OneOf([
                        # randomly remove up to 10% of the pixels
                        iaa.Dropout((0.01, 0.015), per_channel=0.1),
                        iaa.CoarseDropout((0.01, 0.015),
                                          size_percent=(0.02, 0.05),
                                          per_channel=0.1),
                    ]),
                    # change brightness of images (by -10 to 10 of original value)
                    iaa.Add((-10, 10), per_channel=0.5),
                ]),
            sometimes(
                iaa.PerspectiveTransform(scale=(0.01, 0.03), keep_size=False))
        ],
        random_order=True)
