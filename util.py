import os

import imgaug as ia
from imgaug.augmenters.meta import SomeOf
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image

import setting

ia.seed(1)


def xywh_to_bbox(label, x, y, w, h):
    return BoundingBox(x1=x - w / 2,
                       y1=y - h / 2,
                       x2=x + w / 2,
                       y2=y + h / 2,
                       label=label)


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
        label, x, y, w, h = items
        x = float(x) * image_width
        y = float(y) * image_height
        w = float(w) * image_width
        h = float(h) * image_height
        label = int(label)
        bb_list.append(xywh_to_bbox(label, x, y, w, h))
    bbs = BoundingBoxesOnImage(bounding_boxes=bb_list,
                               shape=(image_height, image_width))
    return bbs


def write_yolo_annotations(outpath, annotations, image_width, image_height):
    """
    Write annotations into file following the YOLO format

    :param outpath: filepath to save
    :type outpath: str
    :param annotations: annotations of bounding boxes
    :type annotations: BoundingBoxesOnImage
    :param image_width: width of image
    :type image_width: int
    :param image_height: height of image
    :type image_height: int
    """
    with open(outpath, 'w') as f:
        for anno in annotations.remove_out_of_image().clip_out_of_image():
            label = anno.label
            x = anno.center_x / image_width
            y = anno.center_y / image_height
            w = anno.width / image_width
            h = anno.height / image_height
            f.write('{} {} {} {} {}\n'.format(label, x, y, w, h))


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


def get_group_object_positions(object_group, image_background, dataset_object,
                               aug_object):
    """
    Generate positions for grouped object to paste on background image

    :param object_group: group of objects to appear
    :type object_group: list[int]
    :param image_background: background image
    :type image_background: numpy.array
    :param dataset_object: dataset of object images
    :type dataset_object: dataset.ObjectImageFolderDataset
    :param aug_object: augment instance for object
    :type aug_object: iaa.Sequential
    :return: size and bounding oxes of grouped objects
    """
    bkg_w, bkg_h = image_background.size
    boxes = []
    objs = []
    labels = []
    obj_sizes = []
    for i in object_group:
        # load data
        obj, label = dataset_object[i]
        # TODO move transforms into dataset getting method
        # resize obj
        factor = min([
            (setting.OBJECT_INIT_SCALE_FACTOR * image_background.size[dim]) /
            obj.size[dim] for dim in range(len(obj.size))
        ])
        obj_size = tuple(
            int(obj.size[dim] * factor) for dim in range(len(obj.size)))
        obj_w, obj_h = obj_size
        obj = obj.resize((obj_w, obj_h))
        obj = resize_image(obj)
        # augment obj
        obj_aug = Image.fromarray(aug_object.augment_images([np.array(obj)])[0])
        # add to list
        objs.append(obj_aug)
        labels.append(label)
        obj_sizes.append(obj_aug.size)

    for w, h in obj_sizes:
        # set background image boundaries
        if len(boxes) == 0 or not setting.OBJECT_IN_LINE:
            min_x, min_y = 2 * w, 2 * h
            max_x, max_y = bkg_w - 10 * w, bkg_h - 10 * h
        else:
            min_x = boxes[-1][2] + 1
            min_y = boxes[-1][1] + 1
            max_x = min(bkg_w - 2 * w,
                        boxes[-1][2] + np.random.randint(2, 3, 1)[0])
            max_y = min(bkg_h - 2 * h,
                        boxes[-1][1] + np.random.randint(2, 3, 1)[0])
        if min_x >= max_x or min_y >= max_y:
            print('Ignore invalid box: ', w, h, min_x, min_y, max_x, max_y)
            continue
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


def resize_image(image):
    """
    Resize image by random scale factor
    """
    resize_rate = np.random.choice(
        setting.OBJECT_AUG_SCALE_FACTOR) + np.random.uniform(low=-0.1, high=0.1)
    image = image.resize(
        [int(image.width * resize_rate),
         int(image.height * resize_rate)], Image.BILINEAR)

    return image


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
    return iaa.Sequential([
        sometimes(
            iaa.CropAndPad(
                percent=(-0.05, 0.075), pad_mode=ia.ALL, pad_cval=(0, 255))),
        sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-1, 1))),
        sometimes(iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)),
        iaa.SomeOf((0, 2), [
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),
                iaa.AverageBlur(k=(1, 3)),
                iaa.MedianBlur(k=(1, 3)),
            ]),
            iaa.Affine(scale={
                'x': (0.9, 1.1),
                'y': (0.9, 1.1)
            },
                       rotate=(-5, 5),
                       order=[0, 1],
                       cval=(0, 255),
                       mode=ia.ALL),
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        ]),
        iaa.PerspectiveTransform(scale=(0.06, 0.1),
                                 keep_size=False,
                                 fit_output=True,
                                 cval=(0, 255),
                                 mode=ia.ALL),
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
            sometimes(
                iaa.CropAndPad(percent=(-0.05, 0.075),
                               pad_mode=ia.ALL,
                               pad_cval=(0, 255))),
            sometimes(
                iaa.Affine(
                    scale={
                        'x': (0.9, 1.1),
                        'y': (0.9, 1.1)
                    },
                    translate_percent={
                        'x': (-0.03, 0.03),
                        'y': (-0.03, 0.03)
                    },
                    rotate=(-5, 5),  # rotate by -45 to +45 degrees
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL)),
            iaa.SomeOf(
                (0, 2),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 7)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0),
                                lightness=(0.75, 1.5)),  # sharpen images
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.015), per_channel=0.1),
                        iaa.CoarseDropout((0.01, 0.015),
                                          size_percent=(0.01, 0.015),
                                          per_channel=0.1),
                    ]),
                    iaa.Add((-10, 10), per_channel=0.5),
                ]),
            iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=False)
        ],
        random_order=True)
