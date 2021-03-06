import argparse
import os

import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import dataset
import setting
import util


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Create synthetic training data for object detection algorithms.')
    parser.add_argument("--backgrounds",
                        type=str,
                        default="data/backgrounds/",
                        help="Path to background images folder.")
    parser.add_argument("--objects",
                        type=str,
                        default="data/objects/",
                        help="Path to object images folder.")
    parser.add_argument("--output",
                        type=str,
                        default="data/results/",
                        help="Path to output images folder.")
    return parser.parse_args()


def generate(batch_idx, epoch_idx, out_dir, object_groups, dataset_background,
             dataset_object, aug_object, aug_image):
    # load the background image
    background_idx = np.random.randint(low=0,
                                       high=len(dataset_background),
                                       size=1)[0]
    image_background, bboxes_background = dataset_background[background_idx]

    # get image and bboxes of background
    image_result = image_background.copy()

    # prepare initial annotations
    bbox_list = [bbox for bbox in bboxes_background.bounding_boxes]

    # get sizes and positions
    object_group = object_groups[batch_idx]
    objs, labels, obj_sizes, boxes = util.get_group_object_positions(
        object_group, image_background, dataset_object, aug_object)

    # process each objext in the group
    for i, obj, label, size, box in zip(object_group, objs, labels, obj_sizes,
                                        boxes):
        # generate annotation for this object
        obj_w, obj_h = obj.size
        x_pos, y_pos = box[:2]
        x = int(x_pos + (0.5 * obj_w))
        y = int(y_pos + (0.5 * obj_h))
        bbox_list.append(util.xywh_to_bbox(label, x, y, obj_w, obj_h))
        # paste the obj to the background
        image_result.paste(obj, (x_pos, y_pos))

    # augment image and bboxes of image
    annotations = BoundingBoxesOnImage(bounding_boxes=bbox_list,
                                       shape=(image_result.height,
                                              image_result.width))
    image_result_aug, annotations_aug = aug_image(image=np.array(image_result),
                                                  bounding_boxes=annotations)
    image_result_aug = Image.fromarray(image_result_aug)

    # save result
    output_path_image = os.path.join(
        out_dir, '{:03d}_{:05d}.jpg'.format(epoch_idx, batch_idx))
    image_result_aug.save(fp=output_path_image, format="jpeg")
    # save annotation data
    output_path_anno = os.path.join(
        out_dir, '{:03d}_{:05d}.txt'.format(epoch_idx, batch_idx))
    util.write_yolo_annotations(outpath=output_path_anno,
                                annotations=annotations_aug,
                                image_width=image_result_aug.width,
                                image_height=image_result_aug.height)


def main():
    # parse arguments
    opt = parse_args()

    # read input file lists
    dataset_background = dataset.BackgroundImageFolderDataset(opt.backgrounds)
    print(dataset_background)
    dataset_object = dataset.ObjectImageFolderDataset(opt.objects)
    print(dataset_object)
    # create output directory if not exists
    if not os.path.exists(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    # initialize augmentation functions
    aug_object = util.build_augment_sequence_for_object()
    aug_image = util.build_augment_sequence_for_background()

    # start generating synthetic images
    for epoch_idx in range(setting.NUM_EPOCHS):
        # group 2-4 objects together on a single background
        object_groups = [
            list(
                np.random.randint(low=0,
                                  high=len(dataset_object) - 1,
                                  size=np.random.randint(5, 7, 1)))
            for _ in range(setting.BATCH_SIZE)
        ]
        # func wrapper for multiprocessing
        gen = partial(generate,
                      epoch_idx=epoch_idx,
                      out_dir=opt.output,
                      object_groups=object_groups,
                      dataset_background=dataset_background,
                      dataset_object=dataset_object,
                      aug_object=aug_object,
                      aug_image=aug_image)
        # do generation
        for batch_idx in tqdm(range(setting.BATCH_SIZE),
                              desc='Epoch {}'.format(epoch_idx)):
            gen(batch_idx)
        # with mp.Pool(processes=max(mp.cpu_count() - 1, 4)) as pool:
        #     pool.map(gen, range(setting.BATCH_SIZE))
        # with tqdm(total=setting.BATCH_SIZE,
        #           desc='Epoch {}'.format(epoch_idx)) as pbar:
        #     with mp.Pool(processes=max(mp.cpu_count() - 1, 4)) as pool:
        #         for _ in pool.imap_unordered(gen, range(setting.BATCH_SIZE)):
        #             pbar.update(1)
    print('\nDone!')


if __name__ == '__main__':
    main()
