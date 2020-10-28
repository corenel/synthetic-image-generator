import argparse
import os

import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from tqdm import tqdm

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


def main():
    # parse arguments
    opt = parse_args()

    # read input file lists
    dataset_background = dataset.BackgroundImageFolderDataset(opt.backgrounds)
    dataset_object = dataset.ObjectImageFolderDataset(opt.objects)
    # create output directory if not exists
    if not os.path.exists(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    # initialize augmentation functions
    aug_object = util.build_augment_sequence_for_object()
    aug_image = util.build_augment_sequence_for_background()

    # start generating synthetic images
    count = 0
    pbar = tqdm(total=setting.NUM_EPOCHS * setting.BATCH_SIZE)
    # for image_file_background in image_files_background:
    for epoch_idx in range(setting.NUM_EPOCHS):
        # load the background image
        background_idx = np.random.randint(low=0,
                                           high=len(dataset_background),
                                           size=1)[0]
        image_background, bboxes_background = dataset_background[
            background_idx]

        # group 2-4 objects together on a single background
        object_groups = [
            list(
                np.random.randint(low=0,
                                  high=len(dataset_object) - 1,
                                  size=np.random.randint(2, 5, 1)))
            for _ in range(setting.BATCH_SIZE)
        ]

        for object_group in object_groups:
            # get image and bboxes of background
            image_result = image_background.copy()

            # prepare initial annotations
            bbox_list = [bbox for bbox in bboxes_background.bounding_boxes]

            # get sizes and positions
            objs, labels, obj_sizes, boxes = util.get_group_object_positions(
                object_group, image_background, dataset_object)

            # process each objext in the group
            for i, obj, label, size, box in zip(object_group, objs, labels,
                                                obj_sizes, boxes):
                # resize the object
                obj_w, obj_h = size
                obj = obj.resize((obj_w, obj_h))
                obj = util.resize_image(obj)
                # augment this object
                obj_aug = Image.fromarray(
                    aug_object.augment_images([np.array(obj)])[0])
                # generate annotation for this object
                obj_w, obj_h = obj_aug.size
                x_pos, y_pos = box[:2]
                x = int(x_pos + (0.5 * obj_w))
                y = int(y_pos + (0.5 * obj_h))
                bbox_list.append(util.xywh_to_bbox(label, x, y, obj_w, obj_h))
                # paste the obj to the background
                image_result.paste(obj_aug, (x_pos, y_pos))

            # augment image and bboxes of image
            annotations = BoundingBoxesOnImage(bounding_boxes=bbox_list,
                                               shape=(image_result.height,
                                                      image_result.width))
            image_result_aug, annotations_aug = aug_image(
                image=np.array(image_result), bounding_boxes=annotations)
            image_result_aug = Image.fromarray(image_result_aug)

            # save result
            output_path_image = os.path.join(opt.output,
                                             '{:05d}.png'.format(count))
            image_result_aug.save(fp=output_path_image, format="png")
            # save annotation data
            output_path_anno = os.path.join(opt.output,
                                            '{:05d}.txt'.format(count))
            util.write_yolo_annotations(outpath=output_path_anno,
                                        annotations=annotations_aug,
                                        image_width=image_result_aug.width,
                                        image_height=image_result_aug.height)
            count += 1
            pbar.update(1)
    print('\nDone!')
    pbar.close()


if __name__ == '__main__':
    main()
