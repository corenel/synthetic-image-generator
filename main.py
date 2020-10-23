import argparse
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import util


def parse_args():
    parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
    parser.add_argument("--backgrounds", type=str, default="data/backgrounds/",
                        help="Path to background images folder.")
    parser.add_argument("--objects", type=str, default="data/objects/",
                        help="Path to object images folder.")
    parser.add_argument("--output", type=str, default="data/results/",
                        help="Path to output images folder.")
    return parser.parse_args()


def main():
    # parse arguments
    opt = parse_args()

    # read input file lists
    image_files_background = glob(os.path.join(opt.backgrounds, '*.png'))
    image_files_object = glob(os.path.join(opt.objects, '*.png'))
    # create output directory if not exists
    if not os.path.exists(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    # initialize augmentation functions
    aug_object = util.build_augment_sequence_for_object()
    aug_background = util.build_augment_sequence_for_background()

    # start generating synthetic images
    count = 0
    pbar = tqdm()
    for image_file_background in image_files_background:
        # load the background image
        image_background = Image.open(image_file_background)

        # group 2-4 objects together on a single background
        object_groups = [np.random.randint(low=0, high=len(image_files_object) - 1, size=np.random.randint(2, 5, 1)) for
                         _ in
                         range(2 * len(image_files_object))]

        for object_group in object_groups:

            # get sizes and positions
            obj_sizes, boxes = util.get_group_object_positions(object_group, image_background, image_files_object)
            # get image and bboxes of background
            bkg_w_obj = image_background.copy()
            bkg_boxes = util.read_yolo_annotations(image_file_background.replace('.png', '.txt'),
                                                   image_width=bkg_w_obj.size[0],
                                                   image_height=bkg_w_obj.size[1])
            # augment image and bboxes of background
            bkg_w_obj_aug, bkg_boxes_aug = aug_background(image=np.array(bkg_w_obj), bounding_boxes=bkg_boxes)
            bkg_w_obj_aug = Image.fromarray(bkg_w_obj_aug)
            image_background_width, image_background_height = bkg_w_obj_aug.size
            # prepare initial annotations
            group_annotation = []
            for bkg_box_aug in bkg_boxes_aug.remove_out_of_image().clip_out_of_image().bounding_boxes:
                group_annotation.append({
                    'coordinates': {
                        'height': bkg_box_aug.y2 - bkg_box_aug.y1,
                        'width': bkg_box_aug.x2 - bkg_box_aug.x1,
                        'x': int((bkg_box_aug.x2 + bkg_box_aug.x1) / 2),
                        'y': int((bkg_box_aug.y2 + bkg_box_aug.y1) / 2)
                    },
                    'label': bkg_box_aug.label
                })

            # process each objext in the group
            for i, size, box in zip(object_group, obj_sizes, boxes):
                # read the object image
                obj = Image.open(image_files_object[i])
                obj_w, obj_h = size
                obj = obj.resize((obj_w, obj_h))
                # augment this object
                obj_aug = Image.fromarray(aug_object.augment_images([np.array(obj)])[0])
                # generate annotation for this object
                obj_w, obj_h = obj_aug.size
                x_pos, y_pos = box[:2]
                annotation = {
                    'coordinates': {
                        'height': obj_h,
                        'width': obj_w,
                        'x': int(x_pos + (0.5 * obj_w)),
                        'y': int(y_pos + (0.5 * obj_h))
                    },
                    'label': util.get_label_id(image_files_object[i])
                }
                group_annotation.append(annotation)
                # paste the obj to the background
                bkg_w_obj_aug.paste(obj_aug, (x_pos, y_pos))

            # save result
            output_fp = os.path.join(opt.output, '{:05d}.png'.format(count))
            bkg_w_obj_aug.save(fp=output_fp, format="png")
            # save annotation data
            util.write_yolo_annotations(outpath=output_fp.replace('.png', '.txt'), annotations=group_annotation,
                                        image_width=image_background_width,
                                        image_height=image_background_height)
            count += 1
            pbar.update(1)
    print('\nDone!')
    pbar.close()


if __name__ == '__main__':
    main()
