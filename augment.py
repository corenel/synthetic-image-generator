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
        description='augment data for object detection algorithms.')
    parser.add_argument("--input",
                        type=str,
                        default="data/backgrounds/",
                        help="Path to background images folder.")
    parser.add_argument("--output",
                        type=str,
                        default="data/results/",
                        help="Path to output images folder.")
    return parser.parse_args()


def main():
    # parse arguments
    opt = parse_args()

    # read input file lists
    dataset_background = dataset.BackgroundImageFolderDataset(opt.input)
    print('#images: {}'.format(len(dataset_background)))

    # create output directory if not exists
    if not os.path.exists(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    aug_image = util.build_augment_sequence_for_background()

    # start generating synthetic images
    for epoch_idx in range(setting.NUM_EPOCHS):
        # do generation
        for batch_idx in tqdm(range(setting.BATCH_SIZE),
                              desc='Epoch {}'.format(epoch_idx)):
            # load the background image
            background_idx = np.random.randint(low=0,
                                               high=len(dataset_background),
                                               size=1)[0]
            image_background, bboxes_background = dataset_background[
                background_idx]

            # get image and bboxes of background
            image_result = image_background.copy()

            # prepare initial annotations
            bbox_list = [bbox for bbox in bboxes_background.bounding_boxes]

            # augment image and bboxes of image
            annotations = BoundingBoxesOnImage(bounding_boxes=bbox_list,
                                               shape=(image_result.height,
                                                      image_result.width))
            image_result_aug, annotations_aug = aug_image(
                image=np.array(image_result), bounding_boxes=annotations)
            image_result_aug = Image.fromarray(image_result_aug)

            # save result
            output_path_image = os.path.join(
                opt.output, '{:03d}_{:05d}.png'.format(epoch_idx, batch_idx))
            image_result_aug.save(fp=output_path_image, format="png")
            # save annotation data
            output_path_anno = os.path.join(
                opt.output, '{:03d}_{:05d}.txt'.format(epoch_idx, batch_idx))
            util.write_yolo_annotations(outpath=output_path_anno,
                                        annotations=annotations_aug,
                                        image_width=image_result_aug.width,
                                        image_height=image_result_aug.height)

    print('\nDone!')


if __name__ == '__main__':
    main()