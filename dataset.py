import os

from PIL import Image

import setting
from util import read_yolo_annotations

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory: str, class_to_idx):
    instances = []
    directory = os.path.expanduser(directory)

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def pil_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Dataset(object):
    _repr_indent = 4

    def __init__(self, root):
        root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class ImageFolderDataset(Dataset):
    def __init__(self, root):
        super(ImageFolderDataset, self).__init__(root)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(
                ",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)

        self.loader = pil_loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir_path):
        classes = [d.name for d in os.scandir(dir_path) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class BackgroundImageFolderDataset(ImageFolderDataset):
    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        anno_path = os.path.splitext(image_path)[0] + ".txt"
        image = self.loader(image_path)
        anno = read_yolo_annotations(anno_path,
                                     image_width=image.size[0],
                                     image_height=image.size[1])
        return image, anno


class ObjectImageFolderDataset(ImageFolderDataset):
    def _find_classes(sefl, dir_path):
        classes, class_to_idx = super()._find_classes(dir_path)
        class_to_idx = {
            cls_name: setting.LABELS.index(cls_name)
            for cls_name in classes
        }
        return classes, class_to_idx
