"""
Load, preprocess and yield data to models.

******************************************
"""
# coding: utf8
from skimage.util import view_as_windows
import numpy as np
from skimage.io import imread


def getNextFilePath(output_folder, base_name):
    """
    Get highest indice of all files with name base_name in output_folder.

    output_folder: folder in which to look for files
    base_name: base of file names
    return: next available index
    """
    highest_num = 0
    for f in output_folder.iterdir():
        if f.is_file():
            try:
                f = str(f.with_suffix('').name)
                if f.split('_')[:-1] == base_name.split('_'):
                    split = f.split('_')
                    file_num = int(split[-1])
                    if file_num > highest_num:
                        highest_num = file_num
            except ValueError:
                'The file name "%s" is incorrect. Skipping' % f

    output_file = highest_num + 1
    return output_file


def patchify(image, size):
    """
    Turn image into patches of size.

    ********************************
    """
    pad = size // 2
    arr = view_as_windows(np.pad(image,
                                 ((pad, pad), (pad, pad), (0, 0)),
                                 mode='reflect'),
                          (size, size, 3))
    return np.reshape(arr, (-1, size, size, 3))


def _check_include(obj, include):
    return include is None or obj.name in include


def _check_exclude(obj, exclude):
    return exclude is None or obj.name not in exclude


def _check_valid(obj, include, exclude):
    return (_check_include(obj, include)
            and _check_exclude(obj, exclude)
            and not obj.name.startswith('.'))


def get_items(folder,
              recurse=True,
              extensions=None,
              include=None,
              exclude=None):
    """
    Get files with given suffix in folder and sub-folders.

    ******************************************************
    """
    items = []
    for obj in folder.iterdir():
        if obj.is_file():
            if extensions is None or obj.suffix in extensions:
                items.append(obj)
        elif recurse and _check_valid(obj, include, exclude):
            items_r = get_items(obj, extensions=extensions)
            items += items_r
    return items


class DataGenerator(object):
    """
    DataGenerator.

    **************
    """

    def __init__(self, itemlist, batch_size, percent_kept, patch_size):
        """
        Init.

        ****
        """
        self.files = itemlist
        self.batch_size = batch_size
        self.ratio = percent_kept
        self.patch_size = patch_size
        self.fileidx = 0
        self.batchidx = 0
        self.load_image()

    def __next__(self):
        """
        Next patch.

        ***********
        """
        if self.batchidx * self.batch_size >= self.ratio * self.image.shape[0]:
            if self.fileidx < len(self.files) - 1:
                self.fileidx += 1
            else:
                self.fileidx = 0
            self.load_image()
            self.batchidx = 0
        batch = self.image[self.batchidx *
                           self.batch_size: (self.batchidx + 1) *
                           self.batch_size]
        self.batchidx += 1
        return batch, batch

    def load_image(self):
        """
        Load image.

        ***********
        """
        self.image = patchify(
            imread(self.files[self.fileidx]),
            self.patch_size).astype(np.float32) / 255
        np.random.shuffle(self.image)

    def __iter__(self):
        """
        Iter.

        *****
        """
        return self

    def __len__(self):
        """
        Length.

        *******
        """
        return len(self.files) // self.batch_size


def get_generators(itemlist,
                   batch_size,
                   percent_kept,
                   patch_size,
                   train_ratio=0.75):
    """
    Get train and test generators.

    ******************************
    """
    np.random.shuffle(itemlist)
    end_train = int(train_ratio * len(itemlist))
    trainlist = itemlist[0:end_train]
    validlist = itemlist[end_train::]
    return (DataGenerator(trainlist, batch_size, percent_kept, patch_size),
            DataGenerator(validlist, batch_size, percent_kept, patch_size))
