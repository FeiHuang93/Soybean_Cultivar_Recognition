import tempfile
import os
import random

import numpy as np

from .load_dataset import load_by_image_file, load_by_npz_file
from .base_provider import ImagesDataSet, DataProvider


def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    flip = random.getrandbits(1)   # 随机生成0或1,以此作为是否镜面操作的依据
    if flip:
        image = image[:, ::-1, :]  # 水平镜面处理 
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
              init_x: init_x + init_shape[0],
              init_y: init_y + init_shape[1],
              :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad)
    return new_images


class CustomDataSet(ImagesDataSet):
    def __init__(self, images, labels, n_classes, shuffle, normalization,
                 augmentation):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            n_classes: `int`, number of cifar classes - 10 or 100
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        if self.augmentation:
            images = augment_all_images(images, pad=4)
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class CustomDataProvider(DataProvider):
    """Abstract class for cifar readers"""
    # _data_url = "../data_jpg256"
    _n_classes = 100
    _data_shape = (224, 224, 3)
    # image, npz
    _file_type = "image"
    data_augmentation = False

    def __init__(self, data_url=None, save_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        self._data_url = data_url
        self._save_path = save_path
        self.one_hot = one_hot
        # download_data_url(self.data_url, self.save_path)
        # train_fnames, test_fnames = self.get_filenames(self.save_path)

        # add train and validations datasets
        # images, labels = self.read_cifar(train_fnames)
        if self._file_type == "image":
            images, labels = load_by_image_file(os.path.join(self.data_url, "train.txt"),
                                                (self.data_shape[0], self.data_shape[1]))
        elif self._file_type == "npz":
            images, labels = load_by_npz_file(os.path.join(self.data_url, "train.txt"))
        else:
            print("unknown file type -> " + self._file_type)
            exit(0)

        labels[labels == self.n_classes] = 0
        if one_hot:
            labels = self.labels_to_one_hot(labels)
        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = CustomDataSet(
                images=images[:split_idx], labels=labels[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = CustomDataSet(
                images=images[split_idx:], labels=labels[split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
        else:
            self.train = CustomDataSet(
                images=images, labels=labels,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add test set
        if self._file_type == "image":
            images, labels = load_by_image_file(os.path.join(self.data_url, "test.txt"),
                                                (self.data_shape[0], self.data_shape[1]))
        elif self._file_type == "npz":
            images, labels = load_by_npz_file(os.path.join(self.data_url, "test.txt"))
        labels[labels == self.n_classes] = 0
        if one_hot:
            labels = self.labels_to_one_hot(labels)
        self.test = CustomDataSet(
            images=images, labels=labels, shuffle=None,
            n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(
                tempfile.gettempdir(), 'leaf%d' % self.n_classes)
        return self._save_path

    @property
    def data_url(self):
        return self._data_url

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def n_classes(self):
        return self._n_classes


class CustomAugmentedDataProvider(CustomDataProvider):
    data_augmentation = True


if __name__ == '__main__':
    pass
