import numpy as np
import cv2


def load_by_npz_file(file):
    images = list()
    labels = list()
    with open(file) as t_file:
        all_lines = t_file.readlines()
        for line in all_lines:
            pos = line.strip().rfind(' ')
            npz_file = line[:pos]
            label = int(line[pos+1:])
            labels.append(label)
            f = np.load(npz_file)
            image = f.f.arr_0
            images.append(image)
            print("file -> " + npz_file)
            print("label -> " + str(label))
    return np.asarray(images), np.asarray(labels)


def load_by_image_file(file, size):
    images = list()
    labels = list()
    with open(file) as t_file:
        all_lines = t_file.readlines()
        for line in all_lines:
            pos = line.strip().rfind(' ')
            image_file = line[:pos]
            label = int(line[pos+1:])
            labels.append(label)
            image = cv2.imread(image_file)
            image = cv2.resize(image, size)
            images.append(image)
            print("file -> " + image_file)
            print("label -> " + str(label))
    return np.asarray(images), np.asarray(labels)

# train_images, train_labels, test_images, test_labels = load_dataset("train.txt", "test.txt")
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

