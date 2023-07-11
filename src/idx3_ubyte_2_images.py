#!/usr/bin/env python
# -*- coding:utf-8 -*-

import struct
import numpy as np
from PIL import Image

idx3_file_path = "../data/idx_ubyte/"


def get_full_path(path):
    """
    获取完整路径
    """
    return idx3_file_path + path


def read_idx3_file(filename):
    """
    读取idx3_ubyte文件
    """
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        images = np.zeros((num_images, num_rows, num_cols), dtype=np.uint8)

        for i in range(num_images):
            for row in range(num_rows):
                for col in range(num_cols):
                    pixel_value = struct.unpack('>B', f.read(1))[0]
                    images[i, row, col] = pixel_value

        return images


def read_idx1_file(filename):
    with open(filename, 'rb') as f:
        # 读取魔数和标签数量
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        # 打印信息
        print("Magic Number: {0}".format(magic_number))
        print("Number of Labels: {0}".format(num_labels))

        # 读取标签数据
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('>B', f.read(1))[0]
            labels.append(label)
        return labels


def read_all():
    """
    读取所有手写数字集idx3文件
    """
    train_images = read_idx3_file(idx3_file_path + 'train-images.idx3-ubyte')
    train_labels = read_idx1_file(idx3_file_path + 'train-labels.idx1-ubyte')
    test_images = read_idx3_file(idx3_file_path + 't10k-images.idx3-ubyte')
    test_labels = read_idx1_file(idx3_file_path + 't10k-labels.idx1-ubyte')
    return train_images, train_labels, test_images, test_labels


def save_images():
    image_path = "../data/idx_ubyte/images/"
    train_images = read_idx3_file(idx3_file_path + 'train-images.idx3-ubyte')
    for i in range(len(train_images)):
        if i == 1:
            break
        img = train_images[i]
        # img = img.astype(np.uint8)
        image = Image.fromarray(img)
        image.save(image_path + 'train_{}.png'.format(i))


def normalize_data():
    """
    将像素值进行归一化，归一化成[0,1]范围的二维数组，一维为图像个数，二维为归化后像素值
    """
    train_images = read_idx3_file(idx3_file_path + 'train-images.idx3-ubyte')
    images = np.zeros((60000, 784))
    for x in train_images:
        x = x.reshape(784, order='C')
        max_value = np.max(x)
        min_value = np.min(x)
        x = (x - min_value) / (max_value - min_value)
        np.append(images, x)
    print images


if __name__ == '__main__':
    # save_images()
    normalize_data()
    # train_labels = read_idx1_file(idx3_file_path + 'train-labels.idx1-ubyte')
    # print(train_labels)
    # decode_idx3_ubyte(get_full_path('train-images.idx3-ubyte'))
    # decode_idx1_ubyte(get_full_path('train-labels.idx1-ubyte'))

