#!/usr/bin/env python
# -*- coding:utf-8 -*-
import gzip
import pickle

import numpy as np

import idx3_ubyte_2_images as iu

pkl_path = "../data/idx_ubyte/pkl/"


def get_full_path(path):
    """
    获取完整路径
    """
    return pkl_path + path


def save_pkl():
    """
    读取idx_ubyte格式的mnist训练集，转为pkl格式
    """
    train_images, train_labels, test_images, test_labels = iu.read_all()
    validate_images = train_images[50000:]
    validate_labels = train_labels[50000:]
    data = ((train_images[:50000], np.array(train_labels[:50000])),
            (validate_images, np.array(validate_labels)),
            (test_images, np.array(test_labels)))

    with open(get_full_path('mnist.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print "done"


def compress_pickle_file(input_file, output_file):
    """
    将pkl文件进行gzip压缩
    """
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.write(f_in.read())


if __name__ == '__main__':
    # save_pkl()
    # 压缩pkl文件
    compress_pickle_file(get_full_path('mnist.pkl'), get_full_path('mnist.pkl.gz'))
