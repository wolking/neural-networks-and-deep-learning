#! user/bin/python
# - * -coding:UTF-8 - * -

import network
import mnist_loader
import pickle
import gzip
import numpy as np


def open_pkl():
    with gzip.open('../data/mnist.pkl.gz', 'rb') as file:
        data = pickle.load(file)
    print(data)


def train():
    """
    加载训练数据文件[../data/mnist.pkl.gz]，返回三个ndarray对象，
    training_data是一个只有两个元素的元组
        元组的第一个元素是一个二维数组A1，是真实的图像数据，有5w个元素，一个元素代表一张图像，每个元素是拥有28*28=784个元素的数组
        元组的第二个元素是一个一维数组A2，有5w个元素，每一个元素都是A1每一个数组
        training_data=tuple(array([[0.,0.,...,784个],....5w个]),
                            array())
    """
    # 读取训练集
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # 初始化神经网络
    net = network.Network([784, 30, 10])
    # 开始训练
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def randn():
    """
    生成0-1的正态分布
    """
    a1 = np.random.randn()
    a2 = np.random.randn(1)
    a3 = np.random.randn(2, 3)
    print(a1)
    print(a2)
    print(a3)


def zip_example():
    a1 = [1, 2, 3]
    a2 = ["a", "b", "c"]
    zip_r = zip(a1, a2)
    print(zip_r)
    for x, y in zip([1,2,3],[4,5,6]):
        print(x, y)


if __name__ == "__main__":
    # open_pkl()
    # randn()
    # zip_example()
    train()


