#! user/bin/python
# - * -coding:UTF-8 - * -

import network
import mnist_loader
import pickle
import gzip
import numpy as np


def open_pkl():
    """
    加载训练数据文件[../data/mnist.pkl.gz]，返回三个ndarray对象，
    training_data是一个只有两个元素的元组
        元组的第一个元素是一个二维数组A1，是真实的图像数据，有5w个元素，一个元素代表一张图像，
                每个元素是拥有28*28=784个元素的数组，这些都是像素
        元组的第二个元素是一个一维数组A2，有5w个元素，每一个元素都是A1每一个像素数组，即图像对应的数字
        training_data=tuple(array([[0.,0.,...,784个],....5w个]),
                            array())
    """
    with gzip.open('../data/idx_ubyte/pkl/mnist.pkl.gz', 'rb') as file:
        data = pickle.load(file)
    print(data)


def train():
    # 读取训练集
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # 初始化神经网络
    net = network.Network([784, 50, 30, 10])
    # 开始训练，使用梯度下降算法
    net.SGD(training_data, 5, 10, 3.0, test_data=test_data)


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


def np_zeros():
    # 全零数组
    arr = np.zeros((3, 4), dtype=int, order='F')
    print(arr)


def backprop_first_for():
    # network.backprop第一个for循环的示例
    wt = [[[1, 2, 3], [4, 5, 6]], [[7], [8]]]
    bs = [[[9], [10]], [11]]
    # zip_r = [([[9], [10]], [[1, 2, 3], [4, 5, 6]]),
    #           ([11], [[7], [8]])]
    zip_r = zip(bs, wt)
    for b, w in zip_r:
        print(b)
        print(w)
        print("----------")


if __name__ == "__main__":
    # open_pkl()
    # randn()
    # zip_example()
    train()
    # np_zeros()
    # backprop_first_for()
    # a = [1,2,3]
    # print(a[:-1])


