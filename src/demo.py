#! user/bin/python
# - * -coding:UTF-8 - * -

import network
import mnist_loader
import pickle
import gzip


def open_pkl():
    with gzip.open('../data/mnist.pkl.gz', 'rb') as file:
        data = pickle.load(file)
    print(data)


def train():
    """
    加载训练数据文件[../data/mnist.pkl.gz]，返回三个ndarray对象，
    training_data是一个只有两个元素的元组
        元组的第一个元素是一个二维数组，是真实的图像数据，有5w个元素，一个元素代表一张图像，每个元素是拥有28*28=784个元素的数组
        元组的第二个元素是一个一维数组，有1w个元素，
        training_data=tuple(array([[0.,0.,...,784个],....5w个]),
                            array())
    """
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


if __name__ == "__main__":
    open_pkl()
