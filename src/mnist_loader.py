# coding=utf-8
"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    # tr_d: 是训练集，一个元组，第一个元素是5w个像素集，第二个元素是每一个像素集对应的数字
    # va_d与te_d: 都是验证集，各1w个
    tr_d, va_d, te_d = load_data()
    """
    改变像素集的形状，由原来的二维数组[[784个像素点],5w个图像]，变为[[[1个像素点], 784个像素点集],5w个图像],记为TI_A
    像素点从原来的数值变为只有一个元素的数组。
    """
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    """
    vectorized_result(j)：创建一个二维数组A，总共10个元素[[0], 10个]，每一个元素都是只有一个元素且为0的数组，
                        在第j个元素时将0置为1.0，这里将数字转成一个二维数组且数字所在的元素为1，
                        就是为了验证经过神经网络训练后的结果集，落在该位置的数值是否为1，或近似于1，与1的差越小，说明预测的结果越准确
                        j为784个像素的图像代表的数字，范围在[0,9]之间
    training_results：训练结果集，一个5w个元素的二维数组,记为TR_A，每一个元素都为A
    """
    training_results = [vectorized_result(y) for y in tr_d[1]]
    """
    将变型的像素集TI_A与训练结果集TR_A进行配对，结果为：[([[1个像素点], 784个像素点集], [[0], 10个]), 5w个]
    就是将784个像素点组成的图像和对应的数字组成一对元组
    """
    training_data = zip(training_inputs, training_results)
    # 下面的操作和training_data一样，对像素集变形，然后与结果集进行配对
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
