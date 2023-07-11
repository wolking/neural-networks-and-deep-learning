# coding=utf-8
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        """
        sizes表示神经网络的层数，比如sizes=[1,2,3]表示三层神经网络，第一层有1个神经元，第二层有2个神经元，以此类推，
            一般来说,第一层是输入层，最后一层是输出层，中间是隐藏层
        """
        self.sizes = sizes
        """
        np.random.randn(n1,n2,n3,...):生成服从标准正态分布（均值为0，方差为1）的随机数，随机数的范围(负无穷, 正无穷)
        在实际应用中，生成的随机数通常会在一定的范围内。标准正态分布的均值为0，方差为1。
        因此，生成的随机数的平均值在接近0的位置，而大部分随机数的取值会在接近0附近，并随着离0的距离逐渐减少。
        请注意，np.random.randn()生成的随机数是连续的实数值，而不是离散的值
        
        函数参数个数为0：生成一个浮点数，np.random.randn() = -0.5156079319849818
        函数参数个数为1，数值为m：生成一个一维数组，有m个浮点元素, np.random.randn(1) = [ 0.37017203]
        函数参数个数为m，数值为n1,n2,n3,...：生成一个m维数组，每一维的元素个数为n1,n2,n3,...
                    np.random.randn(2,3) = [[ 0.40235067,0.11849399,0.46751975],[-0.2267899,0.5508104,-1.85147667]]
        """

        """
        sizes[1:]：一个len(sizes)-1个的数组
        biases是一个len(sizes)-1行1列的二维数组，表示每一层神经元的偏置集合
            比如sizes=[1,2,3],表示神经网络有三层,则
            y1 = np.random.randn(2, 1) = [[0.2394872834],[0.1298374289]]
            y2 = np.random.randn(3, 1) = [[0.9385093535],[0.9848957834],[0.9823482472]]
            biases=[y1,y2]，
            第一层使用两个偏置biases[0] = y1,得到的两个输出作为第二层的输入，
            第二层使用三个偏置biases[1] = y2,得到的三个输出作为这个神经网络的最终输出
        因为最后一层不需要用到偏置，设第二层神经元个数为A1，第三层为A2，......，第n层为An，
        偏置总数=A1+A2+A3+...+An
        """
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        """
        数组或list的负索引表示从其末尾开始算起，-1表示最后一个元素，[:-1]表示0到倒数第二个元素组成的数组或list
        zip(*iterables)：Python内置的函数，它用于将多个可迭代对象的元素按索引进行配对，返回一个由元组组成的迭代器。
            每个元组包含来自输入可迭代对象的对应位置的元素。iterables可以是列表、元组、字符串或其他可迭代类型.
            return:返回一个list，元素是一个元组
            基本用法：
            a1 = [1, 2, 3]
            a2 = ["a", "b", "c"]
            zip_r = zip(a1, a2)
            zip_r = [(1, 'a'), (2, 'b'), (3, 'c')]
            当配对的可迭代对象的长度不一致时，zip()函数会以最短的可迭代对象为准，忽略长度超过最短对象的部分。
            a1 = [1, 2,]
            a2 = ["a", "b", "c"]
            zip_r = zip(a1, a2)
            zip_r = [(1, 'a'), (2, 'b')]
        """
        """
        zip(sizes[:-1], sizes[1:])：将神经网络的相邻的两层各自的神经元个数进行两两配对
        np.random.randn(y, x)：此处将y放在第一位是因为y是下一层的神经元个数，当前层需要执行y次损失函数后输出y个输出才能构成下一层
                                
        所以此处的代码是为了初始化各层各个神经元所需要的权重值
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        """
        SGD：梯度下降算法
        training_data：训练数据
        epochs：训练轮数
        mini_batch_size：批次数
        eta：学习率
        test_data：测试数据
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        """
        xrange() 函数用于生成一个指定范围内的整数迭代器,接受起始值、结束值和步长作为参数，并按照指定的步长生成整数序列
                # 生成从0到9的整数序列 for i in xrange(10): print(i)
                # 生成从5到19的奇数序列 for i in xrange(5, 20, 2): print(i)
                注意：该函数只存在2.x版本，（Python 3.x）中已经将 xrange() 函数移除，并将其替换为 range() 函数
        """
        for j in xrange(epochs):
            """
            random.shuffle() 是Python的一个随机模块（random module）中的函数，用于随机打乱（洗牌）可变序列对象的顺序
            此处将training_data数组中的元素进行顺序打乱，打乱的原因如下：
                在深度学习神经网络中，将训练数据随机打乱是一种常见的预处理步骤，其目的是提高训练的效果和泛化能力。以下是几个主要原因：

                1.避免模型对顺序的依赖性：深度神经网络的训练通常使用基于梯度的优化算法，例如随机梯度下降（SGD）。
                    在这些算法中，通过反向传播计算梯度，并根据梯度来更新网络参数。如果训练数据按照固定的顺序呈现给模型，
                    模型可能会对顺序产生依赖，从而导致学习到的模型对于某些特定的顺序表现优秀，但在其他顺序下表现差。
                    通过将训练数据随机打乱，可以破坏顺序的依赖性，使得模型不依赖于特定的数据顺序。
                2.提高泛化能力：深度神经网络的目标是学习输入数据的潜在模式和特征，以便对未知数据进行准确的预测。
                    如果训练数据按照固定的顺序呈现给模型，模型可能会对于特定的顺序学习到过度依赖的模式，导致模型泛化能力下降。
                    通过随机打乱训练数据，可以引入更多的变化和多样性，使得模型能够更好地适应不同的数据分布和未知样本。
                3.确保样本的独立性：如果训练数据集中的样本具有某种固定的排序或分布，例如按类别分组或按某种特征排序，
                    那么在模型训练过程中，网络的不同层可能在不同时期接触到相似的样本。这可能会导致模型对某些样本的过拟合，
                    而在处理不同类别或特征的样本时表现不佳。通过随机打乱数据，可以确保每个训练样本都有机会影响模型的训练过程，
                    并减少过拟合的风险。

                综上所述，将训练数据随机打乱有助于减少模型对顺序的依赖性，提高泛化能力，并确保每个样本对模型的训练都有平等的机会。这样可以促进模型学习到更广泛、更准确的特征和模式，从而改善模型的性能。
            """
            random.shuffle(training_data)
            # 将训练数据切割成小批次
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # 进行小批次训练，更新权重和偏置
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                """
                此处使用未被定义的n_test是被允许的，在python中控制块的块级作用域不存在，
                对于像for,if,while,with等这样的控制块，在其中声明的变量作用域能够延伸到当前控制块之外。
                但有一个问题，一旦该控制块的变量没被执行，在块外使用该变量时会报NameError错误
                比如以下代码就会报错：
                    if False:
                        n_test = 1
                    print(n_test)
                
                这里有对test_data进行判真，结合上下文，n_test是有被初始化的
                """
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """小批量更新权重和偏置 Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        """
        np.zeros()：用于创建指定形状和数据类型的全零数组，
                    np.zeros(shape, dtype=float, order='C')
                    参数解释：
                        shape：数组的形状，可以是一个整数或一个整数元组，用于指定数组的维度。
                        dtype（可选）：数据类型，表示创建的数组的元素类型，默认为 float。
                        order（可选）：数组在内存中的布局顺序。默认为 'C'，表示按行展开（行优先）。
                                    可选值有 'C' 和 'F'，分别表示行优先和列优先。
                    arr = np.zeros((3, 4))
                    print(arr)
                    # 输出：
                    # [[0. 0. 0. 0.]
                    #  [0. 0. 0. 0.]
                    #  [0. 0. 0. 0.]]
        """
        # 初始化一个与偏置数组结构相同的全零数组
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # 初始化一个与权重数组结构相同的全零数组
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # x是像素集，y是结果集
            # 反向传播算法，用于快速计算代价函数的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 下批次的
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward 前向传播：将输入样本通过神经网络进行前向传播，计算每一层的输出和激活值
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            """
            for循环解析：
            我们假设该神经网络的有三层[3,2,1],那么weights应该为[[2行3列],[1行2列]]，biases为[[2行1列],[1行1列]]，将biases与weights进行配对
            得到(3层-1)行1列的数组 [ ([2行1列],[2行3列]), ([1行1列],[1行2列]) ]，其中列为元组，该元组是zip配对后的结果
            结构为 [(b1, w1), (b2, w2), ....]
            进行for循环遍历，元组第一个为偏置b的矩阵，元组第二个为权重w的矩阵

            更简明的示例请看：demo.backprop_first_for()
            """
            """
            np.dot()：计算两个数组的点积（内积）,相当于数学里的矩阵相乘
                a = np.array([[1, 2],[3, 4]])
                b = np.array([[5, 6],[7, 8]])

                c = np.dot(a, b)
                print(c)
                # 输出：
                # [[19 22]
                #  [43 50]]
            """
            # z = w*a + b，a代表上一层的输出，当前层是第一层是，a表示训练数据，w/a/b都是矩阵
            z = np.dot(w, activation) + b
            zs.append(z)
            # 计算下一个激活量，也就是下一层的输入，当前层的输出
            activation = sigmoid(z)
            # 一层一层存储激活值
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        """
        ndarray.transpose():交换数组的维度，也就是矩阵转置，行变列，列变行
                            比如：array = [[1,2,3],[4,5,6]], array.transpose()就等于[[1,4],[2,5],[3,6]]
        """
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        """
        np.argmax()：用于返回数组中最大值的索引。
                比如：a = ndarray(10, 30, 40, 20),则np.argmax(a)=4
        通过feedforward函数以及经过梯度下降算法更新后的权重和偏置，可预测出图像x对应的数字，
        feedforward()返回一个二维数组y，假设图像对应的数字为i，则y[i] = [1.0]，而其他元素均为[0.],通过argmax得到[1.0]的索引，
        该索引是图像对应数字的预测值
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """对sigmoid函数的求偏导数，Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
