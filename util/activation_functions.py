import numpy as np


"""基本的激活函数"""


def linear(x, constant=1):
    return constant*x


def d_linear(constant=1):
    return constant


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    try:
        return 1 - np.power(np.tanh(x), 2)
    except:
        new_x = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                new_x[i, j] = 1 - np.power(np.tanh(x[i,j]), 2)
        return new_x


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    try:
        return 1 if x > 0 else 0
    except:
        new_x = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                new_x[i, j] = 1 if x[i,j] > 0 else 0
        return new_x


def leaky_relu(x, factor=0.01):
    return np.maximum(factor*x, x)


def dleaky_relu(x, factor=0.01):
    try:
        return 1 if x > 0 else factor
    except:
         new_x = np.zeros((x.shape[0], x.shape[1]))
         for i in range(x.shape[0]):
             for j in range(x.shape[1]):
                 new_x[i, j] = 1 if x[i,j] > 0 else factor
         return new_x


def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(np.exp(x))
    res = exp/exp_sum
    return res


def gelu(x):
    res = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    return res


def dsoftmax(x):
    s = softmax(x)
    s = s.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)
