import numpy as np

"""基本的损失函数"""


def mse(y, yhat):
    return np.mean(np.power(y-yhat, 2))


def dmse(y, yhat):
    return 2*(yhat-y)/y.size


def mae(y, yhat):
    return np.sum(np.abs(y-yhat))


def dmae(y, yhat):
    return 1 if y == yhat else -1


def kl_divergence(y, yhat):
    """KL散度"""
    return sum(y[i] * np.log2(y[i]/yhat[i]) for i in range(len(y)))


def entropy(y, factor=1e-15):
    """信息熵"""
    return -sum([y[i] * np.log2(y[i]+factor) for i in range(len(y))])


def cross_entropy(y, yhat, mode=None, factor=1e-15):
    """交叉熵"""
    return np.sum(np.nan_to_num(-y.ravel()*np.log(yhat.ravel())-(1-y.ravel())*np.log(1-yhat.ravel())))
