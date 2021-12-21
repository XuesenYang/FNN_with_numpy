import matplotlib.pyplot as plt
import numpy as np
import random


def get_data_batch(input_x, input_y, batch_size=None, shuffle=False):
    """
    循环产生批量数据batch
    :param inputs:
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    """
    rows = input_x.shape[0]
    indices = list(range(rows))

    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])
        indices = indices[batch_size:] + indices[:batch_size]
        yield input_x[batch_indices], input_y[batch_indices]


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.dloss = None

    def add(self, layer):
        self.layers.append(layer)

    def useLoss(self, loss, dloss):
        self.loss = loss
        self.dloss = dloss

    def useOptimizer(self, optimizer, learning_rate, beta):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta = beta

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, batch_size):
        samples = len(x_train)
        each_epoch_loss = []
        for i in range(epochs):
            loss = 0
            acc = 0
            for batch_x, batch_y in get_data_batch(x_train, y_train, batch_size=batch_size, shuffle=False):
                output = batch_x
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                pred = np.where(output == np.max(output, axis=0))
                true = np.where(batch_y == 1)
                acc = len([i for i, j in zip(pred, true)])
                loss += self.loss(batch_y, output)
                acc += self.dloss(batch_y, output)
                for layer in reversed(self.layers):
                    if layer != self.layers[-1]:
                        b_loss = layer.backward_propagation(b_loss, self.optimizer, self.learning_rate)
            loss /= samples
            acc /= samples
            each_epoch_loss.append(loss)
            print(f'epoch {epochs} loss:{loss} acc{acc}')
        plt.plot(np.array(range(len(each_epoch_loss))), each_epoch_loss, ls='-')
        plt.xlabel('epoch')
        plt.ylabel('numpy result')
        plt.show()
