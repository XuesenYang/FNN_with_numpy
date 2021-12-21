from tensorflow.keras import layers, Sequential
import tensorflow as tf
from cifar10 import CIFAR10
from tensorflow.keras.utils import to_categorical
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np


def build_model(n_layers, nodes_list, activation, dropout_list):
    """
    构建FNN网络
    :param n_layers: 网络层数
    :param nodes_list: 网络节点数量集合
    :param activation: 激活函数
    :param dropout_list: dropout率集合, None表示不使用dropout
    :return: 自定义FNN模型
    """
    if not activation:
        activation = 'relu'
    if not n_layers:
        n_layers = len(nodes_list)
    assert n_layers == len(nodes_list)
    if dropout_list:
        assert len(nodes_list) == len(dropout_list)
    fnn_model = Sequential()
    if dropout_list:
        for node, dropout_rate in zip(nodes_list, dropout_list):
            fnn_model.add(layers.Dense(node, activation=activation))
            fnn_model.add(layers.Dropout(dropout_rate))
    else:
        for node in nodes_list:
            fnn_model.add(layers.Dense(node, activation=activation))

    fnn_model.add(layers.Dense(10, activation='softmax'))
    return fnn_model


if __name__ == '__main__':
    # 1.加载参数
    conf = ConfigParser()
    conf_path = 'config.ini'
    conf.read(conf_path)
    n_layers = conf.getint('basic_config', 'n_layers')
    nodes_list = conf.get('basic_config', 'nodes_list').split(',')
    learning_rate = conf.getfloat('basic_config', 'learning_rate')
    batch_size = conf.getint('basic_config', 'batch_size')
    epochs = conf.getint('basic_config', 'epochs')

    if nodes_list:
        nodes_list = [int(i) for i in nodes_list]
    activation = conf.get('basic_config', 'activation')
    dropout_list = conf.get('basic_config', 'dropout_list').split(',')
    if dropout_list:
        dropout_list = [float(i) for i in dropout_list]

    # 2.加载数据
    data_loader = CIFAR10()
    train_data, val_data = data_loader.load(is_flatten=True, is_cast=True, is_gray=True)

    # 3.数据预处理
    x_train, y_train = train_data[0], train_data[1]
    x_val, y_val = val_data[0], val_data[1]
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # 4.构建模型
    if activation == 'relu':
        activation = tf.nn.relu

    model = build_model(n_layers=n_layers, nodes_list=nodes_list, activation=activation, dropout_list=dropout_list)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 5.模型训练
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
    each_epoch_loss = history.history['loss']
    each_epoch_accuracy = history.history['accuracy']
    plt.plot(np.array(range(len(each_epoch_loss))), each_epoch_loss, ls='-')
    plt.xlabel('epoch')
    plt.ylabel('tensorflow result')
    plt.show()

