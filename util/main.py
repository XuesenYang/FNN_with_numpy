from util.network import Network
from util.dense import Dense
from util.activation_layer import Activation
from util.activation_functions import relu, d_relu, softmax, dsoftmax
from util.loss_functions import mse, cross_entropy, dmse
from util.optimizer_functions import SGD
from cifar10 import CIFAR10
from tensorflow.keras.utils import to_categorical
from configparser import ConfigParser
import os


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
        activation = relu
        dactivation = d_relu
    if activation == relu:
        dactivation = d_relu
    if not n_layers:
        n_layers = len(nodes_list)
    assert n_layers == len(nodes_list)
    if dropout_list:
        assert len(nodes_list) == len(dropout_list)
    fnn_model = Network()
    for net_id, node in enumerate(nodes_list):
        fnn_model.add(Dense(node[0], node[1]))
        fnn_model.add(Activation(activation, dactivation))
    fnn_model.add(Dense(node[1], 10))
    fnn_model.add(Activation(softmax, dactivation=None))
    return fnn_model


if __name__ == '__main__':
    # 1.加载参数
    conf = ConfigParser()
    conf_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "config.ini")
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

    x_train, y_train = train_data[0], train_data[1]
    x_val, y_val = val_data[0], val_data[1]
    # x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    # x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # 3.构建模型
    new_nodes_list = []
    for i, value in enumerate(nodes_list):
        if i == 0:
            new_nodes_list.append([x_train.shape[1], value])
        else:
            new_nodes_list.append([nodes_list[i-1], value])
    model = build_model(n_layers=n_layers, nodes_list=new_nodes_list,
                        activation=relu, dropout_list=dropout_list)
    # 4. 训练
    model.useLoss(cross_entropy, dmse)
    model.useOptimizer(SGD(), learning_rate=learning_rate, beta=0.9)
    model.fit(x_train[:100], y_train[:100], epochs=epochs, batch_size=batch_size)

