import torch
import torch.nn as nn
from cifar10 import CIFAR10
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
        activation = nn.ReLU()
    if not n_layers:
        n_layers = len(nodes_list)
    assert n_layers == len(nodes_list)
    if dropout_list:
        assert len(nodes_list) == len(dropout_list)
    fnn_model = nn.Sequential()
    if dropout_list:
        net_id = 0
        for node, dropout_rate in zip(nodes_list, dropout_list):
            fnn_model.add_module('linear'+str(net_id), nn.Linear(in_features=node[0], out_features=node[1]))
            fnn_model.add_module('act'+str(net_id), activation)
            fnn_model.add_module('dropout'+str(net_id), nn.Dropout(p=dropout_rate))
            net_id += 1
    else:
        for net_id, node in enumerate(nodes_list):
            fnn_model.add_module('linear' + str(net_id), nn.Linear(in_features=node[0], out_features=node[1]))
            fnn_model.add_module('act' + str(net_id), activation)
    fnn_model.add_module('final_linear', nn.Linear(in_features=node[1], out_features=10))
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
    x_train, y_train = train_data[0], train_data[1]
    x_val, y_val = val_data[0], val_data[1]
    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)
    y_train = torch.from_numpy(y_train.ravel().astype('int64'))
    y_val = torch.from_numpy(y_val.ravel().astype('int64'))
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=True)

    # 3. 构建模型
    if activation == 'relu':
        activation = nn.ReLU()

    new_nodes_list = []
    for i, value in enumerate(nodes_list):
        if i == 0:
            new_nodes_list.append([train_data[0][0].size(0), value])
        else:
            new_nodes_list.append([nodes_list[i-1], value])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(n_layers=n_layers, nodes_list=new_nodes_list,
                        activation=activation, dropout_list=dropout_list).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    each_epoch_loss = []
    for epoch in range(epochs):
        train_acc = 0
        train_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            pred = torch.max(outputs, 1)[1]
            acc = (pred == labels).sum()
            train_acc += acc.item()
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        each_epoch_loss.append(train_loss/len(train_loader))
        print(f'Epoch [{epoch + 1}/{epochs}], loss: {train_loss/len(train_loader)}, acc:{train_acc/len(train_data)}')
    plt.plot(np.array(range(len(each_epoch_loss))), each_epoch_loss, ls='-')
    plt.xlabel('epoch')
    plt.ylabel('pytorch result')
    plt.show()
