from libsvm.svmutil import *
from cifar10 import CIFAR10
import matplotlib.pyplot as plt
import numpy as np

# 1. 取数据
data_loader = CIFAR10()
train_data, val_data = data_loader.load(is_flatten=True, is_cast=True, is_gray=True)
x_train, y_train = train_data[0], train_data[1].ravel()
x_val, y_val = val_data[0], val_data[1].ravel()

# 2.加载模型 参数解释在readme文档
m = svm_train(y_train[:100], x_train[:100], '-c 4 -t 0')
p_label, p_acc, p_val = svm_predict(y_val[:100], x_val[:100], m)
print(f"准确率为{p_acc}%")
