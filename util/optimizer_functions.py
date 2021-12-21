import numpy as np

"""基本的优化函数"""


class SGD:
    def minimize(self, w, b, d_w, d_b, v_w, v_b, learning_rate=0.01, beta=0.9):
        """梯度下降法
        :param w: 权重
        :param b: 偏置值
        :param d_w: 权重梯度
        :param d_b: 偏置值梯度
        :param v_w: 历史权重状态值
        :param v_b: 历史偏置值状态值
        :param learning_rate: 学习率
        :param beta: 动量项系数
        :return:
        """

        w_updated = w - learning_rate * d_w
        b_updated = b - learning_rate * d_b
        return w_updated, b_updated, v_w, v_b


class Momentum:
    def minimize(self, w, b, d_w, d_b, v_w, v_b, learning_rate=0.01, beta=0.9):

        v_w = beta * v_w + (1-beta) * d_w
        v_b = beta * v_b + (1-beta) * d_b
        w_updated = w - learning_rate * v_w
        b_updated = b - learning_rate * v_b

        return w_updated, b_updated, v_w, v_b


# class AdaGrad:
#     def __init__(self):
#         super().__init__()
#         self.sum_dx = 0
#         self.sum_dy = 0
#
#     def minimize(self, w, b, d_w, d_b, v_w, v_b, learning_rate=0.01):
#         epsilon = 1e-8
#         self.sum_dx += d_w**2
#         self.sum_dy += d_b**2
#         w_updated = w - (learning_rate/(np.sqrt(epsilon + self.sum_dx)))*d_w
#         b_updated = b - (learning_rate/(np.sqrt(epsilon + self.sum_dy)))*d_b
#
#         return w_updated, b_updated, v_w, v_b
#
#
# class Adam:
#     def __init__(self):
#         super().__init__()
#         self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0, 0, 0, 0, 0
#
#     def minimize(self,  w, b, d_w, d_b, v_w, v_b, learning_rate=0.01):
#         beta_1 = 0.9
#         beta_2 = 0.999
#         self.t += 1
#         epsilon = 1e-8
#
#         self.m_x = beta_1*self.m_x + (1-beta_1)*d_w
#         self.m_y = beta_1*self.m_y + (1-beta_1)*d_b
#         self.v_x = beta_2*self.v_x + (1-beta_2)*(d_w**2)
#         self.v_y = beta_2*self.v_y + (1-beta_2)*(d_b**2)
#
#         m_x_hat = self.m_x/(1-beta_1**self.t)
#         m_y_hat = self.m_y/(1-beta_1**self.t)
#         v_x_hat = self.v_x/(1-beta_2**self.t)
#         v_y_hat = self.v_y/(1-beta_2**self.t)
#
#         w_updated = w - (learning_rate*m_x_hat)/(np.sqrt(v_x_hat)+epsilon)
#         b_updated = b - (learning_rate*m_y_hat)/(np.sqrt(v_y_hat)+epsilon)
#
#         return w_updated, b_updated, v_w, v_b
#
#



