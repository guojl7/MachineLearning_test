# -*- coding:utf-8 -*-
import numpy as np

# Sigmoid激活函数类
class SigmoidActivator():
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return np.multiply(output, (1 - output))

# 全连接层实现类
class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size #本层输入向量的维度
        self.output_size = output_size #本层输出向量的维度
        self.activator = activator #激活函数
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size)) # 权重数组W
        #self.W = np.random.normal(0.0, pow(input_size, -0.5), (output_size, input_size)) # 权重数组W
        self.b = np.zeros((output_size, 1)) # 偏置项b
        self.output = np.zeros((output_size, 1)) # 输出向量
    
    '''
           前向计算
    input_array: 输入向量，维度必须等于input_size
    '''
    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, self.input) + self.b)
    
    '''
            反向计算W和b的梯度
    delta_array: 从上一层传递过来的误差项
    '''
    def backward(self, delta_array):
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array
    
    '''
            使用梯度下降算法更新权重
    '''
    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

# 神经网络类
class Network():
    def __init__(self, layers, learningRate):
        self.learningRate = learningRate
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i+1],SigmoidActivator()))
    '''
           使用神经网络实现预测
    sample: 输入样本
    '''
    def predict(self, sample_array):
        output = sample_array
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    '''
           训练函数
    labels: 样本标签
    data_set: 输入样本
    rate: 学习速率
    epoch: 训练轮数
    '''
    def train(self, data_set, labels, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                data_array = np.array(data_set[d], ndmin=2).T
                label_array = np.array(labels[d], ndmin=2).T
                self.predict(data_array)
                self.calc_gradient(label_array)
                self.update_weight()
            
    def train_one_sample(self, data_array, label_array):
        self.predict(data_array)
        self.calc_gradient(label_array)
        self.update_weight()

    def calc_gradient(self, label_array):
        delta = self.layers[-1].activator.backward(self.layers[-1].output)*(label_array - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    
    def update_weight(self):
        for layer in self.layers:
            layer.update(self.learningRate)

    def evaluate(self, test_data_set, test_labels):
        error = 0
        total = len(test_data_set)
        for i in range(total):
            label = np.argmax(test_labels[i])
            test_data_array = np.array(test_data_set[i], ndmin=2).T
            predict = np.argmax(self.predict(test_data_array))
            if label != predict:
                error += 1
        return float(error) / float(total)

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    '''
            梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    def gradient_check(self, sample_feature, sample_label):
        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)
        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, fc.W_grad[i, j]))
                    
