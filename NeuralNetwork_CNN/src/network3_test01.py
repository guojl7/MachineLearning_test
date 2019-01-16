# -*- coding:UTF-8 -*-
import network3
from network3 import ReLU
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


if __name__ == '__main__':
    """全连接网络  迭代次数:60; 激活函数:sigmoid; 正则化:无 ; 学习速率:0.1 分类准确率:97:80%"""
    '''training_data, validation_data, test_data = network1.load_data_shared()
    mini_batch_size = 10
    net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)'''
    
    """卷积网络1:单层卷积  迭代次数:60; 激活函数:sigmoid; 正则化:无 ; 学习速率:0.1 分类准确率:98:78%"""
    '''training_data, validation_data, test_data = network1.load_data_shared()
    mini_batch_size = 10
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
                   FullyConnectedLayer(n_in=20*12*12, n_out=100),
                   SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)'''
    
    """卷积网络2:两层卷积 迭代次数:60; 激活函数:sigmoid; 正则化:无 ; 学习速率:0.1 分类准确率:99:06%"""
    '''training_data, validation_data, test_data = network1.load_data_shared()
    mini_batch_size = 10
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
                   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),filter_shape=(40, 20, 5, 5), poolsize=(2, 2)),
                   FullyConnectedLayer(n_in=40*4*4, n_out=100),
                   SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)'''
    
    """卷积网络3:两层卷积  迭代次数:60; 激活函数:ReLU; 正则化:L2,使⽤规范化参数lmbda=0.1; 学习速率:0.03 分类准确率:99.23%"""
    '''training_data, validation_data, test_data = network1.load_data_shared()
    mini_batch_size = 10
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
                   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
                   FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)'''
    
    """卷积网络4:两层卷积  迭代次数:60; 激活函数:ReLU; 正则化:L2,使⽤规范化参数lmbda=0.1; 学习速率:0.03; 使用拓展的训练集; 分类准确率:99.37%"""
    '''expanded_training_data, validation_data, test_data = network1.load_data_shared("../data/mnist_expanded.pkl.gz")
    mini_batch_size = 10
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
                   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
                   FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                   SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,validation_data, test_data, lmbda=0.1)'''
    
    
    """卷积网络5:两层卷积 ,两层全连接; 迭代次数:40; 激活函数:ReLU; 正则化:L2,使⽤规范化参数lmbda=0.1; 学习速率:0.03; 使用拓展的训练集; 分类准确率:99.60%"""
    expanded_training_data, validation_data, test_data = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
    mini_batch_size = 10
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
                   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
                   FullyConnectedLayer(n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
                   FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
                   SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], mini_batch_size)
    net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, validation_data, test_data)
    
    