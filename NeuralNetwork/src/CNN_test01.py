#-*- coding:utf-8 -*-
from CNN import *
from activators import *

def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    cl = ConvLayer(5,5,3,3,3,2,1,2,IdentityActivator(),0.001)
    cl.filters[0].weights = np.array(
        [[[-1,1,0],
          [0,1,0],
          [0,1,1]],
         [[-1,-1,0],
          [0,0,0],
          [0,-1,0]],
         [[0,0,-1],
          [0,1,0],
          [1,-1,-1]]], dtype=np.float64)
    cl.filters[0].bias=1
    cl.filters[1].weights = np.array(
        [[[1,1,-1],
          [-1,-1,1],
          [0,-1,1]],
         [[0,1,0],
         [-1,0,-1],
          [-1,1,0]],
         [[-1,0,0],
          [-1,0,1],
          [-1,0,0]]], dtype=np.float64)
    return a, b, cl


def test():
    a, b, cl = init_test()
    cl.forward(a)
    print(cl.output_array)

def test_bp():
    a, b, cl = init_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])
    
'''
梯度检查
'''
def gradient_check():
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    
    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)
    # 计算梯度
    cl.backward(a, sensitivity_array, IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d,i,j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d,i,j] -= 2*epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = int((err1 - err2) / (2 * epsilon))
                cl.filters[0].weights[d,i,j] += epsilon
                print('weights(%d,%d,%d): expected - actural %f - %f' % (d, i, j, expect_grad, cl.filters[0].weights_grad[d,i,j]))
                
def init_pool_test():
    a = np.array(
        [[[1,1,2,4],
          [5,6,7,8],
          [3,2,1,0],
          [1,2,3,4]],
         [[0,1,2,3],
          [4,5,6,7],
          [8,9,0,1],
          [3,4,5,6]]], dtype=np.float64)

    b = np.array(
        [[[1,2],
          [2,4]],
         [[3,5],
          [8,2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4,4,2,2,2,2)
    return a, b, mpl


def test_pool():
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print('input array:\n%s\noutput array:\n%s' % (a, mpl.output_array))


def test_pool_bp():
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (a, b, mpl.delta_array))
    
    
if __name__ == '__main__':
    test()