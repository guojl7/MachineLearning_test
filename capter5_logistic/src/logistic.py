# -*- coding:UTF-8 -*-
import numpy as np
import random

"""
函数说明:梯度上升算法测试函数
求函数f(x) = -x^2 + 4x的极大值
Parameters:无
Returns:无
"""
def Gradient_Ascent_test():
    def f_prime(x_old):                                    #f(x)的导数
        return -2 * x_old + 4
    x_old = -1                                            #初始值，给一个小于x_new的值
    x_new = 0                                            #梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01                                        #步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001                                #精度，也就是更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)            #上面提到的公式
    print(x_new)                                        #打印最终求解的极值近似值

"""
函数说明:sigmoid函数
Parameters: inX 
Returns:
    sigmoid
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))
    weights_array = np.array([])                                            #存储每次更新的回归系数
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array,weights)     #添加回归系数到数组中
    weights_array = weights_array.reshape(maxCycles,n)             #改变维度
    return weights, weights_array

"""
函数说明:随机梯度上升算法
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:weights - 求得的回归系数数组(最优参数)
"""
def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                               #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01
    weights = np.ones(n)                                                       #参数初始化
    weights_array = np.array([])                                            #存储每次更新的回归系数
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))                    #选择随机选取的一个样本，计算h
        error = classLabels[i] - h                                 #计算误差
        weights = weights + alpha * error * dataMatrix[i]          #更新回归系数
        weights_array = np.append(weights_array,weights,axis=0)           #添加回归系数到数组中
    weights_array = weights_array.reshape(numIter*m,n)             #改变维度
    return weights, weights_array                                  #返回
        
"""
函数说明:改进的随机梯度上升算法
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
    weights_array - 每次更新的回归系数
"""
def stocGradAscent1(dataMatrix, classLabels, numIter):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化                                        #存储每次更新的回归系数
    for j in range(numIter):                                            
        dataIndex = list(range(m))
        for i in range(m):            
            alpha = 4/(1.0+j+i)+0.01                                           #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                  #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]          #更新回归系数
            del(dataIndex[randIndex])                                          #删除已经使用的样本
    return weights                                             #返回

"""
函数说明:改进的随机梯度上升算法
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
    weights_array - 每次更新的回归系数
"""
def stocGradAscent11(dataMatrix, classLabels, numIter):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化
    weights_array = np.array([])                                            #存储每次更新的回归系数
    for j in range(numIter):                                            
        dataIndex = list(range(m))
        for i in range(m):            
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            weights_array = np.append(weights_array,weights,axis=0)         #添加回归系数到数组中
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    weights_array = weights_array.reshape(numIter*m,n)                         #改变维度
    return weights, weights_array                                             #返回

"""
函数说明:分类函数
Parameters:
    inX - 特征向量
    weights - 回归系数
Returns: 分类结果
"""
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0