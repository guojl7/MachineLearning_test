# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import SVM_KernelTrans

"""
测试函数
Parameters: k1 - 使用高斯核函数的时候表示到达率
Returns:无
"""
def testRbf(k1 = 1.3):
    dataArr,labelArr = SVM_KernelTrans.loadDataSet('testSetRBF.txt')                        #加载训练集
    SVM_KernelTrans.showDataSet(dataArr, labelArr)
    b,alphas = SVM_KernelTrans.smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))        #根据训练集计算b和alphas
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]                                        #获得支持向量
    sVs = datMat[svInd]                                                     
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = SVM_KernelTrans.kernelTrans(sVs,datMat[i,:],('rbf', k1))                #计算各个点的核
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b     #根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1        #返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("训练集错误率: %.2f%%" % ((float(errorCount)/m)*100))             #打印错误率
    dataArr,labelArr = SVM_KernelTrans.loadDataSet('testSetRBF2.txt')                         #加载测试集
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()         
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = SVM_KernelTrans.kernelTrans(sVs,datMat[i,:],('rbf', k1))                 #计算各个点的核            
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b         #根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1          #返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("测试集错误率: %.2f%%" % ((float(errorCount)/m)*100))                   #打印错误率

if __name__ == '__main__':
    testRbf()