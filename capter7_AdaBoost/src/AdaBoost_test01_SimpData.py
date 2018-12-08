# -*-coding:utf-8 -*-
from numpy import *
import AdaBoost

"""
创建单层决策树的数据集
Parameters:无
Returns:
    dataMat - 数据矩阵
    classLabels - 数据标签
"""
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    weakClassArr, aggClassEst = AdaBoost.adaBoostTrainDS(dataArr, classLabels)
    print(AdaBoost.adaClassify([[0,0],[5,5]], weakClassArr))
    AdaBoost.showDataSet(dataArr, classLabels)
