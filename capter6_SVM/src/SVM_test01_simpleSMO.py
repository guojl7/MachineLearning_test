# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import SVM_Simple_and_Linear
from numpy import *

"""
函数说明:数据可视化
Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
Returns:无
"""
def showDataSet(dataMat, labelMat):
	data_plus = []                                  #正样本
	data_minus = []                                 #负样本
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = array(data_plus)              #转换为numpy矩阵
	data_minus_np = array(data_minus)            #转换为numpy矩阵
	plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1])   #正样本散点图
	plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1]) #负样本散点图
	plt.show()

if __name__ == '__main__':
    dataMat, labelMat = SVM_Simple_and_Linear.loadDataSet('testSetSVM_Simple_and_Lineart')
    b,alphas = SVM_Simple_and_Linear.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = SVM_Simple_and_Linear.get_w(dataMat, labelMat, alphas)
    #w = SVM.calcWs(dataMat, labeSVM_Simple_and_Lineart, alphas)
    SVM_Simple_and_Linear.showClassifer(labelMat, alphas, w, b)
	