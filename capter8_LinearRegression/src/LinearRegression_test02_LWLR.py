# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import LinearRegression


"""
函数说明:绘制多条局部加权回归曲线
Parameters:无
Returns:无
"""
def plotlwlrRegression():
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	xArr, yArr = LinearRegression.loadDataSet('ex0.txt')									#加载数据集
	yHat_1 = LinearRegression.lwlrTest(xArr, xArr, yArr, 1.0)							#根据局部加权线性回归计算yHat
	yHat_2 = LinearRegression.lwlrTest(xArr, xArr, yArr, 0.01)							#根据局部加权线性回归计算yHat
	yHat_3 = LinearRegression.lwlrTest(xArr, xArr, yArr, 0.003)							#根据局部加权线性回归计算yHat
	xMat = np.mat(xArr)													#创建xMat矩阵
	yMat = np.mat(yArr)													#创建yMat矩阵
	srtInd = xMat[:, 1].argsort(0)										#排序，返回索引值
	xSort = xMat[srtInd][:,0,:]
	fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))										

	axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'red')						#绘制回归曲线
	axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'red')						#绘制回归曲线
	axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'red')						#绘制回归曲线
	axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点
	axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点
	axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点

	#设置标题,x轴label,y轴label
	axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0',FontProperties=font)
	axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01',FontProperties=font)
	axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003',FontProperties=font)

	plt.setp(axs0_title_text, size=8, weight='bold', color='red')  
	plt.setp(axs1_title_text, size=8, weight='bold', color='red')  
	plt.setp(axs2_title_text, size=8, weight='bold', color='red')  

	plt.xlabel('X')
	plt.show()

if __name__ == '__main__':
	plotlwlrRegression()
	