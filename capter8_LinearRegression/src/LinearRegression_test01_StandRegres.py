# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import LinearRegression
from dask.array.routines import corrcoef

"""
函数说明:绘制回归曲线和数据点
Parameters:无
Returns:无
"""
def plotRegression():
	xArr, yArr = LinearRegression.loadDataSet('ex0.txt')									#加载数据集
	ws = LinearRegression.standRegres(xArr, yArr)										#计算回归系数
	xMat = np.mat(xArr)													#创建xMat矩阵
	yMat = np.mat(yArr)													#创建yMat矩阵
	xCopy = xMat.copy()													#深拷贝xMat矩阵
	xCopy.sort(0)														#排序
	yHat = xCopy * ws 													#计算对应的y值
	fig = plt.figure()
	ax = fig.add_subplot(111)											#添加subplot
	ax.plot(xCopy[:, 1], yHat, c = 'red')								#绘制回归曲线
	ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5)				#绘制样本点
	plt.title('DataSet')												#绘制title
	plt.xlabel('X')
	plt.show()


if __name__ == '__main__':
	plotRegression()
	