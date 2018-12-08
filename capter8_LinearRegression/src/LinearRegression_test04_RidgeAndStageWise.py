# -*-coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import LinearRegression

"""
函数说明:绘制岭回归系数矩阵
"""
def plotwMat():
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	abX, abY = LinearRegression.loadDataSet('abalone.txt')
	redgeWeights = LinearRegression.ridgeTest(abX, abY)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(redgeWeights)	
	ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
	ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
	ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
	plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()

"""
函数说明:绘制前向逐步线性回归系数矩阵
"""
def plotstageWiseMat():
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	xArr, yArr = LinearRegression.loadDataSet('abalone.txt')
	returnMat = LinearRegression.stageWise(xArr, yArr, 0.005, 1000)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(returnMat)	
	ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties = font)
	ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties = font)
	ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
	plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()

if __name__ == '__main__':
	plotwMat()
	plotstageWiseMat()
