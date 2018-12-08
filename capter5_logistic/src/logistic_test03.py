# -*- coding:UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random
import logistic


"""
函数说明:加载数据
Parameters:无
Returns:
	dataMat - 数据列表
	labelMat - 标签列表
"""
def loadDataSet():
	dataMat = []														#创建数据列表
	labelMat = []														#创建标签列表
	fr = open('testSet.txt')											#打开文件	
	for line in fr.readlines():											#逐行读取
		lineArr = line.strip().split()									#去回车，放入列表
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		#添加数据
		labelMat.append(int(lineArr[2]))								#添加标签
	fr.close()															#关闭文件
	return dataMat, labelMat											#返回

"""
函数说明:绘制数据集
Parameters:
	weights - 权重参数数组
Returns:
	无
"""
def plotBestFit(weights):
	dataMat, labelMat = loadDataSet()									#加载数据集
	dataArr = np.array(dataMat)											#转换成numpy的array数组
	n = np.shape(dataMat)[0]											#数据个数
	xcord1 = []; ycord1 = []											#正样本
	xcord2 = []; ycord2 = []											#负样本
	for i in range(n):													#根据数据集标签进行分类
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])	#1为正样本
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])	#0为负样本
	fig = plt.figure()
	ax = fig.add_subplot(111)											#添加subplot
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)			#绘制负样本
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.title('BestFit')												#绘制title
	plt.xlabel('X1'); plt.ylabel('X2')									#绘制label
	plt.show()		

"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
	weights_array1 - 回归系数数组1
	weights_array2 - 回归系数数组2
Returns:无
"""
def plotWeights(weights_array1,weights_array2):
	#设置汉字格式
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
	fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
	x1 = np.arange(0, len(weights_array1), 1)
	#绘制w0与迭代次数的关系
	axs[0][0].plot(x1,weights_array1[:,0])
	axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
	plt.setp(axs0_title_text, size=20, weight='bold', color='black')  
	plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black') 
	#绘制w1与迭代次数的关系
	axs[1][0].plot(x1,weights_array1[:,1])
	axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
	plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black') 
	#绘制w2与迭代次数的关系
	axs[2][0].plot(x1,weights_array1[:,2])
	axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
	axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
	plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')  
	plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black') 


	x2 = np.arange(0, len(weights_array2), 1)
	#绘制w0与迭代次数的关系
	axs[0][1].plot(x2,weights_array2[:,0])
	axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
	axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
	plt.setp(axs0_title_text, size=20, weight='bold', color='black')  
	plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black') 
	#绘制w1与迭代次数的关系
	axs[1][1].plot(x2,weights_array2[:,1])
	axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
	plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black') 
	#绘制w2与迭代次数的关系
	axs[2][1].plot(x2,weights_array2[:,2])
	axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
	axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
	plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')  
	plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black') 

	plt.show()		

if __name__ == '__main__':
	dataMat, labelMat = loadDataSet()
	numIter = 150	
	weights1, weights_array1 = logistic.stocGradAscent11(np.array(dataMat), labelMat, numIter)

	weights2, weights_array2 = logistic.gradAscent(dataMat, labelMat)
	plotWeights(weights_array1, weights_array2)