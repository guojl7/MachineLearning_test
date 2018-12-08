# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random

"""
函数说明:加载数据
Parameters:
	fileName - 文件名
Returns:
	xArr - x数据集
	yArr - y数据集
"""
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	xArr = []; yArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr

"""
函数说明:绘制数据集
Parameters:无
Returns:无
"""
def plotDataSet():
	xArr, yArr = loadDataSet('ex0.txt')									#加载数据集
	n = len(xArr)														#数据个数
	xcord = []; ycord = []												#样本点
	for i in range(n):													
		xcord.append(xArr[i][1]); ycord.append(yArr[i])					#样本点
	fig = plt.figure()
	ax = fig.add_subplot(111)											#添加subplot
	ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)				#绘制样本点
	plt.title('DataSet')												#绘制title
	plt.xlabel('X')
	plt.show()

"""
函数说明:计算回归系数w
Parameters:
	xArr - x数据集
	yArr - y数据集
Returns:
	ws - 回归系数
"""
def standRegres(xArr,yArr):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	xTx = xMat.T * xMat							#根据文中推导的公示计算回归系数
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

"""
函数说明:使用局部加权线性回归计算回归系数w
Parameters:
	testPoint - 测试样本点
	xArr - x数据集
	yArr - y数据集
	k - 高斯核的k,自定义参数
Returns:ws - 回归系数
"""
def lwlr(testPoint, xArr, yArr, k = 1.0):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	m = np.shape(xMat)[0]
	weights = np.mat(np.eye((m)))										#创建权重对角矩阵
	for j in range(m):                      							#遍历数据集计算每个样本的权重
		diffMat = testPoint - xMat[j, :]     							
		weights[j, j] = np.exp((diffMat * diffMat.T)/(-2.0 * k**2))
	xTx = xMat.T * (weights * xMat)										
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = xTx.I * (xMat.T * (weights * yMat))							#计算回归系数
	return testPoint * ws

"""
函数说明:局部加权线性回归测试
Parameters:
	testArr - 测试数据集
	xArr - x数据集
	yArr - y数据集
	k - 高斯核的k,自定义参数
Returns:
	ws - 回归系数
"""
def lwlrTest(testArr, xArr, yArr, k=1.0):  
	m = np.shape(testArr)[0]											#计算测试数据集大小
	yHat = np.zeros(m)	
	for i in range(m):													#对每个样本点进行预测
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat

"""
函数说明:计算平方误差
Parameters:
	yArr - 预测值
	yHatArr - 真实值
Returns:
"""
def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()



"""
函数说明:数据标准化
Parameters:
	xMat - x数据集
	yMat - y数据集
Returns:
	inxMat - 标准化后的x数据集
	inyMat - 标准化后的y数据集
"""	
def regularize(xMat, yMat):
	inxMat = xMat.copy()														#数据拷贝
	yMean = np.mean(yMat, 0)													#行与行操作，求均值
	inyMat = yMat - yMean														#数据减去均值
	inMeans = np.mean(inxMat, 0)   												#行与行操作，求均值
	inVar = np.var(inxMat, 0)     												#行与行操作，求方差
	inxMat = (inxMat - inMeans) / inVar											#数据减去均值除以方差实现标准化
	return inxMat, inyMat

"""
函数说明:岭回归
Parameters:
	xMat - x数据集
	yMat - y数据集
	lam - 缩减系数
Returns:
	ws - 回归系数
"""
def ridgeRegres(xMat, yMat, lam = 0.2):
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

"""
函数说明:岭回归测试
Parameters:
	xMat - x数据集
	yMat - y数据集
Returns:
	wMat - 回归系数矩阵
"""
def ridgeTest(xArr, yArr):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	#数据标准化
	xMat, yMat = regularize(xMat, yMat)
	numTestPts = 30										#30个不同的lambda测试
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))	#初始回归系数矩阵
	for i in range(numTestPts):							#改变lambda计算回归系数
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))	#lambda以e的指数变化，最初是一个非常小的数，
		wMat[i, :] = ws.T 								#计算回归系数矩阵
	return wMat

"""
函数说明:前向逐步线性回归
Parameters:
	xArr - x输入数据
	yArr - y预测数据
	eps - 每次迭代需要调整的步长
	numIt - 迭代次数
Returns:
	returnMat - numIt次迭代的回归系数矩阵
"""
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T 										#数据集
	xMat, yMat = regularize(xMat, yMat)												#数据标准化
	n = np.shape(xMat)[1]
	returnMat = np.zeros((numIt, n))												#初始化numIt次迭代的回归系数矩阵
	ws = np.zeros((n, 1))															#初始化回归系数矩阵
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):															#迭代numIt次
		# print(ws.T)																	#打印当前回归系数矩阵
		lowestError = float('inf'); 												#正无穷
		for j in range(n):															#遍历每个特征的回归系数
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign												#微调回归系数
				yTest = xMat * wsTest												#计算预测值
				rssE = rssError(yMat.A, yTest.A)									#计算平方误差
				if rssE < lowestError:												#如果误差更小，则更新当前的最佳回归系数
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:] = ws.T 														#记录numIt次迭代的回归系数矩阵
	return returnMat

"""
函数说明:交叉验证岭回归
Parameters:
	xArr - x数据集
	yArr - y数据集
	numVal - 交叉验证次数
Returns:
	wMat - 回归系数矩阵
"""
def crossValidation(xArr, yArr, numVal = 10):
	m = len(yArr)																		#统计样本个数                       
	indexList = list(range(m))															#生成索引值列表
	errorMat = np.zeros((numVal,30))													#create error mat 30columns numVal rows
	for i in range(numVal):																#交叉验证numVal次
		trainX = []; trainY = []														#训练集
		testX = []; testY = []															#测试集
		random.shuffle(indexList)														#打乱次序
		for j in range(m):																#划分数据集:90%训练集，10%测试集
			if j < m * 0.9: 
				trainX.append(xArr[indexList[j]])
				trainY.append(yArr[indexList[j]])
			else:
				testX.append(xArr[indexList[j]])
				testY.append(yArr[indexList[j]])
		wMat = ridgeTest(trainX, trainY)    											#获得30个不同lambda下的岭回归系数
		for k in range(30):																#遍历所有的岭回归系数
			matTestX = np.mat(testX); matTrainX = np.mat(trainX)						#测试集
			meanTrain = np.mean(matTrainX,0)											#测试集均值
			varTrain = np.var(matTrainX,0)												#测试集方差
			matTestX = (matTestX - meanTrain) / varTrain 								#测试集标准化
			yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)						#根据ws预测y值
			errorMat[i, k] = rssError(yEst.T.A, np.array(testY))						#统计误差
	meanErrors = np.mean(errorMat,0)													#计算每次交叉验证的平均误差
	minMean = float(min(meanErrors))													#找到最小误差
	bestWeights = wMat[np.nonzero(meanErrors == minMean)]								#找到最佳回归系数

	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	meanX = np.mean(xMat,0); varX = np.var(xMat,0)
	unReg = bestWeights / varX															#数据经过标准化，因此需要还原
	print('crossValidation: %f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))
