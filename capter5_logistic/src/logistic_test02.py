# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import logistic

"""
函数说明:使用Python写的Logistic分类器做预测
Parameters:
	无
Returns:
	无
"""
def colicTest():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = logistic.stocGradAscent1(np.array(trainingSet), trainingLabels,500)		#使用改进的随即上升梯度训练
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		if int(logistic.classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec) * 100 								#错误率计算
	print("测试集错误率为: %.2f%%" % errorRate)


"""
函数说明:使用Sklearn构建Logistic回归分类器
Parameters:无
Returns:无
"""
def colicSklearn():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingSet, trainingLabels)
	test_accurcy = classifier.score(testSet, testLabels) * 100
	print('正确率:%f%%' % test_accurcy)

if __name__ == '__main__':
	colicTest()
	colicSklearn()