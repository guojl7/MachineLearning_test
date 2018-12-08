# -*-coding:utf-8 -*-
import AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import *

if __name__ == '__main__':
	dataArr, classLabels = AdaBoost.loadDataSet('horseColicTraining2.txt')
	testArr, testLabelArr = AdaBoost.loadDataSet('horseColicTest2.txt')
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
	bdt.fit(dataArr, classLabels)
	predictions = bdt.predict(dataArr)
	errArr = mat(ones((len(dataArr), 1)))
	print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
	predictions = bdt.predict(testArr)
	errArr = mat(ones((len(testArr), 1)))
	print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))