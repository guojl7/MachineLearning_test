# -*-coding:utf-8 -*-
from numpy import *
import AdaBoost

if __name__ == '__main__':
	dataArr, LabelArr = AdaBoost.loadDataSet('horseColicTraining2.txt')
	weakClassArr, aggClassEst = AdaBoost.adaBoostTrainDS(dataArr, LabelArr)
	testArr, testLabelArr = AdaBoost.loadDataSet('horseColicTest2.txt')
	print(weakClassArr)
	predictions = AdaBoost.adaClassify(dataArr, weakClassArr)
	errArr = mat(ones((len(dataArr), 1)))
	print('训练集的错误率:%.3f%%' % float(errArr[predictions != mat(LabelArr).T].sum() / len(dataArr) * 100))
	predictions = AdaBoost.adaClassify(testArr, weakClassArr)
	errArr = mat(ones((len(testArr), 1)))
	print('测试集的错误率:%.3f%%' % float(errArr[predictions != mat(testLabelArr).T].sum() / len(testArr) * 100))
	