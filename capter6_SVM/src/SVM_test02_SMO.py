# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import SVM_Simple_and_Linear

if __name__ == '__main__':
	dataArr, classLabels = SVM_Simple_and_Linear.loadDataSet('testSet.txt')
	b, alphas = SVM_Simple_and_Linear.smoP(dataArr, classLabels, 0.6, 0.001, 40)
	w = SVM_Simple_and_Linear.calcWs(alphas,dataArr, classLabels)
	SVM_Simple_and_Linear.showClassifer(dataArr, classLabels, alphas, w, b)
