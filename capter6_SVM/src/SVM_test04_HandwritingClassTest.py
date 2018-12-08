# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import SVM_KernelTrans
from numpy import *																		#返回SMO算法计算的b和alphas

"""
将32x32的二进制图像转换为1x1024向量。
Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量
"""
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

"""
加载图片
Parameters:
	dirName - 文件夹的名字
Returns:
    trainingMat - 数据矩阵
    hwLabels - 数据标签
"""
def loadImages(dirName):
	from os import listdir
	hwLabels = []
	trainingFileList = listdir(dirName) 
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]     
		classNumStr = int(fileStr.split('_')[0])
		if classNumStr == 9: hwLabels.append(-1)
		else: hwLabels.append(1)
		trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
	return trainingMat, hwLabels    

"""
测试函数
Parameters:
	kTup - 包含核函数信息的元组
Returns:
    无
"""
def testDigits(kTup=('rbf', 10)):
	dataArr,labelArr = loadImages('trainingDigits')
	b,alphas = SVM_KernelTrans.smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	svInd = nonzero(alphas.A>0)[0]
	sVs=datMat[svInd] 
	labelSV = labelMat[svInd];
	print("支持向量个数:%d" % shape(sVs)[0])
	m,n = shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = SVM_KernelTrans.kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]): errorCount += 1
	print("训练集错误率: %.2f%%" % (float(errorCount)/m))
	dataArr,labelArr = loadImages('testDigits')
	errorCount = 0
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	for i in range(m):
		kernelEval = SVM_KernelTrans.kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]): errorCount += 1    
	print("测试集错误率: %.2f%%" % (float(errorCount)/m))

if __name__ == '__main__':
	testDigits()