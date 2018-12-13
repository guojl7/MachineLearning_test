# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import logistic
import random 
import matplotlib.pyplot as plt

"""
函数说明:绘制数据集

Parameters:
    weights - 权重参数数组
Returns:
    无
"""
def plotBestFit(bestWeights, dataSet, dataLabels):
    dataArr = np.array(dataSet)  
    m = np.shape(dataArr)[0]
    
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(m):                                                    #根据数据集标签进行分类
        if int(dataLabels[i]) == 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])    #1为正样本
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(0, 100, 0.1)
    y = (-bestWeights[0] - bestWeights[1] * x) / bestWeights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()        


"""
类说明:保存Logistic回归分类器参数
"""
class logisticParm():
    def __init__(self, penalty='l2', solver='liblinear', max_iter=100, multi_class='ovr', class_weight=None, C=1.0):
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.C = C
"""
函数说明:给定Logistic回归分类器参数，计算十次平均的正确率
                 数据每次随机70%用作训练集，30%用作测试集
Parameters:无
Returns:无
"""
def calcAccurcy(dataSet, dataLabels, testLogisticParm, testTimes=10):
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    testAccurcy = []
    predictY = []
    
    bestTestAccurcy = 0
    bestWeights = []
    for times in range(testTimes):
        trainingSet,testSet,trainingLabels,testLabels = train_test_split(dataSet,dataLabels,test_size=0.3,random_state=None)
        classifier = LogisticRegression(penalty = testLogisticParm.penalty, solver = testLogisticParm.solver, max_iter = testLogisticParm.max_iter).fit(trainingSet, trainingLabels)
        
        m = np.shape(testSet)[0]
        predictY = classifier.predict(testSet)
        errorCount = 0
        for i in range(m):
            if int(predictY[i] == int(testLabels[i])):
                errorCount += 1
                
        if(bestTestAccurcy < (float(errorCount)/m)):
            bestWeights = [classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[0][1]]
        
        testAccurcy.append((float(errorCount)/m))
        
    return sum(testAccurcy)/10 * 100, bestWeights

"""
函数说明:使用Sklearn构建Logistic回归分类器,遍历参数，找到最优的参数
Parameters:无
Returns:无
"""
def findBestLogisticParm():
    frData = open('ex2data1.txt')                                        #打开数据集
    dataSet = []; dataLabels = []
    for line in frData.readlines():
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        dataSet.append(lineArr)
        dataLabels.append(float(currLine[-1]))
    
    penaltyList = ['l1', 'l2']
    solverList = ['liblinear', 'newton-cg', 'lbfgs', 'sag']
    maxIterList = [4000, 5000, 6000, 7000]

    bestLogisticParm = logisticParm()
    bsetTestAccurcy = 0;
    bestWeights = []
    
    for penaltyParm in penaltyList:
        if('l1' == penaltyParm):
            solverParm = 'liblinear'
            for maxIterParm in maxIterList:
                testLogisticParm = logisticParm(penalty=penaltyParm, solver=solverParm, max_iter=maxIterParm)
                testAccurcy, weights = calcAccurcy(dataSet, dataLabels, testLogisticParm)
                if(bsetTestAccurcy < testAccurcy):
                    bsetTestAccurcy = testAccurcy
                    bestLogisticParm = testLogisticParm
                    bestWeights = weights
        else:
            for solverParm in solverList:
                for maxIterParm in maxIterList:
                    testLogisticParm = logisticParm(penalty=penaltyParm, solver=solverParm, max_iter=maxIterParm)
                    testAccurcy, weights= calcAccurcy(dataSet, dataLabels, testLogisticParm) 
                    if(bsetTestAccurcy < testAccurcy):
                        bsetTestAccurcy = testAccurcy
                        bestLogisticParm = testLogisticParm
                        bestWeights = weights

    m = np.shape(dataSet)[0]
    errorCount = 0
    for i in range(m):
        dataArr = []
        dataArr.append(1)
        dataArr.append(dataSet[i][0])
        dataArr.append(dataSet[i][1])
        if int(logistic.classifyVector(np.array(dataArr), bestWeights)) != int(dataLabels[i]):
            errorCount += 1
    errorRate = (float(errorCount)/m) * 100

    print('bestLogisticParm:  penalty:%r solver:%r max_iter:%r' % (bestLogisticParm.penalty, bestLogisticParm.solver, bestLogisticParm.max_iter))
    print('十次平均测试集正确率:%f%%' % bsetTestAccurcy)
    print('全部样本错误率:%f%%' % errorRate)
    plotBestFit(bestWeights, dataSet, dataLabels)

if __name__ == '__main__':
    findBestLogisticParm()
    
    