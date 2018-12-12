# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import logistic
import random 

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
    m = np.array(dataSet).shape[0]
    for times in range(testTimes):
        resultList=random.sample(range(0,m-1),int(0.3*m));                       #表示从[0,m-1]间随机生成0.3*m个数，结果以列表返回
        for i in range(m):
            if(i in resultList):
                testSet.append(dataSet[i])
                testLabels.append(dataLabels[i])
            else:
                trainingSet.append(dataSet[i])
                trainingLabels.append(dataLabels[i])
        
        classifier = LogisticRegression(penalty = testLogisticParm.penalty, solver = testLogisticParm.solver, max_iter = testLogisticParm.max_iter).fit(trainingSet, trainingLabels)
        testAccurcy.append(classifier.score(testSet, testLabels))
        
    return sum(testAccurcy)/10 * 100

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
    
    for penaltyParm in penaltyList:
        if('l1' == penaltyParm):
            solverParm = 'liblinear'
            for maxIterParm in maxIterList:
                testLogisticParm = logisticParm(penalty=penaltyParm, solver=solverParm, max_iter=maxIterParm)
                testAccurcy = calcAccurcy(dataSet, dataLabels, testLogisticParm)
                if(bsetTestAccurcy < testAccurcy):
                    bsetTestAccurcy = testAccurcy
                    bestLogisticParm = testLogisticParm
        else:
            for solverParm in solverList:
                for maxIterParm in maxIterList:
                    testLogisticParm = logisticParm(penalty=penaltyParm, solver=solverParm, max_iter=maxIterParm)
                    testAccurcy = calcAccurcy(dataSet, dataLabels, testLogisticParm) 
                    if(bsetTestAccurcy < testAccurcy):
                        bsetTestAccurcy = testAccurcy
                        bestLogisticParm = testLogisticParm

    print('bestLogisticParm:  penalty:%r solver:%r max_iter:%r' % (bestLogisticParm.penalty, bestLogisticParm.solver, bestLogisticParm.max_iter))
    print('十次平均正确率:%f%%' % bsetTestAccurcy)

if __name__ == '__main__':
    findBestLogisticParm()
    
    