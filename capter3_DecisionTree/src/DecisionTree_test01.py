# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle
import copy
import DecisionTree

if __name__ == '__main__':
    dataSet, labels = DecisionTree.createDataSet()
    print(dataSet)
    print(labels)
    tempLabels = copy.deepcopy(labels)
    myTree = DecisionTree.createTree(dataSet, tempLabels)
    print(myTree)
    testVec = [0,0,1,1]
    #storeTree(myTree,'classifierStorage.txt')  #决策树的储存
    myTree1 = DecisionTree.grabTree('classifierStorage.txt') #决策树的读取
    print(myTree1)
    result = DecisionTree.classify(myTree1, labels, testVec)
    if result == 'yes':
        print('放贷')
    elif result == 'no':
        print('不放贷')
    else:
        print('决策失败')
    DecisionTree.createPlot(myTree) #绘制树形图