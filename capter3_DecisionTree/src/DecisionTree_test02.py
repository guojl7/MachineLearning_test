# -*- coding: UTF-8 -*-
from sklearn.externals.six import StringIO
import numpy as np
import DecisionTree
import copy

"""
使用决策树预测隐形眼镜的类型
"""
if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    tempLensesLabels = copy.deepcopy(lensesLabels)
    lensesTree = DecisionTree.createTree(lenses, tempLensesLabels)
    testVec = ['presbyopic', 'myope', 'yes', 'normal']   #测试数据
    result = DecisionTree.classify(lensesTree, lensesLabels, testVec)
    print(result)
    DecisionTree.createPlot(lensesTree)
