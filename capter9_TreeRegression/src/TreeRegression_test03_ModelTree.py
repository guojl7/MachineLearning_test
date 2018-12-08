#-*- coding:utf-8 -*-
import numpy as np
import TreeRegression
from dask.array.routines import corrcoef


if __name__ == '__main__':
    train_filename = 'bikeSpeedVsIq_train.txt'
    train_Mat = np.mat(TreeRegression.loadDataSet('bikeSpeedVsIq_train.txt'))
    test_Mat = np.mat(TreeRegression.loadDataSet('bikeSpeedVsIq_test.txt'))
    #回归树
    myTree_regressionTree = TreeRegression.createTree(train_Mat, ops=(1,20))
    yHat_regressionTree = TreeRegression.createForeCast(myTree_regressionTree, test_Mat[:,0])
    #模型树
    myTree_ModelTree = TreeRegression.createTree(train_Mat, TreeRegression.modelLeaf, TreeRegression.modelErr, ops=(1,20))
    yHat_ModelTree = TreeRegression.createForeCast(myTree_ModelTree, test_Mat[:,0], TreeRegression.modelTreeEval)
    ws,X,Y = TreeRegression.linearSolve(train_Mat)
    #线性回归
    m = np.shape(test_Mat)[0]
    yHat_linearRegression = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat_linearRegression[i] = test_Mat[i,0]*ws[1,0] + ws[0,0]
    print('回归树相关系数: %r' % np.mat(corrcoef(yHat_regressionTree, test_Mat[:,1],rowvar=0))[0,1])
    print('模型树相关系数: %r' % np.mat(corrcoef(yHat_ModelTree, test_Mat[:,1],rowvar=0))[0,1])
    print('线性回归相关系数: %r' % np.mat(corrcoef(yHat_linearRegression, test_Mat[:,1],rowvar=0))[0,1])