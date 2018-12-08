#-*- coding:utf-8 -*-
import numpy as np
import TreeRegression

if __name__ == '__main__':
    print('剪枝前:')
    train_filename = 'ex2.txt'
    train_Data = TreeRegression.loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = TreeRegression.createTree(train_Mat)
    print(tree)
    print('\n剪枝后:')
    test_filename = 'ex2test.txt'
    test_Data = TreeRegression.loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    treeAfterPrune = TreeRegression.prune(tree, test_Mat)
    print(treeAfterPrune)