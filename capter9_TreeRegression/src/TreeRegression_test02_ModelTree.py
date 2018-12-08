#-*- coding:utf-8 -*-
import numpy as np
import TreeRegression


if __name__ == '__main__':
    train_filename = 'exp2.txt'
    train_Data = TreeRegression.loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = TreeRegression.createTree(train_Mat, TreeRegression.modelLeaf, TreeRegression.modelErr)
    print('模型树:')
    print(tree)