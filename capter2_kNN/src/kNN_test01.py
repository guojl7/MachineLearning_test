# -*- coding: utf-8 -*-
from numpy import *
import operator
import kNN_base

def createDataSet(): #ʹ��python��������
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__ == '__main__':
    group, labels = createDataSet()
    k = 3
    result = kNN_base.classify([0, 0], group, labels, 3)
    
    print(result)