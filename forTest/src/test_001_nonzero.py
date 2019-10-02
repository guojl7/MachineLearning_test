# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
#from numpy import *
import random
#from blaze.expr.expressions import shape
#import theano
import numpy as np
#import theano.tensor as T
import pandas as pd



b1 = np.array([[True], [False], [True], [False]])
b2 = np.array([True, False, True, False])
print('b1: %r' % b1)
print('b2: %r' % b2)

print(np.nonzero(b1))
print(np.nonzero(b2))


'''
ones = theano.shared(np.float32([[2,2,7],[0,0,1]]))
temp = T.arange(ones.shape[0])
print(temp.eval())
'''
print("3333333")
#train_df_org = pd.read_excel('./11111.xlsx')


x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.multiply(x1, x2)


print("1111")