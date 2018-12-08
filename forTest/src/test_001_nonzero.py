# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
from numpy import *
import random
from blaze.expr.expressions import shape

b1 = array([[True], [False], [True], [False]])
b2 = array([True, False, True, False])
print('b1: %r' % b1)
print('b2: %r' % b2)

print(nonzero(b1))
print(nonzero(b2))