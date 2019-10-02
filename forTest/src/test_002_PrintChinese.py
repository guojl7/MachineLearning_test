# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
from numpy import *
import random
from blaze.expr.expressions import shape



class ForceStr(str):
    def __repr__(self):
        return super(ForceStr, self).__str__()

def switch_container( data ):
    ret = None
    if isinstance(data, str):
        ret = ForceStr(data)
    elif isinstance(data, unicode):
        ret = ForceStr(data.encode(sys.stdout.encoding))
    elif isinstance(data, list) or isinstance(data, tuple):
        ret = [switch_container(var) for var in data]
    elif isinstance(data, dict):
        ret = dict((switch_container(k), switch_container(v)) for k, v in data.iteritems())
    else:
        ret = data
    return ret

a = ['测试集的错误率']
print(a)
print(switch_container(a))

data = {'严': 1, 2: ['如'], 3:'玉'}
#print data
#print data[3]
#print(switch_container(data))


