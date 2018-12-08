# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import LinearRegression

if __name__ == '__main__':
	abX, abY = LinearRegression.loadDataSet('abalone.txt')
	print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
	yHat01 = LinearRegression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
	yHat1 = LinearRegression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
	yHat10 = LinearRegression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
	print('k=0.1时,误差大小为:%r' % LinearRegression.rssError(abY[0:99], yHat01.T))
	print('k=1  时,误差大小为:%r' % LinearRegression.rssError(abY[0:99], yHat1.T))
	print('k=10 时,误差大小为:%r' % LinearRegression.rssError(abY[0:99], yHat10.T))

	print('')

	print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
	yHat01 = LinearRegression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
	yHat1 = LinearRegression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
	yHat10 = LinearRegression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
	print('k=0.1时,误差大小为:%r' % LinearRegression.rssError(abY[100:199], yHat01.T))
	print('k=1  时,误差大小为:%r' % LinearRegression.rssError(abY[100:199], yHat1.T))
	print('k=10 时,误差大小为:%r' % LinearRegression.rssError(abY[100:199], yHat10.T))

	print('')

	print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
	print('k=1时,误差大小为:%r' % LinearRegression.rssError(abY[100:199], yHat1.T))
	ws = LinearRegression.standRegres(abX[0:99], abY[0:99])
	yHat = np.mat(abX[100:199]) * ws
	print('简单的线性回归误差大小:%r' % LinearRegression.rssError(abY[100:199], yHat.T.A))