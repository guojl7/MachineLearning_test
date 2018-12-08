#-*- coding:utf-8 -*-
from numpy import *
import pca

if __name__ == '__main__':
    dataMat = pca.replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    print(eigVals)