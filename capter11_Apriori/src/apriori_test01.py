#-*- coding:utf-8 -*-
from numpy import *
import apriori

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

if __name__ == '__main__':
    dataSet = loadDataSet()
    L1, supportData = apriori.apriori(dataSet, 0.5)
    rules = apriori.generateRules(L1, supportData, minConf = 0.7) 
    print(rules)
    
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L_mush,supportData_mush = apriori.apriori(mushDatSet, minSupport = 0.3)
    for item in L_mush[1]:
        if item.intersection('2'):print item