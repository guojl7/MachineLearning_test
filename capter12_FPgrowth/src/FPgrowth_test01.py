#-*- coding:utf-8 -*-
import FPgrowth

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

if __name__ == '__main__':
    minSup = 3
    simpDat = loadSimpDat()
    initSet = FPgrowth.createInitSet(simpDat)
    myFPtree, myHeaderTab = FPgrowth.createTree(initSet, minSup)
    myFPtree.disp()
    myFreqList = []
    FPgrowth.mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    print(myFreqList)
