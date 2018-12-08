#-*- coding:utf-8 -*-
from numpy import *

"""
函数说明:构建第一个候选集集合C1
Parameters:
    dataSet - 数据集
Returns: 
    C1
"""
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])  
    C1.sort()
    return map(frozenset, C1)#use frozen set so we can use it as a key in a dict    


"""
函数说明:构建第一个候选集集合C1
Parameters:
    D - 数据集
    Ck - 候选集列表
    minSupport - 最小支持度
Returns: 
    retList - 最频繁项集
    supportData - 最频繁项集的支持度
"""
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


"""
函数说明:creates Ck
Parameters:
    Lk - 最频繁项集，项集元素个数为k-1
    k - 项集元素个数
Returns: 
    Ck - 候选集列表
"""
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


"""
函数说明:apriori算法
Parameters:
    dataSet - 数据集
    minSupport - 最小支持度
Returns: 
    L - 最频繁项集
    supportData - 最频繁项集的支持度
"""
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


"""
函数说明:关联规则生成函数
Parameters:
    L-频繁项集列表 
    supportData - 包含频繁项集支持数据的字典
    minConf-最小可信度阈值
Returns: 
    bigRuleList - 关联规则list
"""
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

"""
函数说明:计算可信度
Parameters:
    freqSet-频繁项集
    H - 后件集合
    supportData - 包含频繁项集支持数据的字典
    brl- 关联规则list
    minConf-最小可信度阈值
Returns: 
    prunedH - create new list to return
"""
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

"""
函数说明:从最初的项集中生成更多的关联规则
Parameters:
    freqSet-频繁项集
    H - 后件集合
    supportData - 包含频繁项集支持数据的字典
    brl- 关联规则list
    minConf-最小可信度阈值
Returns: 
    无
"""
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
