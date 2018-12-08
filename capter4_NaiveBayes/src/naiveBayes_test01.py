# -*- coding: UTF-8 -*-
import naiveBayes
from numpy import *

"""
函数说明:创建实验样本
Parameters:无
Returns: postingList - 实验样本切分的词条
         classVec - 类别标签向量
"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]                                                           #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec        


"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
"""
def testingNB():
    listOPosts,listClasses = loadDataSet()                                    #创建实验样本
    myVocabList = naiveBayes.createVocabList(listOPosts)                                #创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(naiveBayes.setOfWords2Vec(myVocabList, postinDoc))                #将实验样本向量化
    p0V,p1V,pAb = naiveBayes.trainNB0(array(trainMat),array(listClasses))                #训练朴素贝叶斯分类器
    
    testEntry = ['love', 'my', 'dalmation']                                    #测试样本1
    thisDoc = array(naiveBayes.setOfWords2Vec(myVocabList, testEntry))                #测试样本向量化
    if naiveBayes.classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'insult')                                        #执行分类并打印分类结果
    else:
        print(testEntry,'NonInsult')                                        #执行分类并打印分类结果
    
    testEntry = ['stupid', 'garbage']                                        #测试样本2
    thisDoc = array(naiveBayes.setOfWords2Vec(myVocabList, testEntry))                #测试样本向量化
    if naiveBayes.classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'insult')                                        #执行分类并打印分类结果
    else:
        print(testEntry,'NonInsult')                                        #执行分类并打印分类结果

if __name__ == '__main__':
    testingNB()