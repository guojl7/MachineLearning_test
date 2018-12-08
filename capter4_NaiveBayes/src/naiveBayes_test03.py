# -*- coding: UTF-8 -*-
import naiveBayes
from numpy import *
import re
import feedparser
import operator
from audioop import reverse
from nltk.app.nemo_app import textParams


"""
功能描述：遍历词汇表中的每个词并统计在文本中出现的次数，然后排序返回前30
"""
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30]   

def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    print(minLen)
    print(len(feed1['entries']))
    print(len(feed0['entries']))
    for i in range(minLen):
        wordList = naiveBayes.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = naiveBayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = naiveBayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW)
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(naiveBayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = naiveBayes.trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = naiveBayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if naiveBayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount +=1
    print 'the error rate is:',float(errorCount)/len(testSet)
    return vocabList, p0V, p1V
  

"""
从个人广告中获取区域倾向，rss无法获取
"""
if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss') 
    print(ny)
    print(sf)
    
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse = True)
    print "SF:"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key = lambda pair: pair[1], reverse = True)
    print "NY:"
    for item in sortedNY:
        print item[0]
