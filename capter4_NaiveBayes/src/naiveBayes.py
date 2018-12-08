# -*- coding: UTF-8 -*-
from numpy import *
from functools import reduce
import re

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters: dataSet - 整理的样本数据集
Returns: vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(listOPosts):
	vocabSet = set([])  					#创建一个空的不重复列表
	for document in listOPosts:				
		vocabSet = vocabSet | set(document) #取并集
	return list(vocabSet)


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)									#创建一个其中所含元素都为0的向量
	for word in inputSet:												#遍历每个词条
		if word in vocabList:											#如果词条存在于词汇表中，则置1
			returnVec[vocabList.index(word)] = 1
		else: 
			print("the word: %s is not in my Vocabulary!" % word)
	return returnVec													#返回文档向量


"""
函数说明:根据vocabList词汇表，构建词袋模型
Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词袋模型
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)										#创建一个其中所含元素都为0的向量
    for word in inputSet:												#遍历每个词条
        if word in vocabList:											#如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec													#返回词袋模型


"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 非的条件概率数组
	p1Vect - 侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
"""
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)							#计算训练的文档数目
	numWords = len(trainMatrix[0])							#计算每篇文档的词条数
	pAbusive = sum(trainCategory)/float(numTrainDocs)		#文档属于侮辱类的概率
	p0Num = ones(numWords); p1Num = ones(numWords)	    #创建numpy.zeros数组,
	p0Denom = 2.0; p1Denom = 2.0                        	#分母初始化为0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:							#统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)									#相除        
	p0Vect = log(p0Num/p0Denom) 
	#p1Vect = (p1Num/p1Denom)									#相除        
	#p0Vect = (p0Num/p0Denom)          
	print(p1Vect)
	print(p0Vect)
	return p0Vect,p1Vect,pAbusive							#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 非侮辱类的条件概率数组
	p1Vec -侮辱类的条件概率数组
	pClass1 - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	#p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1    	    #对应元素相乘,未修改分类器
	#p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)               			#对应元素相加
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	print('p0:',p0)
	print('p1:',p1)
	if p1 > p0:
		return 1
	else: 
		return 0
	
	
"""
函数说明:接收一个大字符串并将其解析为字符串列表
Parameters: 无
Returns:无
"""
def textParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写
