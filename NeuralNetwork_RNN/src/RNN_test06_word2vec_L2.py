# -*- coding:UTF-8 -*-
import jieba
import jieba.analyse
import logging
import os
import numpy as np
import re
import string
import nltk
from gensim.models import word2vec
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer


class prepare:
    def fileread(self, filepath):  #读文件
        sents = []
        labels = []
        with open(filepath, 'r', encoding = 'utf-8') as f:
            for line in f:
                sents.append(line[72:])
                labels.append(line[43:46])
            #sents = [line[72:] for line in f]
            #labels = [line[43:46] for line in f]
        return sents, labels
    
    def sentoken(self,raw):        #分句子，去除前面一部分内容
        return [line[72:] for line in raw]
    
    def wordlower(self, line):    #转换成小写
        return line.lower()
    
    def cleanlines(self,line):   #去除标点等无用的符号
        return re.sub('[^a-zA-Z]',' ',line)
        return line
    
    def wordtoken(self,sent):    #分词
        wordsinstr = WordPunctTokenizer().tokenize(sent)
        #wordsinstr=nltk.word_tokenize(sent)
        return wordsinstr
    
    def cleanwords(self,words):   #去除停用词
        cleanwords=[]
        sr={}.fromkeys([line.strip() for line in open("停用词表的地址")])
        for words in words: 
            cleanwords+=[[w.lower() for w in words if w.lower() not in sr]]
        return cleanwords   
    
    def stemwords(self,cleanwordslist):    #词干提取
        temp=[]
        stemwords=[]
        stemmer=SnowballStemmer('english')
        porter=nltk.PorterStemmer()
        for words in cleanwordslist:
            temp+=[[stemmer.stem(w) for w in words]]
        for words in temp:
            stemwords+=[[porter.stem(w) for w in words]]
        return stemwords
    
    def wordstostring(self,stemwords):
        strline=[]
        for words in stemwords:
            strline+=[w for w in words]
        return strline
    
    def main(self, filepath):
        sents, labels = self.fileread(filepath)
        #sents=self.sentoken(raw)
        #taggedLine=self.POSTagger(sents)#暂不启用词性标注
        lowerLines=[self.wordlower(line) for line in sents]
        cleanLines=[self.cleanlines(line) for line in lowerLines]
        words = [self.wordtoken(line) for line in cleanLines]
        return words, labels
        #checkedWords=self.WordCheck(words)#暂不启用拼写检查
        #cleanWords=self.CleanWords(words)
        #stemWords=self.StemWords(cleanWords)
        #cleanWords=self.CleanWords(stemWords)#第二次清理出现问题，暂不启用
        #strLine=self.WordsToStr(stemWords)
        #self.WriteResult(strLine,resultPath)#一个文件暂时存成一行

if __name__ == '__main__':
    enPre = prepare()
    words, labels = enPre.main('./data/word2vec/log_bpb_cpu_4_15_16_7_45')
    model = word2vec.Word2Vec(words, hs = 1, min_count = 1, window = 3, size = 100)
    
    np.savez("./data/word2vec/emb_matrix.npz", words, labels, model.wv.index2word, model.wv.vectors)
    print(model['phy'])
