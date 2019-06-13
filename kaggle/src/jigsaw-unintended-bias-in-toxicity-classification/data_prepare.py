# -*- coding:UTF-8 -*-
import jieba
import jieba.analyse
import logging
import os
import numpy as np
import re
import string
import nltk
import pandas as pd
from gensim.models import word2vec
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer

class prepare:
    def fileread(self, filepath):  #读文件
        train_df_org = pd.read_csv('./data/train.csv', usecols = ['target', 'comment_text'])
        test_df_org = pd.read_csv('./data/test.csv')
    
        sents_train_valide = np.array(train_df_org['comment_text'])
        labels_train_valide = np.array(train_df_org['target'])
        
        sents_test = np.array(test_df_org['comment_text'])
        id_test = np.array(test_df_org['id'])
        return sents_train_valide, labels_train_valide, sents_test, id_test
    
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
        sents_train_valide, labels_train_valide, sents_test, id_test = self.fileread(filepath)
        #sents=self.sentoken(raw)
        #taggedLine=self.POSTagger(sents)#暂不启用词性标注
        lowerLines=[self.wordlower(line) for line in sents_train_valide]
        cleanLines=[self.cleanlines(line) for line in lowerLines]
        words_train_valide = [self.wordtoken(line) for line in cleanLines]
        
        lowerLines=[self.wordlower(line) for line in sents_test]
        cleanLines=[self.cleanlines(line) for line in lowerLines]
        words_test = [self.wordtoken(line) for line in cleanLines]
        return words_train_valide, labels_train_valide, words_test, id_test
        #checkedWords=self.WordCheck(words)#暂不启用拼写检查
        #cleanWords=self.CleanWords(words)
        #stemWords=self.StemWords(cleanWords)
        #cleanWords=self.CleanWords(stemWords)#第二次清理出现问题，暂不启用
        #strLine=self.WordsToStr(stemWords)
        #self.WriteResult(strLine,resultPath)#一个文件暂时存成一行

if __name__ == '__main__':
    enPre = prepare()
    words_train_valide, labels_train_valide, words_test, id_test = enPre.main('./data')
    np.savez("./data/data.npz", words_train_valide, labels_train_valide, words_test, id_test)
