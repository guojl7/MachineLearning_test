# -*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import logging
import os
import re
import string
import nltk
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(1)
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
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
        #return sents_test, id_test
    
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
        #sents_test, id_test = self.fileread(filepath)
        #sents=self.sentoken(raw)
        #taggedLine=self.POSTagger(sents)#暂不启用词性标注
        
        lowerLines=[self.wordlower(line) for line in sents_train_valide]
        cleanLines=[self.cleanlines(line) for line in lowerLines]
        words_train_valide = [self.wordtoken(line) for line in cleanLines]
        
        lowerLines=[self.wordlower(line) for line in sents_test]
        cleanLines=[self.cleanlines(line) for line in lowerLines]
        words_test = [self.wordtoken(line) for line in cleanLines]
        return np.array(words_train_valide), np.array(labels_train_valide), np.array(words_test), np.array(id_test)
        #return words_test, id_test
        #checkedWords=self.WordCheck(words)#暂不启用拼写检查
        #cleanWords=self.CleanWords(words)
        #stemWords=self.StemWords(cleanWords)
        #cleanWords=self.CleanWords(stemWords)#第二次清理出现问题，暂不启用
        #strLine=self.WordsToStr(stemWords)
        #self.WriteResult(strLine,resultPath)#一个文件暂时存成一行

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]  # number of training examples
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))

    for i in range(m):                               # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        #sentence_words = X[i].lower().split()
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in X[i]:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
            else:
                X_indices[i, j] = 0
            # Increment j to j + 1
            j = j + 1
    ### END CODE HERE ###

    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    ### END CODE HERE ###
    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def DoubleLSTM_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    # Create the embedding layer pretrained with GloVe Vectors (鈮? line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences = True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences = False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(1, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs = sentence_indices ,outputs = X)
    ### END CODE HERE ###
    return model


if __name__ == '__main__':
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')
    
    np.savez("./data/word_to_index.npz", word_to_index)
    np.savez("./data/index_to_word.npz", index_to_word)
    np.savez("./data/word_to_vec_map.npz", word_to_vec_map)
    
    enPre = prepare()
    words_train_valide, labels_train_valide, words_test, id_test = enPre.main('./data')
    #words_test, id_test = enPre.main('./data')
    maxLen = max(len(max(words_train_valide, key = len)), len(max(words_test, key = len)))
    #maxLen = len(max(words_test, key = len))
    
    X_train, X_validate, y_train, y_validate = train_test_split(words_train_valide, labels_train_valide, test_size=20000, random_state=0)
    
    
    #X_test_indices = sentences_to_indices(words_test, word_to_index, maxLen)

    X_train_indices = sentences_to_indices(np.array(X_train), word_to_index, maxLen)
    X_validate_indices = sentences_to_indices(np.array(X_validate), word_to_index, maxLen)
    
    model = DoubleLSTM_V2((maxLen,), word_to_vec_map, word_to_index)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_indices, y_train, epochs = 50, batch_size = 32, shuffle = True)


    test_data_Y = model.predict(words_test)
    pd.DataFrame({"id": id_test, "prediction": test_data_Y.reshape(-1)}).to_csv('./data/Submission.csv', index = False, header = True)

