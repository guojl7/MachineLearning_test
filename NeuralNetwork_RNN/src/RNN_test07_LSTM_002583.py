# -*- coding:UTF-8 -*-
import pandas as pd
import tushare as ts
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from sklearn import preprocessing

#收集数据
def get_002583_data():
    data_2011 = ts.get_h_data('002583', start='2011-05-27', end='2011-12-31') #两个日期之间的前复权数据
    data_2012 = ts.get_h_data('002583', start='2012-01-01', end='2012-12-31') #两个日期之间的前复权数据
    data_2013 = ts.get_h_data('002583', start='2013-01-01', end='2013-12-31') #两个日期之间的前复权数据
    data_2014 = ts.get_h_data('002583', start='2014-01-01', end='2014-12-31') #两个日期之间的前复权数据
    data_2015 = ts.get_h_data('002583', start='2015-01-01', end='2015-12-31') #两个日期之间的前复权数据
    data_2016 = ts.get_h_data('002583', start='2016-01-01', end='2016-12-31') #两个日期之间的前复权数据
    data_2017 = ts.get_h_data('002583', start='2017-01-01', end='2017-12-31') #两个日期之间的前复权数据
    data_2018 = ts.get_h_data('002583', start='2018-01-01', end='2018-12-31') #两个日期之间的前复权数据
    data_2019 = ts.get_h_data('002583', start='2019-01-01', end='2019-03-08') #两个日期之间的前复权数据
    
    #data_2011 = pd.read_csv("2011.csv")
    #data_2012 = pd.read_csv("2012.csv")
    #data_2013 = pd.read_csv("2013.csv")
    #data_2014 = pd.read_csv("2014.csv")
    #data_2015 = pd.read_csv("2015.csv")
    #data_2016 = pd.read_csv("2016.csv")
    #data_2017 = pd.read_csv("2017.csv")
    #data_2018 = pd.read_csv("2018.csv")
    #data_2019 = pd.read_csv("2019.csv")
    
    #拼接数据
    data_002583 = pd.concat([data_2019, data_2018, data_2017, data_2016, data_2015, data_2014, data_2013, data_2012, data_2011])
    #data_002583 = data_002583.reset_index(drop=True)
    data_002583.to_csv("./data/002583.csv", index=False)

def load_002583_data(window_lenth = 10, valid_size = 0.2, test_size = 0.2, seed = 7):
    data_X = []
    data_Y = []
    data_002583 = pd.read_csv("./data/002583.csv")
    data_002583 = data_002583.sort_index(ascending=False)
    data_002583 = data_002583.reset_index(drop=True)
    
    if len(data_002583) <= window_lenth:
        raise Exception('invalid window_lenth:' + str(window_lenth))
    
    data_preprocess = preprocessing.scale(data_002583.values[:,1:5])
    for i in range(0, len(data_002583) - window_lenth):
        data_X.append(data_preprocess[i : i + window_lenth])
        int_labels = 0
        if data_002583.values[i + window_lenth, 1] > data_002583.values[i + window_lenth - 1, 1]:
            int_labels = int_labels + 8
        if data_002583.values[i + window_lenth, 2] > data_002583.values[i + window_lenth - 1, 2]:
            int_labels = int_labels + 4
        if data_002583.values[i + window_lenth, 3] > data_002583.values[i + window_lenth - 1, 3]:
            int_labels = int_labels + 2
        if data_002583.values[i + window_lenth, 4] > data_002583.values[i + window_lenth - 1, 4]:
            int_labels = int_labels + 1
        categorical_labels = to_categorical(int_labels, num_classes = 16)
        data_Y.append(categorical_labels)
    
    train_X, valid_test_X, train_Y, valid_test_Y = train_test_split(data_X, data_Y, test_size= (valid_size+test_size), random_state = seed)
    valid_X, test_X, valid_Y, test_Y = train_test_split(valid_test_X, valid_test_Y, test_size= test_size / (valid_size+test_size), random_state = seed)
    
    return np.array(train_X), np.array(train_Y), np.array(valid_X), np.array(valid_Y), np.array(test_X), np.array(test_Y)
    
def LSTM_V2(input_shape):
    """
    Function creating the Emojify-v2 model's graph.
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    Returns:
    model -- a model instance in Keras
    """
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    input_indices = Input(shape = input_shape, dtype = 'float32')
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences = True)(input_indices)
    X = LSTM(128, return_sequences = True)(X)
    # Add dropout with a probability of 0.5
    #X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences = False)(X)
    # Add dropout with a probability of 0.5
    #X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(16, activation='relu')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs = input_indices ,outputs = X)
    ### END CODE HERE ###

    return model


if __name__ == '__main__':
    window_lenth = 10
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = load_002583_data(window_lenth = window_lenth)
    
    model = LSTM_V2((window_lenth,4,))
    model.summary()
    opt = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs = 100, batch_size = 16, shuffle = True)
    
    count_1 = 0
    count_2 = 0
    count_4 = 0
    count_8 = 0
    count_all = 0
    for i in range(0, len(valid_X)):
        X = np.expand_dims(valid_X[i], axis=0)
        Y = model.predict(X)
        
        if np.argmax(Y) == np.argmax(valid_Y[i]):
            count_all+=1
        if (np.argmax(Y)&1) == (np.argmax(valid_Y[i])&1):
            count_1+=1
        if (np.argmax(Y)&2) == (np.argmax(valid_Y[i])&2):
            count_2+=1
        if (np.argmax(Y)&4) == (np.argmax(valid_Y[i])&4):
            count_4+=1
        if (np.argmax(Y)&8) == (np.argmax(valid_Y[i])&8):
            count_8+=1
            
    print(count_all/len(valid_X))
    print(count_1/len(valid_X))
    print(count_2/len(valid_X))
    print(count_4/len(valid_X))
    print(count_8/len(valid_X))

    loss, acc = model.evaluate(valid_X, valid_Y)
    
    
