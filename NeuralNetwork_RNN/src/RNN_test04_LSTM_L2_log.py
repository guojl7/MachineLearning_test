# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(1)
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn import metrics

def convert_to_one_hot(Y, label_to_index):
    Y_index = [label_to_index[label] for label in Y]
    Y_one_hot = np.eye(len(label_to_index))[Y_index]
    return Y_one_hot

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
    m = X.shape[0]                                   # number of training examples
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))
    for i in range(m):                               # loop over training examples
        # Loop over the words of sentence_words
        for j, w in enumerate(X[i]):
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w] + 1
    ### END CODE HERE ###

    return X_indices

def pretrained_embedding_layer(emb_matrix):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    Returns:
    embedding_layer -- pretrained layer Keras instance
    """              
    vocab_len = emb_matrix.shape[0]                  # define dimensionality of your GloVe word vectors (= 50)
    emb_dim = emb_matrix.shape[1]
    emb_matrix_add_raw = np.zeros((vocab_len + 1, emb_dim)) # adding 1 to fit Keras embedding (requirement)
    emb_matrix_add_raw[1 : (vocab_len + 1), :] = emb_matrix
    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    ### END CODE HERE ###
    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def log_categorical_model(input_shape, emb_matrix):
    """
    Function creating the Emojify-v2 model's graph.
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    Returns:
    model -- a model instance in Keras
    """
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    # Create the embedding layer pretrained with GloVe Vectors (鈮? line)
    embedding_layer = pretrained_embedding_layer(emb_matrix)
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
    X = Dense(4, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs = sentence_indices ,outputs = X)
    ### END CODE HERE ###

    return model


if __name__ == '__main__':
    label_to_index = {'Inf' : 0,
                      'Ntc' : 1,
                      'Dbg' : 2,
                      'Err' : 3}
    
    load_data = np.load("./data/word2vec/emb_matrix.npz")
    X_data = load_data["arr_0"]
    Y_data = load_data["arr_1"]
    index_to_word = load_data["arr_2"]
    emb_matrix = load_data["arr_3"]
    maxLen = len(max(X_data, key = len))

    word_to_index = {}
    for index, word in enumerate(index_to_word):
        word_to_index[word] = index
    
    X_data_indices = sentences_to_indices(X_data, word_to_index, maxLen)
    Y_data_oh = convert_to_one_hot(Y_data, label_to_index)

    train_data_X, validate_data_X, train_data_Y, validate_data_Y = train_test_split(np.array(X_data_indices), np.array(Y_data_oh), test_size=0.2, random_state=7)
    
    model = log_categorical_model((maxLen,), emb_matrix)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data_X, train_data_Y, epochs = 2, batch_size = 32, shuffle = True)

    loss, acc = model.evaluate(validate_data_X, validate_data_Y)
    print("Test accuracy = ", acc)

    # This code allows you to see the mislabelled examples
    err_num = 0
    pred = model.predict(validate_data_X)
    Y_validate_data_index = [np.argmax(item) for item in validate_data_Y]
    Y_validate_data_pred_index = [np.argmax(item) for item in pred]
    print(metrics.accuracy_score(Y_validate_data_index, Y_validate_data_pred_index))#准确率
    print(metrics.precision_score(Y_validate_data_index, Y_validate_data_pred_index, average='micro'))  # 微平均，精确率)
    metrics.precision_score(Y_validate_data_index, Y_validate_data_pred_index, average='macro')  # 宏平均，精确率
    metrics.recall_score(Y_validate_data_index, Y_validate_data_pred_index, average='micro')
    metrics.recall_score(Y_validate_data_index, Y_validate_data_pred_index, average='macro')
    metrics.f1_score(Y_validate_data_index, Y_validate_data_pred_index, average='macro') 
    metrics.confusion_matrix(Y_validate_data_index, Y_validate_data_pred_index)
    print(metrics.classification_report(Y_validate_data_index, Y_validate_data_pred_index, target_names=['Inf', 'Ntc', 'Dbg']))
    
    metrics.cohen_kappa_score(Y_validate_data_index, Y_validate_data_pred_index)
    
    for i in range(len(validate_data_X)):
        Y_pred = np.argmax(pred[i])
        Y = np.argmax(validate_data_Y[i])
        if(Y != Y_pred):
            err_num = err_num + 1
            print('Expected:'+ (list(label_to_index.keys())[list(label_to_index.values()).index(Y)]) + ',  prediction:' + ((list(label_to_index.keys())[list(label_to_index.values()).index(Y_pred)])) + ',  sentences:' + str([index_to_word[int(index - 1)] for index in validate_data_X[i] if 0 != index]))

    print(sum(Y_data_oh) * 100/Y_data_oh.sum())
    print('1')