#-*- coding:utf-8 -*-
from DNN import *
from datetime import datetime

def train_and_evaluate():
    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10
    learning_rate = 0.1
    last_error_ratio = 1.0
    epoch = 0

    # load the DNN_test01 training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    # load the DNN_test01 test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    train_data_set = []
    train_labels = []
    test_data_set = []
    test_labels = []
    
    for record in training_data_list:
        all_values = record.split(',')
        
        train_data = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        train_data_set.append(train_data.tolist())
        
        train_label = [0.01 for i in range(output_nodes)]
        train_label[int(all_values[0])] = 0.99
        train_labels.append(train_label)
    
    for record in test_data_list:
        all_values = record.split(',')
        
        test_data = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        test_data_set.append(test_data.tolist())
        
        teat_label = [0.01 for i in range(output_nodes)]
        teat_label[int(all_values[0])] = 0.99
        test_labels.append(teat_label)

    network = Network([input_nodes, hidden_nodes, output_nodes], learning_rate)
    
    equalCnt = 0
    while True:
        epoch += 1
        network.train(train_data_set, train_labels, 1)
        print(' epoch %d finished' % epoch)
        if epoch % 10 == 0:
            error_ratio = network.evaluate(test_data_set, test_labels)
            print(' after epoch %d, error ratio is %f' % ( epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            elif error_ratio == last_error_ratio:
                if 2 <= equalCnt:
                    break
                else:
                    equalCnt += 1
            else:
                equalCnt = 0
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()
