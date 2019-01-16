# -*- coding:UTF-8 -*-
import network1
import mnist_loader
from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM values correct: %s " % ((float)(num_correct*100)/len(test_data[1]))

def network_onehide_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network1.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == '__main__':
    """神经网络预测"""
    network_onehide_baseline()
    
    """SVN预测"""
    #svm_baseline()