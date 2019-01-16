# -*- coding:UTF-8 -*-
import network2
import mnist_loader

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
    net.save("trained_net")
    
    n_test = len(test_data)
    trainedNet = network2.load("trained_net")
    accuracy = trainedNet.accuracy(test_data)
    print "Accuracy on test data: {} %".format((float)(accuracy*100)/n_test)
    