#-*- coding:utf-8 -*-
from DNN import *
from datetime import datetime

# 数据加载器基类
'''
初始化加载器
path: 数据文件路径
count: 文件中的样本个数
'''
class Loader():
    def __init__(self, path, count):
        self.path = path
        self.count = count

    '''
           读取文件内容
    '''
    def get_file_content(self):
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

# 图像数据加载器
class ImageLoader(Loader):
    '''
         内部函数，从文件中获取图像
    '''
    def get_picture(self, content, index):
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(ord(content[start + i * 28 + j]))
        return picture
    
    '''
           内部函数，将图像转化为样本的输入向量
    '''
    def get_one_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    
    '''
           加载数据文件，获得全部样本的输入向量
    '''
    def load(self):
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set

# 标签数据加载器
'''
加载数据文件，获得全部样本的标签向量
'''
class LabelLoader(Loader):
    def load(self):
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels
    
    '''
            内部函数，将一个值转换为10维标签向量
    '''
    def norm(self, label):
        label_vec = []
        label_value = ord(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
    
'''
获得训练数据集
'''
def get_training_data_set():
    image_loader = ImageLoader('DNN_test01_dataset/train-images.idx3-ubyte', 600)
    label_loader = LabelLoader('DNN_test01_dataset/train-labels.idx1-ubyte', 600)
    return image_loader.load(), label_loader.load()

'''
获得测试数据集
'''
def get_test_data_set():
    image_loader = ImageLoader('DNN_test01_dataset/t10k-images.idx3-ubyte', 100)
    label_loader = LabelLoader('DNN_test01_dataset/t10k-labels.idx1-ubyte', 100)
    return image_loader.load(), label_loader.load()

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10], 0.3)
    while True:
        epoch += 1
        network.train(train_data_set, train_labels, 1)
        print(' epoch %d finished' % epoch)
        if epoch % 10 == 0:
            error_ratio = network.evaluate(test_data_set, test_labels)
            print(' after epoch %d, error ratio is %f' % ( epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()
    
