# -*- coding: utf-8 -*-
import numpy as np
import os
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
#from ResNetModel import resnet50
from keras.applications.resnet50 import ResNet50
from keras import backend as K

# 读取图片函数
def load_images(paths, target_size=(224, 224), normalize=True):
    img_train_x = []
    
    for img_path in paths:
        img = image.load_img(img_path, target_size = target_size)
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_train_x.append(x)
    return np.array(img_train_x)

def get_train_batch(X_train, y_train, num_classes, batch_size, target_size):
    '''
           参数：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
        batch_size:批次
        target_size:图片大小
           返回: 一个generator，
        x: 获取的批次图片 
        y: 获取的图片对应的标签
    '''
    while 1:
        for i in range(0, len(X_train), batch_size):
            x = load_images(X_train[i:i+batch_size], target_size)
            y = np_utils.to_categorical(y_train[i:i+batch_size].astype('int32'), num_classes) # labels
            
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield(x, y)

def get_image_data(X_image, y_image, num_classes, target_size):
    '''
           参数：
        X_valid：所有图片路径列表
        y_valid: 所有图片对应的标签列表
        target_size:图片大小
           返回: x: 获取的批次图片 
        y: 获取的图片对应的标签
    '''
    x = load_images(X_image, target_size)
    y = np_utils.to_categorical(y_image.astype('int32'), num_classes) # labels 
    return x, y

if __name__ == '__main__':
    num_classes = 5
    epochs = 1
    batch_size = 4
    target_size = (224, 224)
    
    imageData = np.load("E:/raw_data_wash/imageData.npz")
    img_x = imageData["arr_0"]
    img_y = imageData["arr_1"]
    image_dict = imageData["arr_2"]
    
    
    np.random.seed(43)
    shuffled_indices=np.random.permutation(len(img_x))
    train_indices=shuffled_indices[0:8000]
    valid_indices =shuffled_indices[8000:9000]
    test_indices=shuffled_indices[9000:10000]
    X_train = img_x[train_indices]
    X_valid = img_x[valid_indices]
    X_test = img_x[test_indices]
    Y_train = img_y[train_indices]
    Y_valid = img_y[valid_indices]
    Y_test = img_y[test_indices]
    
    
    # 90%用于测试集，5%用于验证集， 5%用于测试集
    #X_train, X_valid_test, Y_train, Y_valid_test = train_test_split(img_x[0:10000], img_y[0:10000], test_size=0.4, random_state=7)
    #X_valid, X_test, Y_valid, Y_test = train_test_split(X_valid_test, Y_valid_test, test_size=0.5, random_state=7)
    
    K.clear_session()
    model = ResNet50(input_shape = (224, 224, 3), classes = num_classes, weights = None)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

    x_valid_image_data, y_valid_image_data = get_image_data(X_valid, Y_valid, num_classes, target_size)

    result = model.fit_generator(generator = get_train_batch(X_train, Y_train, num_classes, batch_size, target_size), 
                                 epochs = epochs,
                                 steps_per_epoch= X_train.shape[0] // batch_size, 
                                 validation_data=(x_valid_image_data, y_valid_image_data),
                                 verbose=1)
             
    x_test_image_data, y_test_image_data = get_image_data(X_test, Y_test, num_classes, target_size)
    preds = model.evaluate(x_test_image_data, y_test_image_data)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    # 保存模型
    model.save('E:/raw_data_wash/resnet50_test02.h5')   # HDF5文件，pip install h5py


    