# -*- coding: utf-8 -*-
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
from ResNetModel import resnet50_test02
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils


model = load_model('E:/raw_data_wash/resnet50_test02.h5')
print(model.input.name)
print(model.output.name)
print(model.output.op.name)

'''
# 保存图为pb文件
sess = K.get_session()
frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[model.output.op.name])
tf.train.write_graph(frozen_graph_def, 'E:/raw_data_wash', 'resnet50_test02.pb', as_text=False)
'''

num_classes = 5
batch_size = 4
target_size = (224, 224)

imageData = np.load("E:/raw_data_wash/imageData.npz")
img_x = imageData["arr_0"]
img_y = imageData["arr_1"]
image_dict = imageData["arr_2"]


img = image.load_img(img_x[100], target_size = target_size)
x_image_data = image.img_to_array(img)
x_image_data = np.expand_dims(x_image_data, axis=0)
x_image_data = preprocess_input(x_image_data)
y_image_data = np_utils.to_categorical(img_y[100].astype('int32'), num_classes) # labels 

print(x_image_data)
print(y_image_data)
print(model.predict(x_image_data))

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    
    with open('E:/raw_data_wash/resnet50_test02.pb', "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
 
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
 
        input_x = sess.graph.get_tensor_by_name(model.input.name)
        output = sess.graph.get_tensor_by_name(model.output.name)
        print(sess.run(output, feed_dict={input_x : x_image_data}))

