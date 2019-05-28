# -*- coding: utf-8 -*-
import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# this function is for read image,the input is directory name
def load_image(directory_name, target_size=(224, 224)):
    image_dict = {}
    img_x = [] # this if for store all of the image data
    img_y = []
    image_label = 0;
    count = 0
    
    for image_label_name in os.listdir(directory_name):
        image_dict[image_label] = image_label_name

        image_directory_name = directory_name + "/" + image_label_name + "/IMAGES"
    
        # this loop is for read each image in this foder,directory_name is the foder name with images.
        for filename in os.listdir(image_directory_name):
            img_path = image_directory_name + "/" + filename
            #img = image.load_img(img_path, target_size=(204, 204))
            
            try:
                img = image.load_img(img_path)
            except:
                continue
            
            if img.size == (161, 81):
                continue
            
            width_height_tuple = (target_size[1], target_size[0])
            img = img.resize(width_height_tuple, Image.BICUBIC)
            
            x = image.img_to_array(img)
            x = preprocess_input(x)
            img_x.append(x)
            img_y.append(image_label)
            count += 1
            
            if 0 == count % 1024:
                save_name = "E:/test/imageData_" + (str)(count) + ".npz"
                np.savez(save_name, img_x, img_y)
                img_x = []
                img_y = []
                
        
        image_label += 1
    
    
    np.savez("E:/test/imageData_last.npz", img_x, img_y)
    #r = np.load("E:/test/imageData.npz")
    #r["arr_0"]
    
    #保存
    f = open('E:/test/image_dict.txt','w')
    f.write(str(image_dict))
    f.close()
     
    #读取
    '''
    f = open('E:/test/image_dict.txt','r')
    a = f.read()
    dict_name = eval(a)
    f.close()
    '''
    
    return img_x, img_y, image_dict