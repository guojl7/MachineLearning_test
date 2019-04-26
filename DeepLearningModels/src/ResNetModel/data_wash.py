# -*- coding: utf-8 -*-
import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


#清除无法读取以及无效的图片，并保存文件的路径以及lable
def data_wash(read_directory_name, target_size = None):
    image_dict = {}
    img_x = [] # this if for store all of the image data
    img_y = []
    image_label = 0;
    image_count = 0;
    
    for image_label_name in os.listdir(read_directory_name):
        image_dict[image_label] = image_label_name #label与类别之间的映射
        image_directory_name = read_directory_name + "/" + image_label_name + "/IMAGES"
    
        for filename in os.listdir(image_directory_name):
            img_path = image_directory_name + "/" + filename

            try:
                img = image.load_img(img_path, target_size)
            except:
                continue
            
            if img.size == (161, 81):
                continue
            
            img_x.append(img_path) #图像路径
            img_y.append(image_label) #图形label
            image_count +=1
            
            if 0 == image_count % 1000:
                print("%s" % img_x[image_count - 1])
        
        image_label += 1
    
    np.savez("E:/raw_data_wash/imageData.npz", img_x, img_y, image_dict)
    #r = np.load("E:/test/imageData.npz")
    #r["arr_0"]
    
    return img_x, img_y, image_dict

if __name__ == '__main__':
    read_directory_name = 'E:/raw_data'
    save_directory_name = 'E:/raw_data_wash'
    img_x, img_y, image_dict = data_wash(read_directory_name)
    
