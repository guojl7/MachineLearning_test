# -*- coding:UTF-8 -*-
import os


def incode():
    infile = open("guojl.txt","rb")
    infile1 = open("Desktop.rar","rb")
    outfile = open("guojl_outfile.txt","wb")
    
    infileContent = []
    infileContent1 = []
    
    while 1:
        c = infile.read(1)
        if not c:
            infile.close()
            break
        infileContent.append(c)
        
    while 1:
        c = infile1.read(1)
        if not c:
            infile1.close()
            break
        infileContent1.append(c)
        
    infileContent1.reverse()
    
    lengthOfInfile = len(infileContent)
    lengthOfInfile1 = len(infileContent1)
    
    pos = 0;
    i = 0
    while(i < lengthOfInfile - 3):
        if ('0x1d' == hex(ord(infileContent[i]))) & \
           ('0xd1' == hex(ord(infileContent[i + 1]))) & \
           ('0xf2' == hex(ord(infileContent[i + 2]))) & \
           ('0x64' == hex(ord(infileContent[i + 3]))) & \
           ('0x03' == hex(ord(infileContent[i + 3]))) & \
           ('0xf0' == hex(ord(infileContent[i + 3]))):
            i = i + 8
            for count in range(1000):
                infileContent[i + count] = infileContent1[pos]
                i = i + 1
                pos = pos + 1
                if pos >= lengthOfInfile1:
                    break
        else:
            i = i + 1
            
        if pos >= lengthOfInfile1:
            break
    
    for byte in infileContent:
        outfile.write(byte)
         
    outfile.close()
    
def decode():
    infile = open("guojl_outfile.txt","rb")
    outfile = open("Desktop1.rar","wb")
    
    lengthOfoutfile = 100
    outfileContent = []
    infileContent = []
    
    while 1:
        c = infile.read(1)
        if not c:
            infile.close()
            break
        infileContent.append(c)
    
    lengthOfInfile = len(infileContent)
    
    pos = 0;
    i = 0
    while(i < lengthOfInfile - 3):
        if ('0x1d' == hex(ord(infileContent[i]))) & \
           ('0xd1' == hex(ord(infileContent[i + 1]))) & \
           ('0xf2' == hex(ord(infileContent[i + 2]))) & \
           ('0x64' == hex(ord(infileContent[i + 3]))) & \
           ('0x03' == hex(ord(infileContent[i + 3]))) & \
           ('0xf0' == hex(ord(infileContent[i + 3]))):
            i = i + 8
            for count in range(1000):
                outfileContent[pos] = infileContent[i]
                i = i + 1
                pos = pos + 1
                if i >= lengthOfInfile:
                    print("out of length")
                    break
                if pos >= lengthOfoutfile:
                    break
        else:
            i = i + 1
            
        if pos >= lengthOfoutfile:
            break
    
    for byte in outfileContent:
        outfile.write(byte)
         
    outfile.close()
    
import random
import numpy as np

if __name__ == '__main__':
    rand_data = np.random.randint(0,99)
    if rand_data in range(40, 59):
        print("helle word")
    else:
        print(rand_data)
    
    rand_data = np.random.normal(2, 5, 1)
    print("期望为2, 方差为5高斯随机数："+str(rand_data))
    
    rand_data = np.random.rand()*4-2
    print("-2到2均匀分布的随机数："+str(rand_data))
        
        
    #incode()
    #decode()