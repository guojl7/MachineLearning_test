# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Created on Thu Apr 19 08:52:30 2018

@author: zy
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

'''
分布式计算
'''

'''
(1)为每个角色添加IP地址和端口，创建worker 
'''

'''定义IP和端口号'''
#指定服务器ip和port
strps_hosts = '192.168.43.205:1234'
#指定两个终端的ip和port
strworker_hosts =  '192.168.43.206:1235,192.168.43.207:1236'

#定义角色名称
strjob_name = 'worker'
task_index = 1
#将字符串转为数组
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})

#创建server
server = tf.train.Server(cluster_spec, job_name = strjob_name, task_index = task_index)

'''
(2) 为ps角色添加等待函数
'''
print('waiting....')
server.join()
	
	

	






