#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:19:38 2018

@author: hans
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gc
import xml.etree.ElementTree as etxml
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import cv2
import ssd300
import arg_parsing
FLAGS = arg_parsing.parser.parse_args()

'''
SSD检测
'''
def testing():
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        ssd_model = ssd300.SSD300(sess,False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists(FLAGS.model_dir+'session.ckpt.index') :
            saver.restore(sess, FLAGS.model_dir+'session.ckpt')
            img_path, image_data, actual,file_list = get_traindata(1, False)
            pred_class, pred_class_val, pred_location = ssd_model.run(image_data,None)
            print('file_list:' + str(file_list))
            image = cv2.imread(img_path)
            height = image.shape[0]
            width = image.shape[1]
            for index, act in zip(range(len(image_data)), actual):
                for a in act :
                    print('【img-'+str(index)+' actual】:' + str(a))
                print('pred_class:' + str(pred_class[index]))
                print('pred_class_val:' + str(pred_class_val[index]))
                print('pred_location:' + str(pred_location[index]))
                for i in range(len(pred_location[index])):
                    w = int(pred_location[index][i].tolist()[2]*width)
                    h = int(pred_location[index][i].tolist()[3]*height)
                    x = max(0,int(pred_location[index][i].tolist()[0]*width-w/2))
                    y = max(0,int(pred_location[index][i].tolist()[1]*height-h/2))
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.putText(image, lable_arr[pred_class[index][i]], (x+3, y+10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
        else:
            print('No Data Exists!')
        sess.close()
    
    cv2.imshow("image",image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
'''
SSD训练
'''
def training():
    batch_size = FLAGS.batch_size
    running_count = 0
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        ssd_model = ssd300.SSD300(sess,True)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        if os.path.exists(FLAGS.model_dir+'session.ckpt.index') :
            print('\nStart Restore')
            saver.restore(sess, FLAGS.model_dir+'session.ckpt')
            print('\nEnd Restore')
         
        print('\nStart Training')
        min_loss_location = 100000.
        min_loss_class = 100000.
        while((min_loss_location + min_loss_class) > 0.001 and running_count < FLAGS.max_steps):
            running_count += 1
            
            train_data, actual_data,_ = get_traindata(batch_size)
            if len(train_data) > 0:
                loss_all,loss_class,loss_location,pred_class,pred_location = ssd_model.run(train_data, actual_data)
                l = np.sum(loss_location)
                c = np.sum(loss_class)
                if min_loss_location > l:
                    min_loss_location = l
                if min_loss_class > c:
                    min_loss_class = c
                if running_count % FLAGS.log_frequency == 0:
#                    print('Step: %d | Loss All: %.4f(min) %.4f | Location: %.4f | Class: %.4f | pred_class: %.4f(all) %.4f(max) %.4f(min) | pred_location: %.4f(all) %.4f(max) %.4f(min)' 
#                          %(running_count, min_loss_location + min_loss_class, loss_all, np.sum(loss_location), np.sum(loss_class), np.sum(pred_class), np.amax(pred_class), np.min(pred_class), np.sum(pred_location), np.amax(pred_location), np.min(pred_location)))
                     print('Step: %d | Loss All: %.4f(min) %.4f | Location: %.4f | Class: %.4f' 
                          %(running_count, min_loss_location + min_loss_class, loss_all, np.sum(loss_location), np.sum(loss_class)))
                
                # 定期保存ckpt
                if running_count % 1000 == 0:
                    saver.save(sess, FLAGS.model_dir+'session.ckpt')
                    print('session.ckpt has been saved.')
                    gc.collect()
            else:
                print('No Data Exists!')
                break
            
        saver.save(sess, FLAGS.model_dir+'session.ckpt')
        sess.close()
        gc.collect()
            
    print('End Training')
    
'''
获取voc2007训练图片数据
train_data：训练批次图像，格式[None,width,height,3]
actual_data：图像标注数据，格式[None,[None,center_x,center_y,width,height,lable]]
'''
file_name_list = os.listdir(FLAGS.dataset_dir + 'JPEGImages/')
lable_arr = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
# 图像白化，格式:[R,G,B]
whitened_RGB_mean = arg_parsing.MEAN
def get_traindata(batch_size, istraining=True):
    def get_actual_data_from_xml(xml_path):
        actual_item = []
        try:
            annotation_node = etxml.parse(xml_path).getroot()
            img_width =  float(annotation_node.find('size').find('width').text.strip())
            img_height = float(annotation_node.find('size').find('height').text.strip())
            object_node_list = annotation_node.findall('object')       
            for obj_node in object_node_list:                       
                lable = lable_arr.index(obj_node.find('name').text.strip())
                bndbox = obj_node.find('bndbox')
                x_min = float(bndbox.find('xmin').text.strip())
                y_min = float(bndbox.find('ymin').text.strip())
                x_max = float(bndbox.find('xmax').text.strip())
                y_max = float(bndbox.find('ymax').text.strip())
                # 位置数据用比例来表示，格式[center_x,center_y,width,height,lable]
                actual_item.append([((x_min + x_max)/2/img_width), ((y_min + y_max)/2/img_height), ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
            return actual_item  
        except:
            return None
        
    train_data = []
    actual_data = []
    
    file_list = random.sample(file_name_list, batch_size)
    
    for f_name in file_list :
        img_path = FLAGS.dataset_dir + 'JPEGImages/' + f_name
        xml_path = FLAGS.dataset_dir + 'Annotations/' + f_name.replace('.jpg','.xml')
        if os.path.splitext(img_path)[1].lower() == '.jpg' :
            actual_item = get_actual_data_from_xml(xml_path)
            if actual_item != None :
                actual_data.append(actual_item)
            else :
                print('Error : '+xml_path)
                continue
            ori_img = skimage.io.imread(img_path)
            img = skimage.transform.resize(ori_img, (arg_parsing.IMAGE_RESIZE_SHAPE, arg_parsing.IMAGE_RESIZE_SHAPE))
            # 图像白化预处理
            img = img - whitened_RGB_mean
            train_data.append(img)
    if istraining:
        return train_data, actual_data,file_list
    else:
        return img_path, train_data, actual_data,file_list

def printInfo():
    print('-------------------------')
    print('Initial learning rate: %f' %FLAGS.lr)
    print('debug: %s' %FLAGS.debug)
    print('Dataset dir: %s' %FLAGS.dataset_dir)
    print('Model dir: %s' %FLAGS.model_dir)
    if FLAGS.finetune:
        print('Finetune dir: %s' %FLAGS.finetune)
    print('Batch size: %d' %FLAGS.batch_size)
    print('Log frequency: %d' %FLAGS.log_frequency)
    print('Max steps: %d' %FLAGS.max_steps)
    if FLAGS.job_name:
        print('\nDistuibution info: ')
        print('Issync: %s' %FLAGS.issync)
        print('PS HOSTS: %s' %arg_parsing.PS_HOSTS)
        print('WORKER HOSTS: %s' %arg_parsing.WORKER_HOSTS)
    print('-------------------------')
 
'''
主程序入口
'''
if __name__ == '__main__':
    print('\nStart Running')
    if FLAGS.job_name:
        printInfo()
#        train_dis()
    else:
        if (FLAGS.mode == 'testing'):
            testing()
        else:
            printInfo()
            training()
    print('\nEnd Running')