from datetime import datetime
import math
import time
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn import preprocessing
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 50
num_batches = 20000
image_size = 224


     
def load_Img(imgDir):
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    imgNum = len(imgs)
    data = np.empty((imgNum,image_size,image_size,3),dtype="float32")
    for i in range (imgNum):
        img = Image.open(imgDir+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        arr = cv2.resize(arr,(image_size,image_size))
        if len(arr.shape) == 2:
            temp = np.empty((image_size,image_size,3))
            temp[:,:,0] = arr
            temp[:,:,1] = arr
            temp[:,:,2] = arr
            arr = temp
        data[i,:,:,:] = arr
    return data        

def make_label(labelFile):
    label_list = pd.read_csv(labelFile,sep = '\t',header = None) 
    label_list = label_list.sort_values(by=0)
    le = preprocessing.LabelEncoder()
    for item in [1]:
        label_list[item] = le.fit_transform(label_list[item])     
    label = label_list[1].values
    onehot = preprocessing.OneHotEncoder(sparse = False)
    label_onehot = onehot.fit_transform(np.mat(label).T)
    return label_onehot

def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())



imgDir = '/Users/zhuxiaoxiansheng/Desktop/DatasetA_train_20180813/train'
labelFile = '/Users/zhuxiaoxiansheng/Desktop/DatasetA_train_20180813/train.txt'

data = load_Img(imgDir)
data = data/255.
label = make_label(labelFile)
print(data.shape,label.shape)

traindata,testdata,trainlabel,testlabel = train_test_split(data,label,test_size=3000,random_state = 2018)   
print(traindata.shape,testdata.shape)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
my_graph = tf.Graph()
sess = tf.InteractiveSession(graph=my_graph,config=config)

with my_graph.as_default():
    x = tf.placeholder(tf.float32,[None,image_size*image_size*3])
    x = tf.reshape(x,[-1,image_size,image_size,3])
    y = tf.placeholder(tf.float32,[None,190])
    
    #conv1
    c1_kernel = tf.get_variable('weights1',
                                shape=[11,11,3,64],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    c1_kernel = tf.Variable(tf.truncated_normal([11,11,3,64],dtype = tf.float32,stddev = 1e-1),name = 'weights')
    c1_conv = tf.nn.conv2d(x,c1_kernel,[1,4,4,1],padding = 'SAME')
    c1_biases = tf.Variable(tf.constant(0.0,shape=[64],dtype = tf.float32),trainable = True,name='biases')
    c1_bias = tf.nn.bias_add(c1_conv,c1_biases)
    conv1 = tf.nn.relu(c1_bias,name = 'conv1')
    print_activations(conv1)
    
    #pool1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
    print_activations(pool1)
    
    #conv2
    c2_kernel = tf.get_variable('weights2',
                                shape=[5,5,64,192],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    c2_kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
    c2_conv = tf.nn.conv2d(pool1,c2_kernel,[1,1,1,1],padding='SAME')
    c2_biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
    c2_bias = tf.nn.bias_add(c2_conv,c2_biases)
    conv2 = tf.nn.relu(c2_bias,name='conv2')
    print_activations(conv2)
       
    #pool2
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)
    
    #conv3
    c3_kernel = tf.get_variable('weights3',
                                shape=[3,3,192,384],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    c3_kernel = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
    c3_conv = tf.nn.conv2d(pool2,c3_kernel,[1,1,1,1],padding='SAME')
    c3_biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable = True,name='biases')
    c3_bias = tf.nn.bias_add(c3_conv,c3_biases)
    conv3 = tf.nn.relu(c3_bias,name='conv3')
    print_activations(conv3)

    # conv4
    c4_kernel = tf.get_variable('weights4',
                                shape=[3, 3, 384, 256],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    c4_kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],dtype=tf.float32,stddev=1e-1), name='weights')
    c4_conv = tf.nn.conv2d(conv3, c4_kernel, [1, 1, 1, 1], padding='SAME')
    c4_biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
    c4_bias = tf.nn.bias_add(c4_conv, c4_biases)
    conv4 = tf.nn.relu(c4_bias, name='conv4')
    print_activations(conv4)

    # conv5
    c5_kernel = tf.get_variable('weights5',
                                shape=[3, 3, 256, 256],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    c5_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],dtype=tf.float32,stddev=1e-1), name='weights')
    c5_conv = tf.nn.conv2d(conv4, c5_kernel, [1, 1, 1, 1], padding='SAME')
    c5_biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
    c5_bias = tf.nn.bias_add(c5_conv, c5_biases)
    conv5 = tf.nn.relu(c5_bias, name='conv5')
    print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv4,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool5')
    print_activations(pool5)
    
    # flat1
    f1_weight = tf.get_variable('weights6',
                                shape=[6*6*256,4096],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
#    f1_weight = tf.Variable(tf.truncated_normal([3*3*256,2048],stddev=0.01))
    f1_biases = tf.Variable(tf.constant(0.1,shape=[4096]))
    pool5_flat = tf.reshape(pool5,[-1,6*6*256])
    flat1 = tf.nn.relu(tf.matmul(pool5_flat,f1_weight)+f1_biases,name = 'flat1')
    print_activations(flat1)
    
    # flat2
    f2_weight = tf.get_variable('weights7',
                                shape=[4096,4096],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
#    f2_weight = tf.Variable(tf.truncated_normal([2048,2048],stddev=0.01))
    f2_biases = tf.Variable(tf.constant(0.1,shape=[4096]))
    flat2 = tf.nn.relu(tf.matmul(flat1,f2_weight)+f2_biases,name = 'flat2')
    print_activations(flat2)
        
    # output
    op_weight = tf.get_variable('weights8',
                                shape=[4096,190],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
#    op_weight = tf.Variable(tf.truncated_normal([2048,190],stddev=0.1))
    op_biases = tf.Variable(tf.constant(0.1,shape=[190]))
    y_conv = tf.nn.softmax(tf.matmul(flat2,op_weight)+op_biases,name = 'output')
    print_activations(y_conv)
        

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    sess.run(tf.global_variables_initializer())

    for i in range(num_batches):
        rand_index = np.random.choice(35221,size=(batch_size))
        train_step.run(feed_dict={x:traindata[rand_index],y:trainlabel[rand_index]})
         
        if i%100 == 0:
            rand_index = np.random.choice(35221,size=(3000))
            train_accuracy = accuracy.eval(feed_dict={x:traindata[rand_index],y:trainlabel[rand_index]})
            print('step %d, training accuracy %g'%(i,train_accuracy))
            print('step %d, training accuracy %g'%(cross_entropy.eval(feed_dict={x:traindata[rand_index],y:trainlabel[rand_index]})))
    print("test accuracy %g"%accuracy.eval(feed_dict={x:testdata,y:testlabel}))
    



