import tensorflow as tf
import numpy as np
import os
from os import walk
import re
import glob
import nibabel as nib
import sys
import pickle
import time
import dicom
from scipy.io import loadmat
from datetime import datetime

def maxpool2dWrap(x, k=2):
# MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv2dWrap(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv2dTransWrap(x, W, b, shape, strides=[1,2,2,1]):
    x = tf.nn.conv2d_transpose(x, W, shape, strides, padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x

def outWrap(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')
    x = tf.nn.bias_add(x, b)
    return(x)

def uNet(x,weights,biases):
    #L1
    conv0  = conv2dWrap(x, weights['wConv0'], biases['bConv0'])
    conv1  = conv2dWrap(conv0, weights['wConv1'], biases['bConv1'])
    conv2  = conv2dWrap(conv1, weights['wConv2'], biases['bConv2'])
    conv2p  = maxpool2dWrap(conv2)
    #print(conv2p.shape)

    #L2
    conv3  = conv2dWrap(conv2p, weights['wConv3'], biases['bConv3'])
    conv3_new  = conv2dWrap(conv3, weights['wConv3_new'], biases['bConv3_new'])
    conv4  = conv2dWrap(conv3_new, weights['wConv4'], biases['bConv4'])
    conv4p = maxpool2dWrap(conv4)
    #print(conv4p.shape)

    #L3
    conv5  = conv2dWrap(conv4p, weights['wConv5'], biases['bConv5'])
    conv5_new  = conv2dWrap(conv5, weights['wConv5_new'], biases['bConv5_new'])
    conv6  = conv2dWrap(conv5_new, weights['wConv6'], biases['bConv6'])
    conv6p = maxpool2dWrap(conv6)
    #print(conv6p.shape)

    #l4
    conv7  = conv2dWrap(conv6p, weights['wConv7'], biases['bConv7'])
    conv7_new  = conv2dWrap(conv7, weights['wConv7_new'], biases['bConv7_new'])
    conv8  = conv2dWrap(conv7_new, weights['wConv8'], biases['bConv8'])
    conv8p = maxpool2dWrap(conv8)
    #print(conv8p.shape)

    #L5
    conv9       = conv2dWrap(conv8p, weights['wConv9'], biases['bConv9'])
    conv10      = conv2dWrap(conv9, weights['wConv10'], biases['bConv10'])
    conv10_new  = conv2dWrap(conv10, weights['wConv10_new'], biases['bConv10_new'])
    conv10_new2  = conv2dWrap(conv10_new, weights['wConv10_new2'], biases['bConv10_new2'])

    #L4
    convT1  = conv2dTransWrap(conv10_new2,weights['wT1'], biases['bT1'],[1,28,32,256])
    conv11  = conv2dWrap(tf.concat([convT1,conv8],3), weights['wConv11'], biases['bConv11'])
    conv12  = conv2dWrap(conv11, weights['wConv12'], biases['bConv12'])

    #L3
    convT2  = conv2dTransWrap(conv12,weights['wT2'], biases['bT2'],[1,56,64,128])
    conv13  = conv2dWrap(tf.concat([convT2,conv6],3), weights['wConv13'], biases['bConv13'])
    conv14  = conv2dWrap(conv13, weights['wConv14'], biases['bConv14'])

    #L2
    convT3  = conv2dTransWrap(conv14,weights['wT3'], biases['bT3'],[1,112,128,64])
    conv15  = conv2dWrap(tf.concat([convT3,conv4],3), weights['wConv15'], biases['bConv15'])
    conv16  = conv2dWrap(conv15, weights['wConv16'], biases['bConv16'])

    #L1
    convT4  = conv2dTransWrap(conv16,weights['wT4'], biases['bT4'],[1,224,256,32])
    conv17  = conv2dWrap(tf.concat([convT4,conv2],3), weights['wConv17'], biases['bConv17'])
    conv18  = conv2dWrap(conv17, weights['wConv18'], biases['bConv18'])


    convOUT = outWrap(conv18,weights['wOUT'], biases['bOUT'])
    #print(convOUT.shape)
    return convOUT

launch  = os.getcwd()
trainDir = launch+'/chrissy_fdg_machine_learning/'
valDir = launch+'/validationSet/'
trainListMRI = []

for dirpath, dirnames, filenames in walk(trainDir):
    for file in filenames:
        if re.match(r'\d{6}\.hdr',file):
            p = os.path.join(dirpath, file)
            trainListMRI.append(p)

trainListLabel = []
for dirpath, dirnames, filenames in walk(trainDir):
        for file in filenames:
                if re.match(r'\b[A-Z]{5}\d{3}\b.mc.mrcoreg.mean.hdr',file):
                        p = os.path.join(dirpath, file)
                        trainListLabel.append(p)

valListMRI = []
for dirpath, dirnames, filenames in walk(valDir):
    for file in filenames:
        if re.match(r'\d{6}\.hdr',file):
            p = os.path.join(dirpath, file)
            valListMRI.append(p)

valListLabel = []
for dirpath, dirnames, filenames in walk(valDir):
        for file in filenames:
                if re.match(r'\b[A-Z]{5}\d{3}\b.mc.mrcoreg.mean.hdr',file):
                        p = os.path.join(dirpath, file)
                        valListLabel.append(p)


weights = {
'wConv0':  tf.Variable(tf.random_normal([5, 5, 1,  32],0,.0055),       name='wC0'),
'wConv1':  tf.Variable(tf.random_normal([5, 5, 32,  32],0,.0055),       name='wC1'),
'wConv2':  tf.Variable(tf.random_normal([5, 5, 32, 32],0,.0055),       name='wC2'),
'wConv3':  tf.Variable(tf.random_normal([5, 5, 32, 64],0,.0078),      name='wC3'),
'wConv3_new':  tf.Variable(tf.random_normal([5, 5, 64, 64],0,.0078),      name='wC3_new'),
'wConv4':  tf.Variable(tf.random_normal([5, 5, 64, 64],0,.0078),     name='wC4'),
'wConv5':  tf.Variable(tf.random_normal([5, 5, 64, 128],0,.001),     name='wC5'),
'wConv5_new':  tf.Variable(tf.random_normal([5, 5, 128, 128],0,.001),     name='wC5_new'),
'wConv6':  tf.Variable(tf.random_normal([5, 5, 128, 128],0,.001),     name='wC6'),
'wConv7':  tf.Variable(tf.random_normal([5, 5, 128, 256],0,.001),     name='wC7'),
'wConv7_new':  tf.Variable(tf.random_normal([5, 5, 256, 256],0,.001),     name='wC7_new'),
'wConv8':  tf.Variable(tf.random_normal([5, 5, 256, 256],0,.001),     name='wC8'),
'wConv9':  tf.Variable(tf.random_normal([5, 5, 256, 512],0,.001),    name='wC9'),
'wConv10':  tf.Variable(tf.random_normal([5, 5, 512, 512],0,.001),  name='wC10'),
'wConv10_new':  tf.Variable(tf.random_normal([5, 5, 512, 512],0,.001),  name='wC10_new'),
'wConv10_new2':  tf.Variable(tf.random_normal([5, 5, 512, 512],0,.001),  name='wC10_new2'),
'wConv11':  tf.Variable(tf.random_normal([5, 5, 512, 256],0,.001),  name='wC11'),
'wConv12':  tf.Variable(tf.random_normal([5, 5, 256, 256],0,.001),  name='wConv12'),
'wConv13':  tf.Variable(tf.random_normal([5, 5, 256, 128],0,.001),  name='wConv13'),
'wConv14':  tf.Variable(tf.random_normal([5, 5, 128, 128],0,.001),  name='wConv14'),
'wConv15':  tf.Variable(tf.random_normal([5, 5, 128, 64],0,.0078),  name='wConv15'),
'wConv16':  tf.Variable(tf.random_normal([5, 5, 64, 64],0,.0078),  name='wConv16'),
'wConv17':  tf.Variable(tf.random_normal([5, 5, 64, 32],0,.0055),  name='wConv17'),
'wConv18':  tf.Variable(tf.random_normal([5, 5, 32, 32],0,.0055),  name='wConv18'),
'wT1':     tf.Variable(tf.random_normal([2, 2, 256, 512],0,.0055),  name='wT1'),
'wT2':     tf.Variable(tf.random_normal([2, 2, 128, 256],0,.001),  name='wT2'),
'wT3':     tf.Variable(tf.random_normal([2, 2, 64, 128],0,.0078),  name='wT3'),
'wT4':     tf.Variable(tf.random_normal([2, 2, 32, 64],0,.0055),  name='wT4'),
'wOUT'  :  tf.Variable(tf.random_normal([1, 1, 32, 1],0,.001),       name='wOUT')
}

biases = {
'bConv0': tf.Variable(tf.random_normal([32],0,0.001),   name='bConv0'),
'bConv1': tf.Variable(tf.zeros([32], tf.float32),   name='bConv1'),
'bConv2': tf.Variable(tf.random_normal([32],0,0.001),  name='bConv2'),
'bConv3': tf.Variable(tf.random_normal([64],0,0.001),  name='bConv3'),
'bConv3_new': tf.Variable(tf.random_normal([64],0,0.001),  name='bConv3_new'),
'bConv4': tf.Variable(tf.random_normal([64],0,0.001), name='bConv4'),
'bConv5': tf.Variable(tf.random_normal([128],0,0.001),  name='bConv5'),
'bConv5_new': tf.Variable(tf.random_normal([128],0,0.001),  name='bConv5_new'),
'bConv6': tf.Variable(tf.random_normal([128],0,0.001),  name='bConv6'),
'bConv7': tf.Variable(tf.random_normal([256],0,0.001),  name='bConv7'),
'bConv7_new': tf.Variable(tf.random_normal([256],0,0.001),  name='bConv7_new'),
'bConv8': tf.Variable(tf.random_normal([256],0,0.001),  name='bConv8'),
'bConv9': tf.Variable(tf.random_normal([512],0,0.001),  name='bConv9'),
'bConv10': tf.Variable(tf.random_normal([512],0,0.001),  name='bConv10'),
'bConv10_new': tf.Variable(tf.random_normal([512],0,0.001),  name='bConv10_new'),
'bConv10_new2': tf.Variable(tf.random_normal([512],0,0.001),  name='bConv10_new2'),
'bConv11': tf.Variable(tf.random_normal([256],0,0.001),  name='bConv11'),
'bConv12': tf.Variable(tf.random_normal([256],0,0.001),  name='bConv12'),
'bConv13': tf.Variable(tf.random_normal([128],0,0.001),  name='bConv13'),
'bConv14': tf.Variable(tf.random_normal([128],0,0.001),  name='bConv14'),
'bConv15': tf.Variable(tf.random_normal([64],0,0.001),  name='bConv15'),
'bConv16': tf.Variable(tf.random_normal([64],0,0.001),  name='bConv16'),
'bConv17': tf.Variable(tf.random_normal([32],0,0.001),  name='bConv17'),
'bConv18': tf.Variable(tf.random_normal([32],0,0.001),  name='bConv18'),
'bT1': tf.Variable(tf.random_normal([256],0,0.001),  name='bT1'),
'bT2': tf.Variable(tf.random_normal([128],0,0.001),  name='bT2'),
'bT3': tf.Variable(tf.random_normal([64],0,0.001),  name='bT3'),
'bT4': tf.Variable(tf.random_normal([32],0,0.001),  name='bT4'),
'bOUT': tf.Variable(tf.zeros([1]), name='bOUT')}

NUM_EPOCHS=100
KEEP_PROB=1
LEARNING_RATE=.00001
VALIDATION_CUTOFF=0.005

x = tf.placeholder(tf.float32, name='x')  #input
y = tf.placeholder(tf.float32, name='y') #'labels'

def train_uNet(x):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainLosses=[]
    trainTimes=[]
    valLosses=[]
    print('Beginning Training!')
    print(NUM_EPOCHS)
    pred = uNet(x, weights, biases)
    #The cost function should be REDUCE_SUM not REDUCE_MEAN!
    #cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=tf.reshape(pred,[-1])))
    cost = tf.reduce_sum(tf.losses.mean_squared_error(y,pred))

#    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(y,shape=(256**2, 2)),
#    logits=tf.reshape(pred,shape=(256**2, 2)))) #256 is task-specific and can be dynamically detemrined

    regularizer= (0.1*(tf.reduce_mean(tf.nn.l2_loss(weights['wConv1'])))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv2']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv3']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv4']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv5']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv6']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv7']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv8']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv9']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv10']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv11']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv12']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv13']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv14']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv15']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv16']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv17']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wConv18']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wT1']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wT2']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wT3']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wT4']))
                 +0.1*tf.reduce_mean(tf.nn.l2_loss(weights['wOUT'])))
    cost = cost + regularizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    #SUMMARIES

#    TRAINING_COST = tf.placeholder(tf.float32)
#    TRAINING_SUMMARY = tf.summary.scalar('Training', TRAINING_COST)
#
#    VALIDATION_COST = tf.placeholder(tf.float32)
#    VALIDATION_SUMMARY = tf.summary.scalar('Validation', VALIDATION_COST)
#    #VALIDATION_COST = tf.summary.scalar("validation cost", cost)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Beginning Session!')
        writer  =  tf.summary.FileWriter ( './graphs' ,  sess.graph)
        for epoch in range(NUM_EPOCHS):
                predStack=[] #to get the mean dice coefficient after each epoch
                roiStack=[]  #...^
                t0=time.time()
                print('Epoch #:')
                print(epoch)
                epochLossTrain=0
                counter=0
                for pid in trainListMRI:
                    #for pid in range(0, 1):
		    print(pid)
                    data=nib.load(trainListMRI[counter])
                    data=data.get_data()
                    #print(data)
                    mri=data
                    #mri=mri.dataobj[:,:,:,0:2]
                    print(mri.shape)
                    labels=nib.load(trainListLabel[counter])
                    
                    labels=labels.get_data()
                    print(trainListLabel[counter])
                    print(trainListMRI[counter])
                    #labels=labels['roi_mat']
                    #mlabels=1-labels
                    mri  =  np.expand_dims(mri, 0)
                    mri  =  np.expand_dims(mri, -1)
                    for zSlice in range(mri.shape[3]):
                        mask=labels[:,:,zSlice] #1-hot tumor
                        mask=np.expand_dims(mask, 0)
                        mask=np.expand_dims(mask, -1)
                        #mask[mask==0]=0.01
                        #mask2=np.ndarray.flatten(mlabels[:,:,zSlice]) #1-hot normal tissue
                        #mask=np.concatenate((mask,mask2),axis=-1)
                        _, c = sess.run([optimizer, cost], feed_dict = {x: mri[:,:,:,zSlice,:],y: mask})
                        epochLossTrain += c
                        #print(c)
                    counter+=1
                #tSummary = sess.run(TRAINING_SUMMARY, feed_dict={TRAINING_COST: epochLossTrain/len(trainList)})
                #writer.add_summary(tSummary,epoch)
                trainLosses.append(epochLossTrain)
                epochLossVal=0
                t1=time.time()
                t=t1-t0
                trainTimes.append(t)
                print('EPOCH #:', epoch, 'is complete')
                print('It took this much time:', t)
                print('Running validation for', epoch)
                save_path=saver.save(sess, './PseudoOUT/model')

                counter=0
                for pid in valListMRI: #validation stuff
                    #for pid in range(0, 1):
		    #print(pid)
                    #k
                    data=nib.load(valListMRI[counter])
		    mri = data.get_data()
                    #mri=data['data']
                    #mri=mri[:,:,:,0:2]
                    labels=nib.load(valListLabel[counter])
                    labels=labels.get_data()
		    #labels=labels['roi_mat']
                    #mlabels=1-labels
                    mri  =  np.expand_dims(mri, 0)
		    mri = np.expand_dims(mri, -1)
                    for zSlice in range(mri.shape[3]):
                        mask=labels[:,:,zSlice] #1-hot tumor
                        mask=np.expand_dims(mask, 0)
                        mask=np.expand_dims(mask, -1)
			#mask=np.ndarray.flatten(labels[:,:,zSlice]) #1-hot tumor
                        #mask[mask==0]=0.01
                        #mask2=np.ndarray.flatten(mlabels[:,:,zSlice]) #1-hot normal tissue
                        #mask=np.concatenate((mask,mask2),axis=-1)
                        c = sess.run( cost, feed_dict = {x: mri[:,:,:,zSlice,:],y: mask})
                        epochLossVal += c
                    counter+=1
#                vSummary=sess.run(VALIDATION_SUMMARY, feed_dict={VALIDATION_COST: mean_validation_loss})
#                writer.add_summary(vSummary,epoch)
                valLosses.append(epochLossVal)
                save_path=saver.save(sess, './PseudoOUT/model')

                ###################
                ####PICKLE DUMP####
                ###################
                with open('./PseudoOUT/dump.pickle', 'w+') as f:
                    pickle.dump([valLosses,  trainLosses], f)

    print('EPOCHS COMPLETE-- SAVING')
    parameters = weights.copy()
    parameters.update(biases)

train_uNet(x)
