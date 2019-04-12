#!/usr/bin/env python
# coding: utf-8
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
import cv2
import tifffile as tiff
from keras.utils import np_utils
import numpy as np
from keras.callbacks import ModelCheckpoint
import random
from keras.optimizers import Adam
import keras.backend as K
import math
from keras import layers
from keras.layers import Lambda
from vgg19 import VGG19
from vgg16 import VGG16
from utils import Unet
from resnet50 import ResNet50
from resnetinception import InceptionResNetV2
from densenet import DenseNet121
from densenet import DenseNet169
from densenet import DenseNet201
def RetModel():
    return [VGG16(input_shape=(256,256,7),classes=2),VGG19(input_shape=(256,256,7),classes=2),
           ResNet50(input_shape=(256, 256, 7), classes=2),InceptionResNetV2(input_shape=(256, 256, 7), classes=2),
           DenseNet121(input_shape=(256, 256, 7),classes=2),DenseNet169(input_shape=(256, 256, 7), classes=2),
           DenseNet201(input_shape=(256, 256, 7),classes=2)]
vgg16net,vgg19net,resnet50,inceptionresnet,densenet121,densenet169,densenet201 = RetModel()
netList=[vgg16net,vgg19net,resnet50,inceptionresnet,densenet121,densenet169,densenet201]
skip_con_List=["vgg","vgg","resnet","inceptionresnet","densenet","densenet","densenet"]
indexList=[18,21,173,779,425,593,705]
backbone_name_List=["vgg16","vgg19","resnet50","inceptionresnetv2","densenet121","densenet169","densenet201"]
DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':            ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19':            ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'), 
    'resnet50':         (141,79,37,4),
    'inceptionresnetv2':    (606,266,16,9),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}#18,21,173,779,425,593,705
import tifffile
imgpath =["result_train/2014tr.png",
         "result_train/2015tr.png",
         "result_train/2016tr.png",
         "result_train/2017tr.png"]
tifpath = ["120041/LC81200412014210LGN00_merge_result.tif",
          "120041/LC81200412015213LGN00_merge_result.tif",
          "120041/LC81200412016232LGN00_merge_result.tif",
          "120041/LC08_L1TP_120041_20170721_20170728_01_T1_merge_result.tif"]
imgSize = cv2.imread(imgpath[0],0)
from keras.utils import np_utils
def getData(maskpath,tifpath):
    tmpData = np.zeros((4,imgSize.shape[0],imgSize.shape[1],7))
    tmpMask = np.zeros((4,imgSize.shape[0],imgSize.shape[1],2))
    for i in range(4):
        tmpData[i,:,:,:]=tifffile.imread(tifpath[i])
        tmpMask[i,:,:,:] = np_utils.to_categorical(cv2.imread(maskpath[i],0),2)
    return tmpData/255,tmpMask

X_train,y_train = getData(imgpath,tifpath)



img_cols=64*4
img_rows=64*4
num_channels=7
num_mask_channels=2
batch_size=2
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, img_rows, img_cols,num_channels))
    y_batch = np.zeros((batch_size, img_rows, img_cols,num_mask_channels))
    X_height = X.shape[1]
    X_width = X.shape[2]
    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, random_height: random_height + img_rows, random_width: random_width + img_cols,:]
        X_batch[i] = np.array(X[random_image,random_height: random_height + img_rows, random_width: random_width + img_cols,:])
    return X_batch, y_batch

def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)
            X_batch[i] = xb
            y_batch[i] = yb
        yield X_batch,y_batch




# In[37]:


def fit(netList,skip_con_List,indexList,backbone_name_List):
    for i in range(len(backbone_name_List)):
        model_name = backbone_name_List[i]
        checkpoint = ModelCheckpoint(model_name +'_model_weight.h5',
                                     save_best_only=True,
                                     save_weights_only=True)
        net = Unet(netList[i],skip_index=DEFAULT_SKIP_CONNECTIONS,
                   skip_con=skip_con_List[i],index=indexList[i],backbone_name=backbone_name_List[i])
        net.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=[dice_coef])
        net.fit_generator(batch_generator(X_train,y_train,batch_size),steps_per_epoch=400,
                               validation_data = batch_generator(X_train,y_train,batch_size),validation_steps=2,
                               callbacks=[checkpoint],epochs=50,verbose=1)
fit(netList,skip_con_List,indexList,backbone_name_List)

