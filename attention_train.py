#!/usr/bin/env python
# coding: utf-8
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import dot
from keras.layers import MaxPool2D,Reshape
import tensorflow as tf
from keras.layers import ConvLSTM2D
from keras.layers import Dense
import math
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import Concatenate
from keras.layers import Activation,Add
import keras.backend as K
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from vgg19 import VGG19
from vgg16 import VGG16
from resnet50 import ResNet50
from resnetinception import InceptionResNetV2
from densenet import DenseNet121
from densenet import DenseNet169
from densenet import DenseNet201
from utils import Unet
from tensorflow.python.framework import ops
from numpy import ndarray
from types import FunctionType
from typing import Tuple,List
import tifffile
import cv2
import random
import numpy as np
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
def getData(maskpath:List,tifpath:List)->Tuple[ndarray,ndarray]:
    '''
    To get training data and training mask
    args:
        maskpath:the path of mask file
        tifpath:the path of tif file
    returns:
        tmpData:trainng data with shape(1,imgSize.shape[0],imgsize.shape[1],7)
        tmpMask:training mask with shape(1,imgSize.shape[0],imgsize.shape[1],2)
    '''
    assert isinstance(maskpath,list) and isinstance(tifpath,list)
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
def dice_coef(y_true:ops.Tensor, y_pred:ops.Tensor)->ops.Tensor:
    '''
    To evacuate training model
    args:
        y_true:real mask
        y_pred:predicted mask
    returns:
        :IOU of two mask
    '''
    assert isinstance(y_true,ops.Tensor) and isinstance(y_pred,ops.Tensor)
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
def flip_axis(x:ndarray, axis:int)->ndarray:
    '''
    To flip certain axis for given ndarray
    args:
        x:the ndarray we should flip
        axis:which axis we should flip
    returns:
        :fliped ndarray
    '''
    assert isinstance(x,ndarray) and isinstance(axis,int)
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def form_batch(X:ndarray, y:ndarray, batch_size:int)->Tuple[ndarray,ndarray]:
    '''
    To get a batch training data
    args:
        X:whole training data
        y:whole training mask
        batch_size: training batch size
    return:
        X_batch:single batch with shape [batch_size,img_rows,img_cols,num_channels]
        y_batch:single batch with shape [batch_size,img_rows,img_cols,num_mask_channels]
    '''
    assert isinstance(X,ndarray) and isinstance(y,ndarray) and isinstance(batch_size,int)
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

def batch_generator(X:ndarray, y:ndarray, batch_size:int, horizontal_flip:bool=False, vertical_flip:bool=False, swap_axis:bool=False)->Tuple[ndarray,ndarray]:
    '''
    To get a batch training data
    args:
        X:whole training data
        y:whole training mask
        batch_size: training batch size
        horizontal_flip:the way to flip data horizontally
        vertical_flip:the way to flip data vertically
        swap_axis:the way to swap axis
    return:
        X_batch:single batch with shape [batch_size,img_rows,img_cols,num_channels]
        y_batch:single batch with shape [batch_size,img_rows,img_cols,num_mask_channels]
    '''
    assert isinstance(X,ndarray) and isinstance(y,ndarray) and isinstance(batch_size,int)
    assert isinstance(horizontal_flip,bool) and isinstance(horizontal_flip,bool) and isinstance(horizontal_flip,bool)
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
        yield ([X_batch,X_batch,X_batch,X_batch,X_batch,X_batch,X_batch],y_batch)


# In[24]:


def RetModel()->Tuple[Model,Model,Model,Model,Model,Model,Model]:
    '''
    To get all training model
    returns:
        load vgg,resnet and densenet model
    '''
    return [VGG16(input_shape=(256,256,7),classes=2),VGG19(input_shape=(256,256,7),classes=2),
           ResNet50(input_shape=(256, 256, 7), classes=2),InceptionResNetV2(input_shape=(256, 256, 7), classes=2),
           DenseNet121(input_shape=(256, 256, 7),classes=2),DenseNet169(input_shape=(256, 256, 7), classes=2),
           DenseNet201(input_shape=(256, 256, 7),classes=2)]
DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':            ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19':            ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'), 
    'resnet50':         (141,79,37,4),
    'inceptionresnetv2':    (606,266,16,9),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}
vgg16net,vgg19net,resnet50,inceptionresnet,densenet121,densenet169,densenet201 = RetModel()
netList=[vgg16net,vgg19net,resnet50,inceptionresnet,densenet121,densenet169,densenet201]
skip_con_List=["vgg","vgg","resnet","inceptionresnet","densenet","densenet","densenet"]
indexList=[18,21,173,779,425,593,705]
backbone_name_List=["vgg16","vgg19","resnet50","inceptionresnetv2","densenet121","densenet169","densenet201"]
def keras_matmul(xylist:List)->ops.Tensor:
    '''
    To multiply two tensor
    args:
        xylist:list includes two tensor
    return:
        :result of two tensor's mat
    '''
    assert isinstance(xylist,List)
    return K.dot(xylist[0],xylist[1])
def attention_layer(channels:int)->FunctionType:
    '''
    Use keras stype to implemment attention layer 
    args:
        channels:how many channels we should output
    return:
        a function implemmented self-attention
    '''
    assert isinstance(channels,int)
    def self_attention(x:ops.Tensor)->ops.Tensor:
        '''
        To implemment self-attention
        args:
            x:input tensor for operation
        return:
            x:data after self-attention layers
        '''
        assert isinstance(x,ops.Tensor)
        f = Conv2D(channels//8,kernel_size=1,strides=1,padding="same")(x)
        g = Conv2D(channels//8,kernel_size=1,strides=1,padding="same")(x)
        h = Conv2D(channels,kernel_size=1,strides=1,padding="same")(x)
        reg = Reshape((int(g.shape[-1]),int(g.shape[1])*int(g.shape[2])))(g)
        regf = Reshape((int(f.shape[1])*int(f.shape[2]),int(f.shape[-1])))(f)
        s = dot([reg,regf],axes=[1,2])
        beta = Activation("softmax")(s)
        regh = Reshape(((int(h.shape[-1]),int(h.shape[1])*int(h.shape[2]))))(h)
        o = dot([beta,regh],axes=[1,2])
        o = Reshape((x.shape[1],x.shape[2],x.shape[3]))(x)
        x = 0.2*o +x
        return x
    return self_attention
def keras_mean(x:ops.Tensor)->ops.Tensor:
    '''
    To get average of tensor
    args:
        x:tensor data
    return:
        :average of a tensor
    '''
    assert isinstance(x,ops.Tensor)
    return K.mean(x,axis=-1, keepdims=True)
def keras_max(x:ops.Tensor)->ops.Tensor:
    '''
    To get maximum of tensor
    args:
        x:tensor data
    return:
        :maximum of a tensor
    '''
    assert isinstance(x,ops.Tensor)
    return K.max(x,axis=-1,keepdims=True)
def attention_block(channels:int)->FunctionType:
    '''
    Use keras stype to implemment attention layer 
    args:
        channels:how many channels we should output
    return:
        a function implemmented self-attention
    '''
    def channel_spatial_attention(x:ops.Tensor)->ops.Tensor:
        '''
        To implemment self-attention
        args:
            x:input tensor for operation
        return:
            x:data after self-attention layers
        '''
        xp = GlobalAveragePooling2D()(x)
        xp = Dense(channels//10,activation='relu')(xp)
        xp = Dense(channels)(xp)
        xm = GlobalMaxPooling2D()(x)
        xm = Dense(channels//10,activation='relu')(xm)
        xm = Dense(channels)(xm)
        ad = Add()([xp,xm])
        scale = Reshape((1,1,channels))(ad)
        scale = Activation("sigmoid")(scale)
        x = x*scale
        x_avg = Lambda(keras_mean)(x)
        x_max = Lambda(keras_max)(x)
        scale = Concatenate(axis=-1)([x_avg,x_max])
        scale = Conv2D(1,kernel_size=3,padding='same',strides=1,activation='sigmoid')(scale)
        x = x*scale
        return x
    return channel_spatial_attention
def model(baseModel_list:List)->Tuple[List,ops.Tensor,List]:
    '''
    To get training model with attention layer and attention block,we use Lambda tranform tensorflow tensor to keras tensor
    args:
        baseModel_list:ensambling model list
    return:
        base_model_input:each base model input tensor
        x:tensor after all keras layer
    '''
    assert isinstance(baseModel_list,List)
    base_model_output = []
    base_model_input = []
    res_net_list=[]
    for i in range(len(baseModel_list)):
        model_name = backbone_name_List[i]+'_model_weight.h5'
        net = Unet(netList[i],skip_index=DEFAULT_SKIP_CONNECTIONS,
                   skip_con=skip_con_List[i],index=indexList[i],backbone_name=backbone_name_List[i])
        net.load_weights(model_name)
        res_net_list.append(net)
        base_model_input.append(net.input)
        base_model_output.append(net.output)
    conc = Concatenate(axis=-1)(base_model_output)
    x = Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(base_model_output[0])
    x = Reshape((1,int(x.shape[1]),int(x.shape[2]),int(x.shape[3])))(x)
    #x = ConvLSTM2D(64,kernel_size=3,padding='same',strides=1,return_sequences=False)(x)
    x = Lambda(attention_layer(128))(x)
    x = Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(x)
    x = Reshape((1,int(x.shape[1]),int(x.shape[2]),int(x.shape[3])))(x)
    #x = ConvLSTM2D(64,kernel_size=3,padding='same',strides=1,return_sequences=False)(x)
    x = Lambda(attention_block(64))(x)
    x = Conv2D(2,kernel_size=1,strides=1,padding='same',activation='softmax')(x)
    return base_model_input,x,res_net_list
base_in_list,resX,res_list= model(netList)
deepModel = Model(base_in_list,resX)
for net in res_list:
    for layer in net.layers:
        layer.trainable=False
deepModel.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=['accuracy'])
callbacks_list = [ModelCheckpoint("attentionPlusSevenUnet.h5",save_best_only=True,save_weights_only=True),
                  ReduceLROnPlateau(monitor='val_loss',patience=10)]
deepModel.fit_generator(batch_generator(X_train,y_train,batch_size=batch_size),steps_per_epoch=200,epochs=50,
                 validation_data=batch_generator(X_train,y_train,batch_size=batch_size),validation_steps=2,verbose=1,
                 callbacks=callbacks_list)

