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
from keras.applications import VGG19
from keras.applications import DenseNet121
from keras.applications import InceptionResNetV2 as ri
from keras.applications import InceptionV3 as iv
from keras import layers
from keras.layers import Lambda
from keras.applications import ResNet50 as res
from typing import Tuple,List
from tensorflow.python.framwork import ops
def dense_block(x:ops.Tensor, blocks:int, name:str)->ops.Tensor:
    '''
    To implemment dense block
    args:
       x:operate tensor
       block:number of operation
       name:block name
    returns:
       x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(blocks,int) and isinstance(name,str)
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x:ops.Tensor, reduction:float, name:str)->ops.Tensor:
    '''
    To implemmnet transition block
    args:
        x:operate tensor
        reduction:reduction rate
        name:block name
    returns:
        x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(reduction,float) and isinstance(name,str)
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x:ops.Tensor, growth_rate:int, name:str)->ops.Tensor:
    '''
    To wrap a block with conv2d,batchnormalization and concatenate
    args:
        x:operate tensor
        growth_rate:dialate rate
        name:block name
    returns:
        x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(growth_rate,int) and isinstance(name,str)
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks:list,input_shape:Tuple=(256,256,5),classes:int=2)->Model:
    '''
    To wrap densenet
    args:
        blocks:up-sample index
        input_shape:model trains shape
        classes:number of label
    returns:
        model:unet with backbone densenet
    '''
    assert isinstance(input_shape,Tuple) and isinstance(classes,int)
    assert isinstance(blocks,list)
    img_input = Input(shape=input_shape)
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    if blocks == [6, 12, 24, 16]:
        model = Model(img_input, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(img_input, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(img_input, x, name='densenet201')
    else:
        model = Model(img_input, x, name='densenet')
    return model


def DenseNet121(input_shape:Tuple=(256,256,5),classes:int=2)->Model:
    '''
    To implemment unet with backbone densenet121
    args:
        input_shape:model trains shape
        classes:number of label
    returns:
        model:unet with backbone densenet121
    '''
    assert isinstance(input_shape,Tuple) and isinstance(classes,int)
    return DenseNet([6, 12, 24, 16],input_shape,classes)


def DenseNet169(input_shape:Tuple=(256,256,5),classes:int=2)->Model:
    '''
    To implemment unet with backbone densenet169
    args:
        input_shape:model trains shape
        classes:number of label
    returns:
        model:unet with backbone densenet169
    '''
    assert isinstance(input_shape,Tuple) and isinstance(classes,int)
    return DenseNet([6, 12, 32, 32],input_shape,classes)

def DenseNet201(input_shape:Tuple=(256,256,5),classes:int=2)->Model:
    '''
    To implemment unet with backbone densenet201
    args:
        input_shape:model trains shape
        classes:number of label
    returns:
        model:unet with backbone densenet201
    '''
    assert isinstance(input_shape,Tuple) and isinstance(classes,int)
    return DenseNet([6, 12, 48, 32],input_shape,classes)

