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
from typing import List,Tuple
from keras.applications import ResNet50 as res
from tensorflow.python.framwork import ops
def identity_block(input_tensor:ops.Tensor, kernel_size:int, filters:list, stage:int, block:str)->ops.Tensor:
    '''
    To implemment resnet block
    args:
        input_tensor:operate tensor
        kernel_size:convolution slide widow size
        filters:output channels
        stage:block stage
        block:convolution block
    returns:
        x:operated tensor
    '''
    assert isinstance(input_tensor,ops.Tensor) and isinstance(kernel_size,int)
    assert isinstance(filters,list) and isinstance(stace,int) and isinstance(block,str)
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_blockt(input_tensor:ops.Tensor, kernel_size:int, filters:tuple, stage:int, block:str, strides:Tuple=(2, 2))->ops.Tensor:
    '''
    To implemment resnet block
    args:
        input_tensor:operate tensor
        kernel_size:convolution slide widow size
        filters:output channels
        stage:block stage
        block:convolution block
        strides:convolution stride size
    returns:
        x:operated tensor
    '''
    assert isinstance(input_tensor,ops.Tensor) and isinstance(kernel_size,int) and isinstance(strides,Tuple)
    assert isinstance(filters,list) and isinstance(stace,int) and isinstance(block,str)
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape:Tuple=(256,256,5),classes:int=2)->Model:
    '''
    To implemment unet with backbone resnet
    args:
        input_shape:model trains shape
        classes:number of labels
    returns:
        model:unet with backbone resnet
    '''
    assert isinstance(input_shape,Tuple) and isinstance(classes,int)
    img_input = Input(shape=input_shape)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2), strides=(2, 2))(x)
    
    x = conv_blockt(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    
    x = conv_blockt(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_blockt(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_blockt(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x, name='resnet50')
    return model
