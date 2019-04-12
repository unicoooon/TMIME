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
def VGG19(input_shape:Tuple,classes:int=2)->Model:
    '''
    To implemment unet with backbone vgg19
    args:
        input_shape:model trains shape
        classes:number of label
    returns:
        model:unet with backbone vgg19
    '''
    assert isinstance(input_shape,Tuple) and isinstance(classes,int)
    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(img_input, x, name='vgg19')
    return model    

