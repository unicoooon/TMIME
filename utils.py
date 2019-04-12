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
from typing import Tuple,List
from types import FunctionType
from keras.applications import ResNet50 as res
from tensorflow.python.framwork import ops
def get_layer_number(model:Model, layer_name:str)->int:
    '''
    To get index of layer name
    args:
        model:which model you want to up-sample
        layer_name:which layers you want to up-sample
    returns:
        i:from which numbers you up-sample
    '''
    assert isinstance(model,Model) and isinstance(layer_name,str)
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))

def handle_block_names(stage:int)->Tuple[str,str,str,str]:
    '''
    To get name of block operation
    args:
        stage:number of model block stage
    returns:
        conv_name:convolution name
        bn_name:batchnormalization name
        relu_name:activation name
        up_name:up-sample name
    '''
    assert isinstance(stage,int)
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters:int, kernel_size:Tuple,use_batchnorm:bool=False, conv_name:str='conv', bn_name:str='bn', relu_name:str='relu')->FunctionType:
    '''
    To warp conv2d and batchnormalization
    args:
        filters:output channels
        kernel_size:shape with two like [3,3] or [1,1]
        use_batchnorm:True or False,if it's False,we don't use batchnormalization ,or we use it
        conv_name:convolution name
        bn_name:batchnormalization name
        relu_name:activation name
    returns:
        :FunctionType
    '''
    assert isinstance(filters,int) and isinstance(kernel_size,Tuple)
    assert isinstance(use_batchnorm,bool) and isinstance(conv_name,str) and isinstance(bn_name,str)
    def layer(x:ops.Tensor)->ops.Tensor:
        '''
        To implemment conv2d and batchnormlization
        args:
            x:operate tensor
        returns:
            x:operated tensor
        '''
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters:int, stage:int, kernel_size:Tuple=(3,3), upsample_rate:Tuple=(2,2),
                     use_batchnorm:bool=False, skip:str=None)->FunctionType:
    '''
    To wrap convrelu and upsampling2d
    args:
        filters:output channels
        stage:index of model block
        kernel_size:shape with two like [3,3] or [1,1]
        upsample_rate:up-sample strides
        use_batchnorm:True or False,if it's False,we don't use batchnormalization ,or we use it
        skip:skip connection
    returns:
        :FunctionType
    '''
    assert isinstance(filters,int) and isinstace(stage,int)
    assert isinstance(kernel_size,Tuple) and isinstance(upsample_rate,Tuple)
    assert isinstance(use_batchnorm,bool) and isinstance(skip,str)
    def layer(input_tensor:ops.Tensor)->ops.Tensor:
        '''
        To implemment upsample and convrelu
        args:
            input_tensor:operate tensor
            x:operated tensor
        '''
        assert isinstance(input_tensor,ops.Tensor)
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        #x = input_tensor
        
        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer

def Transpose2D_block(filters:int, stage:int, kernel_size:Tuple=(3,3), upsample_rate:Tuple=(2,2),
                      transpose_kernel_size:Tuple=(4,4), use_batchnorm:bool=False, skip:str=None)->FunctionType:
    '''
    To wrap transpose convolution
    args:
        filters:output channels
        stage:index of model block
        kernel_size:shape with two like [3,3] or [1,1]
        upsample_rate:up-sample strides
        use_batchnorm:True or False,if it's False,we don't use batchnormalization ,or we use it
        skip:skip connection
        transpose_kernel_size:shape with two like [3,3] or [1,1]
    returns:
        :FunctionType
    '''
    assert isinstance(filters,int) and isinstance(stage,int)
    assert isinstance(kernel_size,Tuple) and isinstance(upsample_rate,Tuple)
    assert isinstance(use_batchnorm,bool) and isinstance(skip,str) and isinstance(transpse_kernel_size,Tuple)
    def layer(input_tensor:ops.Tensor)->ops.Tensor:
        '''
        To implemment convolution transpose and convrelu
        args:
            input_tensor:operate tensor
            x:operated tensor
        '''
        assert isinstance(input_tensor,ops.Tensor)
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        x = input_tensor
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer
def to_tuple(x)->tuple:
    '''
    To transform x to tuple
    args:
        x:operate data
    returns:
        x:tuple from data
    '''
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

def build_unet(backbone:Model, classes:int, skip_connection_layers:List,skip_con:str,index:int,
               decoder_filters:Tuple=(256,128,64,32,16),
               upsample_rates:Tuple=(2,2,2,2,2),
               n_upsample_blocks:int=5,
               block_type:str='upsampling',
               activation:str='softmax',
               use_batchnorm:bool=True)->Model:
    '''
    To implemment unet
    args:
        backbone:'vgg','resnet','densenet'
        classes:number of label
        skip_connection_layers:which layer you want to up-sample
        index:index for up-sampling
        decoder_filters:each output of each layer
        upsample_rates:upsample strides
        n_upsample_blocks:number of block
        block_type:type of operation
        activation:activation function
        use_batchnorm:True or False,if it's False,we don't use batchnormalization ,or we use it
    returns:
        model:kinds of unet models
    '''
    assert isinstance(backbone,Model) and isinstance(classes,int) and isinstance(skip_connection_layers,list)
    assert isinstance(skip_con,str) and isinstance(index,int)
    assert isinstance(decoder_filter,Tuple) and isinstance(upsample_rates,Tuple)
    assert isinstance(block_type,str) and isinstance(activation,str) and isinstance(use_batchnorm,bool)
    input = backbone.input
    x = backbone.layers[index].output
    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block
    if skip_con in ["vgg","densenet"]:
        skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])
    elif skip_con in ['inceptionresnet']:
        skip_connection_idx =(606,266,16,9)
    else:
        skip_connection_idx = (141,79,37,4) 
    # convert layer names to indices
    #for vgg
    #skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
    #                           for l in skip_connection_layers])
    #skip_connection_idx = (141,79,37,4) #it's for resnet50
    #skip_connection_idx =(606,266,16,9)#it's for resnetinceptionv2
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model

def Unet(net:Model,skip_index:list,skip_con:str,index:int,backbone_name:str='inceptionresnetv2',
         input_shape:Tuple=(256,256,5),
         input_tensor:Tuple=None,
         freeze_encoder:bool=False,
         skip_connections:str='default',
         decoder_block_type:str='upsampling',
         decoder_filters:Tuple=(256,128,64,32,16),
         decoder_use_batchnorm:bool=True,
         n_upsample_blocks:int=5,
         upsample_rates:Tuple=(2,2,2,2,2),
         classes:int=2,
         activation:str='softmax')->Model:
    '''
    To warp unet model
    args:
        net:'vgg','resnet','densenet'
        classes:number of label
        skip_connection_layers:which layer you want to up-sample
        index:index for up-sampling
        decoder_filters:each output of each layer
        upsample_rates:upsample strides
        n_upsample_blocks:number of block
        block_type:type of operation
        activation:activation function
        use_batchnorm:True or False,if it's False,we don't use batchnormalization ,or we use it
        backbone_name:name of backbone
        input_tensor:shape of input
        input_shape:input size
    returns:
        model:unet model
    '''
    assert isinstance(net,Model) and isinstance(skip_index,list)
    assert isinstance(skip_con,str) and isinstance(index,int)
    assert isinstance(decoder_filter,Tuple) and isinstance(upsample_rates,Tuple)
    assert isinstance(block_type,str) and isinstance(activation,str) and isinstance(use_batchnorm,bool)
    assert isinstance(classes,int) and isinstance(input_tensor,Tuple)
    backbone = net
    if skip_connections == 'default':
        skip_connections = skip_index[backbone_name]

    model = build_unet(backbone,
                       classes,
                       skip_connections,
                       skip_con,index,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)
    model.name = 'u-{}'.format(backbone_name)
    return model
