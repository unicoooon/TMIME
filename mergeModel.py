# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from keras.layers import Conv2D,Input
from keras.layers import ConvLSTM2D
from keras.layers import Reshape
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint
import tifffile
from keras.layers import UpSampling2D
import  matplotlib.pyplot as plt
import cv2
from keras.backend import binary_crossentropy
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
import gdal
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization,Input,DepthwiseConv2D,add,Dropout,AveragePooling2D,Concatenate
from keras.models import Model
import keras.backend as K
from keras.engine import Layer,InputSpec
from keras.optimizers import Adam,SGD
from keras.utils import conv_utils
from keras.backend import binary_crossentropy
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Reshape,ConvLSTM2D
from keras.utils import multi_gpu_model
from typing import Tuple,List
smooth = 1e-12

class unet():
    '''
    To build unet model
    attributions:
        self.num_classes:number of labels
        self.input_shape:training shape
        self.vgg_weight_path:pre-trained model
        self.img_imput:keras tensor
    '''
    def __init__(self,num_classes:int,input_shape:Tuple,vgg_weight_path:str=None)->None:
        '''
        To initialize unet class
        args:
            self.num_classes:number of labels
            self.input_shape:training shape
            self.vgg_weight_path:pre-trained model
            self.img_imput:keras tensor
        '''
        assert isinstance(num_classes,int) and isinstance(input_shape,Tuple) and isinstance(vgg_weights_path,str)
        self.num_classes=num_classes
        self.input_shape=input_shape
        self.vgg_weight_path = vgg_weight_path
        self.img_input = Input(self.input_shape)
    def modelUnet(self):->ops.Tensor:
        '''
        To build unet model
        returns:
            x:operated tensor
        '''
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(self.img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)

        block_1_out = Activation('relu')(x)

        x = MaxPooling2D()(block_1_out)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        block_2_out = Activation('relu')(x)


        x = MaxPooling2D()(block_2_out)

        # Block 3
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        block_3_out = Activation('relu')(x)

        x = MaxPooling2D()(block_3_out)

        # Block 4
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        block_4_out = Activation('relu')(x)



        x = MaxPooling2D()(block_4_out)

        # Block 5
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        for_pretrained_weight = MaxPooling2D()(x)

        # Load pretrained weights.
        if self.vgg_weight_path is not None:
            vgg16 = Model(img_input, for_pretrained_weight)
            vgg16.load_weights(vgg_weight_path, by_name=True)

        # UP 1
        x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_4_out])
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # UP 2
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_3_out])
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # UP 3
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_2_out])
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # UP 4
        x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_1_out])
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # last conv
        x = Conv2D(self.num_classes, (3,3), activation='softmax', padding='same')(x)
        return x



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

def jaccard_coef(y_true:ops.Tensor, y_pred:ops.Tensor)->ops.Tensor:
    '''
    To evacuate training model
    args:
        y_true:real mask
        y_pred:predicted mask
    returns:
        :IOU of two mask
    '''
    assert isinstance(y_true,ops.Tensor) and isinstance(y_pred,ops.Tensor)
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true:ops.Tensor, y_pred:ops.Tensor)->ops.Tensor:
    '''
    To evacuate training model
    args:
        y_true:real mask
        y_pred:predicted mask
    returns:
        :IOU of two mask
    '''
    assert isinstance(y_true,ops.Tensor) and isinstance(y_pred,ops.Tensor)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true:ops.Tensor, y_pred:ops.Tensor)->ops.Tensor:
    '''
    To train model with this loss function
    args: 
        y_true:real mask
        y_pred:predicted mask
    returns:
        :IOU loss +binary loss
    '''
    assert isinstance(y_true,ops.Tensor) and isinstance(y_pred,ops.Tensor)
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


class BilinearUpsampling(Layer):
    '''
    To build upsampling with interpretation
    attrbutions:
        self.data_format:data format
        self.upsampling:upsampling size
        self.input_spec:input spec
    '''
    def __init__(self, upsampling:Tuple=(2, 2), data_format:str=None, **kwargs)->None:
        '''
        To initialize bilinearupsampling class
        args:
            self.data_format:data format
            self.upsampling:upsampling size
            self.input_spec:input spec
        '''
        assert isinstance(upsampling,Tuple) and isinstance(data_format:str)
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape:Tuple)->Tuple:
        '''
        To compute output shape
        args:
            input_shape:training shape
        returns:
            :return output shape
        '''
        assert isinstance(input_shape,Tuple)
        height = self.upsampling[0] *             input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] *             input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs:ops.Tensor)->ops.Tensor:
        '''
        To get interpretation value
        args:
            inputs:operate tensor
        returns:
            :operated tensor
        '''
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

    def get_config(self)->dict:
        '''
        To get model config
        returns:
            :return model config
        '''
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def xception_downsample_block(x:ops.Tensor,channels:int,top_relu:bool=False)->ops.Tensor:
    '''
    To build xception downsample block
    args:
        x:operate tensor
        channels:output shape
        top_relu:we use relu at the end of model
    returns:
        x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(channels,int) and isinstance(top_relu,bool)
	if top_relu:
		x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	##separable conv2
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	##separable conv3
	x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	return x
def res_xception_downsample_block(x:ops.Tensor,channels:int)->ops.Tensor:
    '''
    To bulid res xceptioin downsample block
    args:
        x:operate tensor
        channels:output shpae
    returns:
        x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(channels,int)
	res=Conv2D(channels,(1,1),strides=(2,2),padding="same",use_bias=False)(x)
	res=BatchNormalization()(res)
	x=xception_downsample_block(x,channels)
	x=add([x,res])
	return x
def xception_block(x:ops.Tensor,channels:int)->ops.Tensor:
    '''
    To bulid res xceptioin downsample block
    args:
        x:operate tensor
        channels:output shpae
    returns:
        x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(channels,int)
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	
	##separable conv2
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	
	##separable conv3
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
        #tmp_x = np.zeros((X_train.shape[0],X_train.shape[1],7))
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	return x	
def res_xception_block(x:ops.Tensor,channels:int)->ops.Tensor:
    '''
    To bulid res xceptioin block
    args:
        x:operate tensor
        channels:output shpae
    returns:
        x:operated tensor
    '''
    assert isinstance(x,ops.Tensor) and isinstance(channels,int)
	res=x
	x=xception_block(x,channels)
	x=add([x,res])
	return x
def aspp(x:ops.Tensor,input_shape:Tuple,out_stride:int)->ops.Tensor:
    '''
    To build aspp layer
    args:
        x:operate tensor
        input_shape:training shape
        out_stride:dilate rate
    returns:
        x:operated tensor
    '''
	b0=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	b0=BatchNormalization()(b0)
	b0=Activation("relu")(b0)
	
	b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
	b1=BatchNormalization()(b1)
	b1=Activation("relu")(b1)
	b1=Conv2D(256,(1,1),padding="same",use_bias=False)(b1)
	b1=BatchNormalization()(b1)
	b1=Activation("relu")(b1)
	
	b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
	b2=BatchNormalization()(b2)
	b2=Activation("relu")(b2)
	b2=Conv2D(256,(1,1),padding="same",use_bias=False)(b2)
	b2=BatchNormalization()(b2)
	b2=Activation("relu")(b2)	

	b3=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
	b3=BatchNormalization()(b3)
	b3=Activation("relu")(b3)
	b3=Conv2D(256,(1,1),padding="same",use_bias=False)(b3)
	b3=BatchNormalization()(b3)
	b3=Activation("relu")(b3)
	
	out_shape=int(input_shape[0]/out_stride)
	b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
	b4=Conv2D(256,(1,1),padding="same",use_bias=False)(b4)
	b4=BatchNormalization()(b4)
	b4=Activation("relu")(b4)
	b4=BilinearUpsampling((out_shape,out_shape))(b4)
	
	x=Concatenate()([b4,b0,b1,b2,b3])
	return x
class deeplabv3_plus():
    '''
    To build deeplabv3 plus
    attributoins:
        self.input_shape:training shape
        self.out_stride:dialate rate
        self.num_classes:number of labels
        self.img_input:keras tensor
    '''
    def __init__(self,input_shape:Tuple,num_classes:int,out_stride:int=16)->None:
        '''
        To initialize deeplabv3_plus class
        args:
            self.input_shape:training shape
            self.out_stride:dialate rate
            self.num_classes:number of labels
            self.img_input:keras tensor
        '''
        assert isinstance(input_shape,Tuple) and isinstance(num_classes,int) and isinstance(out_stride,int)
        self.input_shape=input_shape
        self.out_stride=out_stride
        self.num_classes=num_classes
        self.img_input=Input(shape=self.input_shape)
    def modelDeeplabv3_plus(self)->ops.Tensor:
        '''
        To build base deeplabv3 plus
        returns:
            x:operated model for aspp layer
        '''
        x=Conv2D(32,(3,3),strides=(2,2),padding="same",use_bias=False)(self.img_input)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=Conv2D(64,(3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)

        x=res_xception_downsample_block(x,128)

        res=Conv2D(256,(1,1),strides=(2,2),padding="same",use_bias=False)(x)
        res=BatchNormalization()(res)	
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
        skip=BatchNormalization()(x)
        x=Activation("relu")(skip)
        x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)	
        x=add([x,res])

        x=xception_downsample_block(x,728,top_relu=True)

        for i in range(16):
            x=res_xception_block(x,728)

        res=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
        res=BatchNormalization()(res)	
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(728,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)	
        x=add([x,res])

        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Conv2D(2048,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)	
        x=Activation("relu")(x)

        #aspp
        x=aspp(x,self.input_shape,self.out_stride)
        x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=Dropout(0.9)(x)

        ##decoder 
        x=BilinearUpsampling((4,4))(x)
        dec_skip=Conv2D(48,(1,1),padding="same",use_bias=False)(skip)
        dec_skip=BatchNormalization()(dec_skip)
        dec_skip=Activation("relu")(dec_skip)
        x=Concatenate()([x,dec_skip])

        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)

        x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=Conv2D(self.num_classes,(1,1),padding="same")(x)
        x=BilinearUpsampling((4,4))(x)
        x=Activation("relu")(x)
        return x

class SegNet():
    '''
    To build segnet
    attributions:
        self.input_shape:training shape
        self.classes:number of labels
        self.img_input:keras tensor
    '''
    def __init__(self,input_shape:Tuple,classes:int)->None:
        '''
        To initialize segnet class
        args:
            self.input_shape:training shape
            self.classes:number of labels
            self.img_input:keras tensor
        '''
        assert isinstance(input_shape,Tuple) and isinstance(classes,int)
        self.input_shape=input_shape
        self.classes=classes
        self.img_input = Input(shape=self.input_shape)
    def modelseg(self)->ops.Tensor:
        '''
        To build unet model
        returns:
            x:operated tensor
        '''
        x = self.img_input
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Decoder
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(self.classes, (1, 1), padding="valid")(x)
        #x = Reshape((input_shape[0] * input_shape[1], classes))(x)
        x = Activation("relu")(x)
        return x

def AddModel(input_shape:Tuple,num_class:int)->Model:
    '''
    To build net with unet and lstm
    args:
        input_shape:training shape
        num_classes:number of labels
    returns:
        rg:model with unet and lstm
    '''
    modelU = unet(input_shape=input_shape,num_classes=num_class)
    modelut = modelU.modelUnet()
    modelD = deeplabv3_plus(input_shape=input_shape,num_classes=num_class)
    modeldeep = modelD.modelDeeplabv3_plus()
    modelS = SegNet(input_shape=input_shape,classes=num_class)
    modelsg = modelS.modelseg()
    modelAdd = Concatenate(axis=3)([modelut,modeldeep,modelsg])
    x = Conv2D(64,kernel_size=3,padding='same',activation='relu')(modelAdd)
    #x = Conv2D(32,kernel_size=1,padding='same',activation="relu")(x)
    #x = Reshape((1,int(x.shape[1]),int(x.shape[2]),int(x.shape[3])))(x)
    #x = ConvLSTM2D(32,kernel_size=3,padding='same',return_sequences=False,activation='relu')(x)
    #x = Conv2D(64,kernel_size=3,padding='same',activation='relu')(x)
    x = Conv2D(num_class,kernel_size=1,padding='same',activation='softmax')(x)
    rg = Model(inputs=[modelU.img_input,modelD.img_input,modelS.img_input],outputs=x)
    #modelp = multi_gpu_model(rg,gpus=3)
    rg.compile(loss="categorical_crossentropy",optimizer=Adam(),metrics=[dice_coef])
    return rg




