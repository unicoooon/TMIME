# coding: utf-8
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Model
import cv2
from numpy import ndarray
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
import cv2
import tifffile
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
from resnet50 import ResNet50
from resnetinception import InceptionResNetV2
from densenet import DenseNet121
from densenet import DenseNet169
from densenet import DenseNet201
from utils import Unet
from typing import Tuple
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
}#18,21,173,779,425,593,705
vgg16net,vgg19net,resnet50,inceptionresnet,densenet121,densenet169,densenet201 = RetModel()
netList=[vgg16net,vgg19net,resnet50,inceptionresnet,densenet121,densenet169,densenet201]
skip_con_List=["vgg","vgg","resnet","inceptionresnet","densenet","densenet","densenet"]
indexList=[18,21,173,779,425,593,705]
backbone_name_List=["vgg16","vgg19","resnet50","inceptionresnetv2","densenet121","densenet169","densenet201"]
def collect_model(netList:list,skip_con_List:list,indexList:list,backbone_name_List:list)->list:
    '''
    To load all models we trained
    args:
        netList:all ensambling models
        skip_con_List:which backbone we use
        indexList:which layers we up-sampling
        backbone_name_List:list all backbones
    return:
        load_weight_model:return model list
    '''
    assert isinstance(netList,list) and isinstance(skip_con_List,list)
    assert isinstance(indexList,list) and isinstance(backbone_name_List,list)
    load_weight_model=[]
    for i in range(len(backbone_name_List)):
        model_name = backbone_name_List[i]+'_model_weight.h5'
        net = Unet(netList[i],skip_index=DEFAULT_SKIP_CONNECTIONS,
                   skip_con=skip_con_List[i],index=indexList[i],backbone_name=backbone_name_List[i])
        net.load_weights(model_name)
        load_weight_model.append(net)
    return load_weight_model
def predictData(modelt:Model,data_path:str)->ndarray:
    '''
    To predict data
    args:
        modelt:model we finished training
        data_path:tif path we should predict
    returns:
        :predicted ndarray
    '''
    assert isinstance(modelt,Model) and isinstance(data_path,str)
    x_img = tifffile.imread(data_path)/255
    ocr = np.zeros((math.ceil(x_img.shape[0]/256)*256,math.ceil(x_img.shape[1]/256)*256,7),'float16')
    ocr[0:x_img.shape[0],0:x_img.shape[1],:]=x_img
    #ocr[x_img.shape[0]:,x_img.shape[0]:,:]=0
    tmp = np.zeros((math.ceil(x_img.shape[0]/256)*256,math.ceil(x_img.shape[1]/256)*256))
    for i in range(int(ocr.shape[0]/128)-1):
        for j in range(int(ocr.shape[1]/128)-1):
            pred = modelt.predict(np.expand_dims(ocr[128*i:128*(i+1)+128,128*j:128*(j+1)+128,:],0))
            pred = np.squeeze(pred)
            tmp[128*i:128*(i+1)+128,128*j:128*(j+1)+128] = pred.argmax(axis=2)
    rg =np.zeros((x_img.shape[0],x_img.shape[1]))
    rg = tmp[0:x_img.shape[0],0:x_img.shape[1]]
    tmpt = np.zeros((x_img.shape[0],x_img.shape[1],7))
    for t in range(7):
        tmpt[:,:,t]=rg
    tmpt[x_img==0] = 0
    return tmpt[:,:,0]
def ensambleModel(model_list:list,data_path:str)->ndarray:
    '''
    To ensamble all trained models
    args:
        model_list:all ensambling trained models
        data_path:tif file we should predict
    returns:
        vote_mask:predicted test data
    '''
    assert isinstance(model_list,list) and isinstance(data_path,str)
    result_list=[]
    for j in range(7):
        result_list.append(predictData(model_list[j],data_path))
    height,width = result_list[0].shape
    vote_mask = np.zeros((height,width))
    for h in range(height):
        for w in range(width):
            record = np.zeros((1,2))
            for n in range(len(result_list)):
                mask = result_list[n]
                pixel = mask[h,w]
                record[0,int(pixel)]+=1
            label = record.argmax()
            vote_mask[h,w] = label
    return vote_mask
load_model = collect_model(netList,skip_con_List,indexList,backbone_name_List)
res = ensambleModel(load_model,"120041/LC81200412014210LGN00_merge_result.tif")
cv2.imwrite("/home/langyan/langyan/resultt/seveng.png",res)



