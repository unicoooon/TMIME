from Big_model import AddModel
from unetLstm import unet,add_model
from sklearn.externals import joblib
from keras.models import Model
import numpy as np
from numpy import ndarray
import math
import cv2
import tifffile import tiff
from numpy import ndarray
from typing import Tuple,List
def norm(tif_data:ndarray,pred:ndarray)->ndarray:
    '''
    To delete block bandory and opsite all pixels
    args:
        tif_data:tiff data with block pixels
        pred:predicted data
    returns:
        :return data deleting block bandory
    '''
    assert isinstance(tif_data,ndarray) and isinstance(pred,ndarray)
    tmpImg = np.zeros((pred.shape[0],pred.shape[1]))
    tmpImg[pred==0]=0
    tmpImg[pred==1]=1
    npm = np.zeros((pred.shape[0],pred.shape[1],7))
    for i in range(7):
        npm[:,:,i]=tmpImg
    npm[tif_data==0]=0
    return npm[:,:,0]

def option14(bigimg:ndarray,img:ndarray)->ndarray:
    '''
    To delete block bandory
    args:
        bigimg:tiff data with block pixels
        img:predicted data
    returns:
        :return data deleting block bandory
    '''
    tmpImg = np.zeros((img.shape[0],img.shape[1]))
    tmpImg[img==0]=1
    tmpImg[img==1]=0
    npm = np.zeros((img.shape[0],img.shape[1],7))
    for i in range(7):
        npm[:,:,i]=tmpImg
    npm[bigimg==0]=0
    return npm[:,:,0]

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

def predictCloud(data_pathh:str):
    '''
    To predict cloud data,model has three inputs,because model is ensambled by three deep model
    args:
        data_pathh:data we should predict
    returns:
        :returns predicted d data
    '''
    assert isinstance(data_pathh,str)
    tif_datat=tifffile.imread(data_pathh)
    ocrr = np.zeros((math.ceil(tif_datat.shape[0]/256)*256,math.ceil(tif_datat.shape[1]/256)*256,7),'float16')
    ocrr[0:tif_datat.shape[0],0:tif_datat.shape[1],:]=tif_datat
    ttg = np.zeros((math.ceil(tif_datat.shape[0]/256)*256,math.ceil(tif_datat.shape[1]/256)*256))
    for i in range(int(ocrr.shape[0]/128)-1):
        for j in range(int(ocrr.shape[1]/128)-1):
            pred = modelCloud.predict([np.expand_dims(ocrr[128*i:128*(i+1)+128,128*j:128*(j+1)+128,:],0),
                                       np.expand_dims(ocrr[128*i:128*(i+1)+128,128*j:128*(j+1)+128,:],0),
                                       np.expand_dims(ocrr[128*i:128*(i+1)+128,128*j:128*(j+1)+128,:],0)])
            pred = np.squeeze(pred)
            ttg[128*i:128*(i+1)+128,128*j:128*(j+1)+128] = pred.argmax(axis=2)
    rg =np.zeros((tif_datat.shape[0],tif_datat.shape[1]))
    rg = ttg[0:tif_datat.shape[0],0:tif_datat.shape[1]]
    return norm(tif_datat,rg)

def predictdl(imgList:List,unetModel:Model,yunModel:Model)->ndarray:
    '''
    To predict liaohuandi
    args:
        imgList:four years data
        unetModel:pre-trained unet
        yunModel:pre-trained cloud model
    returns:
        :returns liaohuandi result
    '''
    tifSize = tifffile.imread(imgList[0])
    add_img = np.zeros((tifSize.shape[0],tifSize.shape[1]))
    for k in range(len(imgList)):
        tddata = tifffile.imread(imgList[k])
        if k == 0:
            dlres = option14(tddata,predictData(unetModel,imgList[k]))
        else:
            dlres = norm(tddata,predictData(unetModel,imgList[k]))
        add_img+=dlres
    res = np.zeros((tifSize.shape[0],tifSize.shape[1]))
    res[add_img==4]=1
    sumC = np.zeros((tifSize.shape[0],tifSize.shape[1]))
    for jj,tif_pathy in enumerate(imgList):
        if jj==0:
            continue
        else:
            tif_datat = tifffile.imread(tif_pathy)
            predy = yunModel(tif_pathy)
            pred_outt = norm(tif_datat,predy)
        sumC+=pred_outt
    rest = np.zeros((tifSize.shape[0],tifSize.shape[1]))
    rest[sumC==1]=1
    rest[sumC==2]=1
    rest[sumC==3]=1
    rt = np.zeros((tifSize.shape[0],tifSize.shape[1]))
    rg = res - rest
    rt[rg==1]=1
    return rt  
def batch_run(modelunet:Model,modelcloud:Model,nlist:List)->None:
    '''
    To get batch data
    args:
        nlist:path/row list
        Nlist:name list
        fList:forest list
    return:
        :get_
    '''
    for i in range(len(nlist)):
        cv2.imwrite(str(i)+".png",predictdl(nlist[i],unetModel,modelCloud))
    
    
if __name__ == "__main__":
    DeepModel = unet(input_shape=(256,256,7), num_classes=2, lr_init=1e-3, lr_decay=5e-4)
    modelz = Model(DeepModel.input,add_model(DeepModel))
    modelz.load_weights("/home/langyan/deeplstm.h5")
    modelCloud = AddModel(input_shape=(256,256,7),num_class=2)
    modelCloud.load_weights("/data/test_h/5jin/doubleModelUandSSKYUN.h5")
    batch_run(modelz,modelCloud,nlist)

