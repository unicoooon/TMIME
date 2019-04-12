from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.backend import binary_crossentropy
smooth =1e-12
from tensorflow.python.framework import ops
from typing import Tuple,List

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


def jaccard_coef_int(y_true:ops.Tensor, y_pred:Tensor)->ops.Tensor:
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

def unet(num_classes:int, input_shape:Tuple, lr_init:float, lr_decay:float, vgg_weight_path:str=None)->Model:
    '''
    To build unet model
    args:
        num_classes:number of labels
        input_shape:training shape
        lr_init:lr rate
        lr_decay:weights decay
        vgg_weight_path:pre-trained model
    return 
        model:unet model
    '''
    assert isinstance(num_classes,int) and isinstance(input_shape,Tuple) and isinstance(lr_init,float)
    assert isinstance(lr_decay,float) and isinstance(vgg_weight_path,str)
    img_input = Input(input_shape)
    
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
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
    if vgg_weight_path is not None:
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
    x = Conv2D(num_classes, (3,3), activation='softmax', padding='same')(x)
    
    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model

def add_model(unet_model:Model)->ops.Tensor:
    '''
    To implemment unet with lstm
    args:
        unet_model:pre-tarined unet model
    returns:
        x:tensor with lstm and unet
    '''
    assert(unet_model,Model)
    unet_model.load_weights("/home/langyan/langyan/keras-image-segmentation/h5File/unet_model_weight.h5")
    xt = unet_model.output
    x = Reshape((1,int(xt.shape[1]),int(xt.shape[2]),int(xt.shape[3])))(xt)
    x = ConvLSTM2D(64,kernel_size=3,padding='same',return_sequences=False,activation='relu')(x)
    x = MaxPooling2D(pool_size=2,padding='same')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(16,kernel_size=3,padding='same',activation='relu')(x)
    x = Reshape((1,int(x.shape[1]),int(x.shape[2]),int(x.shape[3])))(x)
    x = ConvLSTM2D(64,kernel_size=3,padding='same',return_sequences=False,activation='relu')(x)
    x = Conv2D(2,kernel_size=1,padding='same',activation='softmax')(x)
    return x