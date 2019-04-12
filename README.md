# TMIME

## Introduction

This repository is to make image segmentation easy and efficency, we segment image with attention method and LSTM,this two methods can easily improve the result of segmentation.espacally,attention layer get connnetion with all pixels,it can make model to attention all pixels rather than kernel-size area,furthermore,LSTM can link the relationship with each pixel.the skill can delete hole and small area,which make the result of segmentation clean and more accuracy.we mplemment unet with different backbones to prove this method,and make it’s result with different backbones’s ensambling,it shows that this skill works well.

## Requirement

* keras
* tensorflow
* opencv


## Usage


1\. Training 1:to train ensambleTrain.py to get pretrained weights, there it is (vgg_based model,resnet_based model,densenet_based model)

   ```shell
  python3 ensambleTrain.py
   ```

   The pretrained_model dir will get all pretrained model.

2 \. Train 2:Get first step’s result to make it as pretrained model,Train model based vgg,resnet,densenet(ensamble three based unet) with attention and LSTM.

   ```shell
   python3 attention_train.py
   ```

   The attention_weights will get the trained model

3\. test:

 ```shell
   python3 attention_predict.py
   ```
 
## Main ideas
* Use attention to ensamble different model,for example,we use attention to ensamble different backbone-based unet.
* Use LSTM to improve the result of segmentation,previously,we use crf to make result nice more,but it just a trick for result,it not a layer that train at model,but,LSTM can train model end-to-end.
* Ensamble result with different models,we use single model to predict image and use votes to decide final results.


## Update
* Add more image segmentation skills