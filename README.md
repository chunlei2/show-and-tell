# show-and-tell
Given a picture, automatically output a sentence.

## Introduction
This repository is an implementation of the paper: Show and Tell: A Neural Image Caption Generator.https://arxiv.org/pdf/1411.4555.pdf

The dataset is MSCOCO 2014 train images http://images.cocodataset.org/zips/train2014.zip, MSCOCO 2014 validation images:http://images.cocodataset.org/zips/val2014.zip, MSCOCO 2014 train/val annotations:http://images.cocodataset.org/annotations/annotations_trainval2014.zip.

There are three pretrained models I have tried. Resnet50, Resnet101, Resnet152. 
For resnet 50, tune all the parameters in the decoder, only tune the last two layers in the encoder to save memory. One epoch training time is about 4.5 hours. One epoch testing time is about 1.5 hours. After one epoch, my average training loss is 3.2695, my average top5 accuracy is 63.535%. My average testing loss is 2.751, my average top5 accuracy is 70.107%, my BLEU-4 score is 0.1868. 

For resnet 101, tune all the parameters in the decoder, only tune the last layer in the encoder to avoid the out of memory error. One epoch training time is about 5 hours. One epoch testing time is about 4.5 hours. One epoch testing time is about 2 hours. After one epoch, my average training loss is 3.2577, my average top5 accuracy is 63.700%. My average testing loss is 2.745, my average top5 accuracy is 70.245%, my BLEU-4 score is 0.1869. 

For resnet 152, tune all the parameters in the decoder, only tune the last layer in the encoder to avoid the out of memoru error. One epoch training time is about 5 hours. One epoch testing time is about 2 hours. After one epoch, my average training loss is 3.2526, my average top5 accuracy is 63.830%. My average testing loss is 2.735, my average top5 accuracy is 70.361%, my BLEU-4 score is 0.1889. 

## resnet101_final_dataset.py
This file create a dataset which can be loaded into the MSCOCO dataloader. We used official MSCOCO API, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb.

Here is the explanation of all the parameters:
image_folder: leads to a directory which saves all the images.
caption_path: 
split: decide we use the data set for training or testing. The only difference is testing will return all_captions which have 5 available caption for this image. All_captions will be used to calculate the BLEU-4 score.

