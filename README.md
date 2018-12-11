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

The explanation of the parameters:
image_folder: leads to a directory which saves all the images.

caption_path: leads to a json file which saves all the captions, can be train image captions or validation image folder.

word_map: a dictionary which maps from words to indexes.

split: decides we use the data set for training or testing. The only difference is testing will return all_captions which have 5 available caption for this image. All_captions will be used to calculate the BLEU-4 score.

captions_per_image: defines the number of captions for each image, default is 5.

max_len_caption: defines the max length of the caption we used, if the caption length is less than 100, we will pad the caption, if the caption length is bigger than 100, we will drop that caption and sample captions with replacement to make sure we have 5 captions for each image.

transform: defined the transformation we will apply to the image, default is none.

The architecture:
One image with one caption is one basic item in the final data set. Each image has five captions. After read in the image, there might be grey scale images. So we make sure each image has 'RGB' channels, the image size is 3,224,224, each pixel is from 0 to 1 and do the defined transformation.

Based on the image-id we chose, use COCOapi to get 5 captions. After removing too long captions and sampling with replacement, we get 5 captions for the image. Each caption along with the image is one item. We encoded the caption with the word-map we created before. Adding <start> at the beginning of the sentence which will be used as the original input of the LSTM Cell. Adding <end> at the end of the sentence which will give the guide when we should stop the sentence. Encode the rest captions, if the dictionary doesn't have the word, it will be <unk>. If the length of the caption is smaller than the max length, pad the sentence with <pad>. The caption length is the number of original tokens plus two.
  
For train data, it will return the image, the caption along with the image, and the caption length.

For test data, it will return the same as the train data except it has all captions which will be used to calculate the BLEU score.



