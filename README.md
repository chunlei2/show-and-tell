# show-and-tell
Given a picture, automatically output a sentence.

## Introduction
This repository is an implementation of the paper: Show and Tell: A Neural Image Caption Generator.https://arxiv.org/pdf/1411.4555.pdf

The dataset is MSCOCO 2014 train images http://images.cocodataset.org/zips/train2014.zip, MSCOCO 2014 validation images:http://images.cocodataset.org/zips/val2014.zip, MSCOCO 2014 train/val annotations:http://images.cocodataset.org/annotations/annotations_trainval2014.zip.

There are three pretrained models we have tried. Resnet50, Resnet101, Resnet152.

There are three three dropout rates we have tried. 0.3, 0.5 and 0.8. 0.3 behaves best.

For resnet 50, tune all the parameters in the decoder, only tune the last two layers in the encoder to save memory. One epoch training time is about 4.5 hours. One epoch testing time is about 1.5 hours. After one epoch, the average training loss is 3.2695, the average top5 accuracy is 63.535%. The average testing loss is 2.751, the average top5 accuracy is 70.107%, the BLEU-4 score is 0.1868. 

For resnet 101, tune all the parameters in the decoder, only tune the last layer in the encoder to avoid the out of memory error. One epoch training time is about 5 hours. One epoch testing time is about 4.5 hours. One epoch testing time is about 2 hours. After one epoch, the average training loss is 3.2577, the average top5 accuracy is 63.700%. The average testing loss is 2.745, the average top5 accuracy is 70.245%, the BLEU-4 score is 0.1869. 

For resnet 152, tune all the parameters in the decoder, only tune the last layer in the encoder to avoid the out of memoru error. One epoch training time is about 5 hours. One epoch testing time is about 2 hours. After one epoch, the average training loss is 3.2526, the average top5 accuracy is 63.830%. The average testing loss is 2.735, the average top5 accuracy is 70.361%, the BLEU-4 score is 0.1889.

The general overview of the whole project is https://github.com/chunlei2/show-and-tell/blob/master/IE534_final_report.pdf.

## Example:
![Image of Web image](https://github.com/chunlei2/show-and-tell/blob/master/result.png)
