# show-and-tell
Given a picture, automatically output a sentence.

This repository is an implementation of the paper: Show and Tell: A Neural Image Caption Generator.https://arxiv.org/pdf/1411.4555.pdf

There are three pretrained models I have tried. Resnet50, Resnet101, Resnet152. For resnet 50, tune all the parameters in the decoder, only tune the last two layers in the encoder to save memory. 
