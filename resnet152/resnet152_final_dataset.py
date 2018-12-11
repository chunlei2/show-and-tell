import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
import skimage.io as io
from random import seed, choice, sample
from scipy.misc import imread, imresize
from pycocotools.coco import COCO
from PIL import Image

class CaptionDataset(Dataset):
    def __init__(self, image_folder, caption_path, word_map, split, max_len_caption=100, captions_per_image=5, transform=None):   
        self.split = split    
        self.capFile = caption_path
        self.coco_caps=COCO(self.capFile)
        self.imgIds = self.coco_caps.getImgIds()
        self.cpi = captions_per_image
        self.image_folder = image_folder
        self.word_map = word_map
        self.max_len_caption = max_len_caption
        self.transform = transform
    def __getitem__(self, i):
        img = self.coco_caps.loadImgs(self.imgIds[i // self.cpi])[0]
        filename = self.image_folder + img['file_name']
        I = imread(filename)
        if len(I.shape) == 2: # deal with images with only grey scale
            I = I[:, :, np.newaxis]
            I = np.concatenate([I, I, I], axis=2)
        I = imresize(I, (224, 224))
        I = I.transpose(2, 0, 1)
        assert I.shape == (3, 224, 224)
        assert np.max(I) <= 255
        I = torch.FloatTensor(I / 255.)
        if self.transform is not None:
            I = self.transform(I)
        annIds = self.coco_caps.getAnnIds(imgIds=img['id']) 
        anns = self.coco_caps.loadAnns(annIds) #A list with 5 dictionaries 
        tokens = [caption['caption'].replace('.', '').split() for caption in anns]
        tokens = [list(map(lambda x:x.lower(), token)) for token in tokens]
        for token in tokens:
            if len(token) > self.max_len_caption:
                    tokens.remove(token)
        if len(tokens) < self.cpi:
            tokens = tokens + [choice(tokens) for _ in range(self.cpi - len(tokens))]
        else:
            tokens = sample(tokens, k=self.cpi) 
        if self.split is 'train':
            all_captions = list()
            caplens = list()
            for j, c in enumerate(tokens):
                # Encode captions
                enc_c = [self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in c] + [
                        self.word_map['<end>']] + [self.word_map['<pad>']] * (self.max_len_caption - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                all_captions.append(enc_c)
                caplens.append(c_len)
            caplen = torch.LongTensor([caplens[i%self.cpi]])
            caption = torch.LongTensor(all_captions[i%self.cpi])
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return I, caption, caplen         
            # caption = tokens[i%self.cpi]
            # enc_c = [self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in caption] + [
            #             self.word_map['<end>']] + [self.word_map['<pad>']] * (self.max_len_caption - len(caption))
            # caplen = len(caption) + 2
            # caption = torch.LongTensor(enc_c)
            # return I, caption, caplen
        else:
            all_captions = list()
            caplens = list()
            for j, c in enumerate(tokens):
                # Encode captions
                enc_c = [self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in c] + [
                        self.word_map['<end>']] + [self.word_map['<pad>']] * (self.max_len_caption - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                all_captions.append(enc_c)
                caplens.append(c_len)
            caplen = torch.LongTensor([caplens[i%self.cpi]])
            all_captions = torch.LongTensor(all_captions)
            caption = torch.LongTensor(all_captions[i%self.cpi])
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return I, caption, caplen, all_captions



    
    def __len__(self):
        return len(self.imgIds)*self.cpi