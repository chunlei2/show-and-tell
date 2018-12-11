import os
import numpy as np
import h5py
import json
import torch
import datetime
from scipy.misc import imread, imresize
from collections import Counter
from random import seed, choice, sample

def create_word_map(train_annot_path, val_annot_path, output_folder="final/data",
                    min_word_freq=5, captions_per_image=5, dataset_name='coco'):
    # Read the json file that stores captions
    with open(train_annot_path, 'r') as j:
        train_data = json.load(j)
    with open(val_annot_path, 'r') as j:
        val_data = json.load(j)
    
    # Count word frequencies
    word_freq = Counter()
    start_time = datetime.datetime.now()
    for i, annot in enumerate(train_data["annotations"]):
        tmp_caption = annot["caption"].replace('.','')
        tmp_tokens = tmp_caption.split()
        for c in tmp_tokens: # Update word frequency
            word_freq.update(c)
        if (i+1)%1000 == 0:
            now_time = datetime.datetime.now()
            print("{} annotations finished, cost Time: {}".format(i+1, now_time-start_time))
    for i, annot in enumerate(val_data["annotations"]):
        tmp_caption = annot["caption"].replace('.','')
        tmp_tokens = tmp_caption.split()
        for c in tmp_tokens: # Update word frequency
            word_freq.update(c)
        if (i+1)%1000 == 0:
            now_time = datetime.datetime.now()
            print("{} annotations finished, cost Time: {}".format(i+1, now_time-start_time))
            
    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0  # word_map: {'<pad>':0, 'a':1, 'b':2, 'c':3, '<unk>':4, '<start>':5, '<end>':6}
    
    # Create a base/root name for all output files
    base_filename = dataset_name + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w+') as j:
        json.dump(word_map, j)

if __name__ == "__main__":
    data_folder = "/projects/training/bauh/COCO/"
    train_annot_path = data_folder + "annotations/captions_train2014.json"
    val_annot_path = data_folder + "annotations/captions_val2014.json"
    output_folder = "~/"
    # output_folder = "final/data"
    create_word_map(train_annot_path, val_annot_path, output_folder=output_folder)
