# This python file will read an image, beam_size, model and word_map in order to produce a caption.

# Import modules
import numpy as np
import pandas as pd
import datetime
import json
import os
import re
import os.path
import argparse

from PIL import Image
from scipy.misc import imread, imresize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets
import torch.utils.data as data

from resnet101_models import Encoder, Decoder
from resnet101_final_dataset import CaptionDataset


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def captions_beam_search(image_path, encoder, decoder, word_map, beam_size=3):
    """
    image_path: Path to an input image
    encoder: encoder model
    decoder: decoder model
    word_map: Dictionary that maps words into indices
    beam_size: Parameter to search a sentence
    """
    k = beam_size
    vocab_size = len(word_map)

    # Read image and preprocess
    img = imread(image_path)
    if len(img.shape) == 2:  # Grey Image
        img = np.tile(img, (3, 1, 1))
        img = img.transpose(1, 2, 0)

    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = img.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k`
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    # Tensor to store top k previous words at each step; now they're just <start>`
    prev_words = torch.LongTensor([[word_map["<start>"]]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences; now they're just <start>`
    seqs = prev_words  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0`
    top_k_scores = torch.zeros([k, 1]).to(device)  # (k, 1)

    # Initialize lists
    complete_seqs = []
    complete_scores = []

    # Decode
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)  # (1, decoder_dim)
    # Iterate until all k sequences are completed
    while True:
        # Compute scores of current k previous words
        embeddings = decoder.embedding(prev_words).squeeze(1)  # (k, 1, embed_dim) to (k, embed_dim)
        h, c = decoder.decode_step(
            embeddings,  # (1, embed_dim)
            (h, c))  # (1, decoder_dim)
        scores = decoder.fc(h)  # (k, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add (i.e. multiply because of 'log' above) to current scores
        scores = top_k_scores.expand_as(scores) + scores
        # Take the maximum k elements in (k * vocab_size) combinations
        if step == 1:  # Initialize
            top_k_scores, top_k_locations = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_locations = scores.view(-1).topk(k, 0, True, True)
        # Row and Column indices of k largest elements
        top_k_prev_ind = top_k_locations / vocab_size  # (k, 1)
        top_k_next_ind = top_k_locations % vocab_size  # (k, 1)

        # Update sequences
        seqs = torch.cat([seqs[top_k_prev_ind], top_k_next_ind.unsqueeze(1)], dim=1)  # (k, step+1)

        # Check whether a sequence is completed
        incomplete_inds = [j for j, next_word in enumerate(top_k_next_ind) if next_word != word_map["<end>"]]
        complete_inds = list(set(range(len(top_k_next_ind))) - set(incomplete_inds))

        # Deal with completed sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length

        # Deal with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[top_k_prev_ind[incomplete_inds]]
        c = c[top_k_prev_ind[incomplete_inds]]
        encoder_out = encoder_out[top_k_prev_ind[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        prev_words = top_k_next_ind[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    max_i = np.argmax(complete_scores)
    max_seq = complete_seqs[max_i]

    return max_seq


if __name__ == "__main__":
    # Load parser
    parser = argparse.ArgumentParser(description="Show_and_Tell_Caption_Generator")

    parser.add_argument("--img", "-i", help="Path to one image")
    parser.add_argument("--model", "-m", help="Path to model (both encoder and decoder)")
    parser.add_argument("--word_map", "-wm", help="Path to word map")
    parser.add_argument("--beam_size", "-b", default=5, type=int, help="beam_size(default as 5)")

    args = parser.parse_args()

    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    model_ckpt = torch.load(args.model, map_location='cpu')
    decoder = model_ckpt["decoder"].to(device)
    encoder = model_ckpt["encoder"].to(device)
    decoder.eval()
    encoder.eval()

    # Run through caption beam search function
    best_seq = captions_beam_search(image_path=args.img, encoder=encoder, decoder=decoder, word_map=word_map, beam_size=args.beam_size)
    words = [rev_word_map[ind] for ind in best_seq]

    print(words)


# python3.6 captions.py -i 'eg_img/dog.jpg' -m 'res101_0.3.pth.tar' -wm 'data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' -b 3
# python3.6 captions.py -i 'eg_img/kobe_2.jpg' -m 'lcl.pth.tar' -wm 'data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' -b 5
