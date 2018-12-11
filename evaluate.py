# Import modules
import numpy as np
import pandas as pd
import datetime
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
#from torch.autograd import Variable
#import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import resnet101_models

try:
    from nltk.translate.bleu_score import corpus_bleu
except ImportError:
    print("Trying to Install required module: nltk\n")
    os.system("python -m pip install --user nltk")
    from nltk.translate.bleu_score import corpus_bleu

from utils import init_embedding, load_embeddings, save_checkpoint, adjust_learning_rate, compute_accuracy, AverageMeter
from models import Encoder, Decoder
from datasets import CaptionDataset

#hyper parameters

batch_size = 1
workers = 1

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data parameter
image_folder = "/projects/training/bauh/COCO" # Save images
# data_folder = "scratch" # Save preprocessed captions and image paths
data_folder = 'scratch/final/data' # path for dataset
dataset_name = 'coco_5_cap_per_img_5_min_word_freq' # Should be saved in data_folder
word_map_path = os.path.join(data_folder, 'WORDMAP_'+dataset_name+".json")
# checkpoint_path = os.path.join(data_folder, "BEST_"+"checkpoint_" + dataset_name + ".pth.tar")
checkpoint_path = os.path.join(data_folder, "BEST_resnet101_checkpoint_" + dataset_name + ".pth.tar")

## Read model
model_ckpt = torch.load(checkpoint_path)
decoder = model_ckpt["decoder"].to(device)
encoder = model_ckpt["encoder"].to(device)
decoder.eval()
encoder.eval()
## Read word_map
with open(word_map_path, 'r') as j:
    word_map = json.load(j)
inv_word_map = {v: k for k,v in word_map.items()}
vocab_size = len(word_map)

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
val_set = CaptionDataset(data_folder = data_folder, data_name = dataset_name,
                             split = "VAL", transform = transforms.Compose([normalize]))
val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True,
                            num_workers = workers, pin_memory = True)



# Define validation process
def validation(beam_size):
    """
    Evaluation Process
    
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    
    start_time = datetime.datetime.now()
    print("Start training at: ", start_time)
        
    for j, (images, captions, caplens, all_caps) in enumerate(val_loader):
        k = beam_size
        start = datetime.datetime.now()
            
        images = images.to(device)
        
        # Forward
        encoder_out = encoder(images)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(-1)
        ## Flatten encoding``
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        ## We'll treat the problem as having a batch size of k`
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        ## Tensor to store top k previous words at each step; now they're just <start>`
        prev_words = torch.LongTensor([[word_map["<start>"]]] * k).to(device) # (k, 1)
        ## Tensor to store top k sequences; now they're just <start>`
        seqs = prev_words # (k, 1)
        ## Tensor to store top k sequences' scores; now they're just 0`
        seqs_scores = torch.zeros([k,1]).to(device) # (k, 1)
    
        # Initialize lists
        complete_seqs = []
        complete_scores = []

        print("start decode")
        # Decode
        step = 1
        h,c = decoder.init_hidden_state(encoder_out) # (k, decoder_dim)
        ## Iterate until all k sequences are completed 
        while True:
            # Compute scores of current k previous words
            embeddings = decoder.embedding(prev_words).squeeze(1) # (k, 1, embed_dim) to (k, embed_dim)
            h, c = decoder.decode_step(
                embeddings, # (1, embed_dim)
                (h, c))  # (1, decoder_dim)
            scores = decoder.fc(decoder.dropout(h))  # (k, vocab_size)
            scores = F.log_softmax(scores,dim=2)
        
            # Add (i.e. multiply because of 'log' above) to current scores
            scores = seqs_scores.expand_as(scores) + scores
            # Take the maximum k elements in (k * vocab_size) combinations
            if step == 1: ## Initialize
                top_scores, top_k_locations = scores[0].topk(k, 0, True, True)
            else:
                top_scores, top_k_locations = scores.view(-1).topk(k, 0, True, True)
            # Row and Column indices of k largest elements
            top_k_prev_ind = top_k_locations // vocab_size # (k, 1)
            top_k_next_ind = top_k_locations % vocab_size # (k, 1)
        
            # Update sequences
            seqs = torch.cat([seqs[top_k_prev_ind], top_k_next_ind.unsqueeze(1)], dim=1) # (k, step+1)
        
            # Check whether a sequence is completed
            comp_seqs_ind = [j for j, next_word in enumerate(top_k_next_ind) if next_word == word_map["<end>"]]
            incomp_seqs_ind = list(set(range(seqs.size(0))) - set(comp_seqs_ind))
        
            # Deal with completed sequences
            if len(comp_seqs_ind) > 0:
                complete_seqs.extend(seqs[comp_seqs_ind].tolist())
                complete_scores.extend(seqs_scores[comp_seqs_ind])
            k -= len(comp_seqs_ind)  # reduce beam length
        
            # Deal with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomp_seqs_ind]
            h = h[top_k_prev_ind[incomp_seqs_ind]]
            c = c[top_k_prev_ind[incomp_seqs_ind]]
            encoder_out = encoder_out[top_k_prev_ind[incomp_seqs_ind]]
            #seqs_scores = seqs_scores[incomp_seqs_ind].unsqueeze(1)
            seqs_scores = seqs_scores[incomp_seqs_ind]
            #prev_words = top_k_next_ind[incomp_seqs_ind].unsqueeze(1)
            prev_words = top_k_next_ind[incomp_seqs_ind]

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        
        max_i = np.argmax(complete_scores)
        #max_i = complete_scores.index(max(complete_scores))
        max_seq = complete_seqs[max_i]
            
        ## Store references (true captions), and hypothesis (prediction) for each image
        ## If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        ## references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
            
        # references
        # all_caps = all_caps[sort_ind]
        print("reference")
        for k in range(all_caps.shape[0]):
            img_caps = all_caps[k].tolist()
            img_captions = list(map(lambda c: [w for w in c if w not in {word_map["<start>"],word_map["<pad>"]}], img_caps))
            references.append(img_captions)

        # hypotheses
        hypotheses.append([w for w in max_seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)
        
    ## Compute BLEU-4 Scores
    bleu4 = corpus_bleu(references, hypothesese)
        
    return bleu4
        
if __name__ == "__main__":
    beam_size = 5
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, validation(beam_size)))