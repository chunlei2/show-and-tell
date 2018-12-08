# Import modules
import numpy as np
import pandas as pd
import datetime
import time
import json
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

try:
    from nltk.translate.bleu_score import corpus_bleu
except ImportError:
    print("Trying to Install required module: nltk\n")
    os.system("python -m pip install --user nltk")
    from nltk.translate.bleu_score import corpus_bleu

from resnet101_utils import clip_gradient, save_accuracy_checkpoint, save_checkpoint, adjust_learning_rate, compute_accuracy, AverageMeter
from resnet101_models import Encoder, Decoder
from resnet101_final_dataset import CaptionDataset

print('This is resnet101 with fine tuned last two layer')
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data parameter
image_folder = "/projects/training/bauh/COCO" # Save images
data_folder = "./data/" # Save preprocessed captions and image paths
dataset_name = 'coco_5_cap_per_img_5_min_word_freq' # Should be saved in data_folder

# Model hyperparameters
embed_dim = 2048
decoder_dim = 512 # Can be adjusted
dropout_rate = 0.5
# Training hyperparameters
num_epochs = 20 # Not important, because it will usually strike the walltime before finishing this
# workers = 1
encoder_lr = 1e-4
decoder_lr = 1e-4
grad_clip = 5 # Can try to set as 2
print_freq = 100

# Some hyperparameters that may need to be tuned
batch_size = 32 # 32 seems to be the maximum safe batch size to avoid CUDA error: out of memory
#batch_size = 48 # May need to shrink to 32, in case of out-of-memory error
#fine_tune_encoder = False # Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder?
fine_tune_encoder = True # Try to improve the model
checkpoint = 'resnet101_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# checkpoint = "checkpoint_"+dataset_name+".pth.tar" # Path to checkpoint
start_epoch = 0
epochs_since_improvement = 0
best_bleu4 = 0
ave_epoch_train_loss = []
ave_epoch_val_loss = []
ave_freq_train_loss = []
ave_freq_val_loss = []
bleu4s = []

# Define main process
def main():
    """
    Describe main process including train and validation.
    """

    global start_epoch, checkpoint, fine_tune_encoder, best_bleu4, epochs_since_improvement, word_map, ave_epoch_train_loss, ave_epoch_val_loss, ave_freq_train_loss, ave_freq_val_loss, bleu4s

    # Read word map
    word_map_path = os.path.join(data_folder, 'WORDMAP_'+dataset_name+".json")
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    
    # Set checkpoint or read from checkpoint
    if checkpoint is None: # No pretrained model, set model from beginning
        decoder = Decoder(embed_dim = embed_dim, decoder_dim = decoder_dim, vocab_size = len(word_map),
                         dropout = dropout_rate)
        decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()), lr = decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()), 
                                       lr = encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()), lr = encoder_lr)
    
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Data loader
    data_start_time = datetime.datetime.now()
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    train_set = CaptionDataset(image_folder = image_folder+'/train2014/', caption_path = image_folder+"/annotations/captions_train2014.json",
                               word_map = word_map, split = 'train',  transform = transforms.Compose([normalize]))
    val_set = CaptionDataset(image_folder = image_folder+'/val2014/', caption_path = image_folder+"/annotations/captions_val2014.json",
                               word_map = word_map, split = 'val', transform = transforms.Compose([normalize]))
    # train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True,
    #                           num_workers = workers, pin_memory = True)
    # val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True,
    #                         num_workers = workers, pin_memory = True)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)
    data_now_time = datetime.datetime.now()
    print("Data Reading costs: ", data_now_time - data_start_time)

    total_start_time = datetime.datetime.now()
    print("Start the 1st epoch at: ", total_start_time)
    
    # Epoch
    for epoch in range(start_epoch, num_epochs):
        # Pre-check by epochs_since_improvement
        if epochs_since_improvement == 20: # If there are 20 epochs that no improvements are achieved
            break
        if epochs_since_improvement % 8 == 0 and epochs_since_improvement > 0: 
            adjust_learning_rate(decoder_optimizer)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer)
                
        # For every batch
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy
        decoder.train()
        encoder.train()
        
        start = time.time()
        start_time = datetime.datetime.now() # Initialize start time for this epoch

        # TRAIN
        for j, (images, captions, caplens) in enumerate(train_loader):
            if fine_tune_encoder and (epoch-start_epoch > 0 or j > 10):
                for group in encoder_optimizer.param_groups:
                    for p in group['params']:
                        state = encoder_optimizer.state[p]
                        if (state['step'] >= 1024):
                            state['step'] = 1000

            if (epoch-start_epoch > 0 or j > 10):
                for group in decoder_optimizer.param_groups:
                    for p in group['params']:
                        state = decoder_optimizer.state[p]
                        if (state['step'] >= 1024):
                            state['step'] = 1000

            data_time.update(time.time() - start)
            
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            # Forward
            enc_images = encoder(images)
            predictions, enc_captions, dec_lengths, sort_ind = decoder(enc_images, captions, caplens)
            
            # Define target as original captions excluding <start>
            target = enc_captions[:, 1:] # (batch_size, max_caption_length-1)
            target, _ = pack_padded_sequence(target, dec_lengths, batch_first=True) # Delete all paddings and concat all other parts
            predictions, _ = pack_padded_sequence(predictions, dec_lengths, batch_first=True) # (batch_size, sum(dec_lengths))
            
            loss = criterion(predictions, target)
            
            # Backward
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()
            ## Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)
            ## Update
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()
            
            # Update metrics (AverageMeter)
            acc_top5 = compute_accuracy(predictions, target, k=5)
            top5accs.update(acc_top5, sum(dec_lengths))
            losses.update(loss.item(), sum(dec_lengths))
            batch_time.update(time.time() - start)
            
            # Print current status
            if (j+1) % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Current Batch Time: {batch_time.val:.3f} (Average: {batch_time.avg:.3f})\t'
                    'Current Data Load Time: {data_time.val:.3f} (Average: {data_time.avg:.3f})\t'
                    'Current Loss: {loss.val:.4f} (Average: {loss.avg:.4f})\t'
                    'Current Top-5 Accuracy: {top5.val:.3f} (Average: {top5.avg:.3f})'.format(epoch+1, j+1, len(train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time, loss=losses,
                                                                            top5=top5accs))
                now_time = datetime.datetime.now()
                print("Epoch Training Time: ", now_time - start_time)
                print("Total Time: ", now_time - total_start_time)
                ave_freq_train_loss.append(losses.avg)
            
            start = time.time()
        
        ave_epoch_train_loss.append(losses.avg)

        # VALIDATION
        decoder.eval()
        encoder.eval()
        
        batch_time = AverageMeter()  # forward prop. + back prop. time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        start_time = datetime.datetime.now()
        
        for j, (images, captions, caplens, all_caps) in enumerate(val_loader):
            start = time.time()
            
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            
            # Forward
            enc_images = encoder(images)
            predictions, enc_captions, dec_lengths, sort_ind = decoder(enc_images, captions, caplens)
            
            # Define target as original captions excluding <start>
            predictions_copy = predictions.clone()
            target = enc_captions[:, 1:] # (batch_size, max_caption_length-1)
            target, _ = pack_padded_sequence(target, dec_lengths, batch_first=True) # Delete all paddings and concat all other parts
            predictions, _ = pack_padded_sequence(predictions, dec_lengths, batch_first=True) # (batch_size, sum(dec_lengths))
            
            loss = criterion(predictions, target)
            
            # Update metrics (AverageMeter)
            acc_top5 = compute_accuracy(predictions, target, k=5)
            top5accs.update(acc_top5, sum(dec_lengths))
            losses.update(loss.item(), sum(dec_lengths))
            batch_time.update(time.time() - start)
            
            # Print current status
            if (j+1) % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch+1, j+1, len(val_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time, loss=losses,
                                                                            top5=top5accs))
                now_time = datetime.datetime.now()
                print("Epoch Validation Time: ", now_time - start_time)
                print("Total Time: ", now_time - total_start_time)
                ave_freq_val_loss.append(losses.avg)

            ave_epoch_val_loss.append(losses.avg)
            
            ## Store references (true captions), and hypothesis (prediction) for each image
            ## If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            ## references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
            
            # references
            all_caps = all_caps[sort_ind]
            for k in range(all_caps.shape[0]):
                img_caps = all_caps[k].tolist()
                img_captions = list(map(lambda c: [w for w in c if w not in {word_map["<start>"],word_map["<pad>"]}], img_caps))
                references.append(img_captions)
                
            # hypotheses
            _, preds = torch.max(predictions_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for i, p in enumerate(preds):
                temp_preds.append(preds[i][:dec_lengths[i]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
        
        ## Compute BLEU-4 Scores
        #recent_bleu4 = corpus_bleu(references, hypotheses, emulate_multibleu=True)
        recent_bleu4 = corpus_bleu(references, hypotheses)
        bleu4s.append(recent_bleu4)
        
        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss = losses, top5 = top5accs, bleu = recent_bleu4))
        
        # CHECK IMPROVEMENT
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        # SAVE CHECKPOINT
        save_accuracy_checkpoint(dataset_name, epoch, epochs_since_improvement, ave_freq_train_loss,
                                 ave_epoch_train_loss, ave_freq_val_loss, ave_epoch_val_loss, bleu4s, is_best)
        save_checkpoint(dataset_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        print("Epoch {}, cost time: {}\n".format(epoch+1, now_time - total_start_time))
        
if __name__ == "__main__":
    main()
