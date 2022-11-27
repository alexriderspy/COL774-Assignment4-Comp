import torch
import torchvision
import torchtext
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import tqdm
from torchtext.data import get_tokenizer
from torchtext import data
import math
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertModel, AdamW, BertConfig
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

directory = '/kaggle/input/col774-2022/'
dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv'))
dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv'))
dataframe_val_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv'))
dataframe_val_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv'))

batch_size = 50

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

max_len = 0
input_ids = []
attention_masks = []

sentences = np.hstack((dataframe_x['Title'].values,  dataframe_val_x['Title'].values))
labels = np.hstack((dataframe_y['Genre'].values, dataframe_val_y['Genre'].values))

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = 64,  pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

learning_rate = 5e-5

def ret_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased", 
        num_labels = 30, 
        output_attentions = False, 
        output_hidden_states = False,
    )
    return model

def ret_optim(model):
    print('Learning_rate = ',learning_rate )
    optimizer = AdamW(model.parameters(),
                      lr = learning_rate, 
                      eps = 1e-8 
                    )
    return optimizer

def ret_dataloader():
    print('batch_size = ', batch_size)
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader,validation_dataloader

from transformers import get_linear_schedule_with_warmup

num_epochs = 4

def ret_scheduler(dataloader,optimizer):
    epochs = num_epochs
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    return scheduler

import random
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#training starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ret_model()
model.to(device)
#wandb.init(config=sweep_defaults)
train_dataloader,validation_dataloader = ret_dataloader()
optimizer = ret_optim(model)
scheduler = ret_scheduler(train_dataloader,optimizer)

#print("config ",wandb.config.learning_rate, "\n",wandb.config)
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []
epochs = num_epochs
# For each epoch...
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        #print("ok")
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        loss, logits = outputs['loss'], outputs['logits']
        total_train_loss += loss.item()
        avg_train_loss = loss.item()
        #print("Loss after each batch : ", avg_train_loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            

    # Measure how long this epoch took.

    #wandb.log({'avg_train_loss':avg_train_loss})

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("")
    print("Running Validation...")

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                  token_type_ids=None, 
                                  attention_mask=b_input_mask,
                                  labels=b_labels)
            loss, logits = outputs['loss'], outputs['logits']

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.9f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    print("  Validation Loss: {0:.9f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy
        }
    )

print("")
print("Training complete!")

import csv

dataframe_val_x = pd.read_csv(os.path.join(directory,'comp_test_x.csv'))

input_ids = []
attention_masks = []

sentences = dataframe_val_x['Title'].values
labels = dataframe_val_x['Id'].values

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = 64,  pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

dataset2 = TensorDataset(input_ids, attention_masks, labels)

test_dataloader = DataLoader(
            dataset2, # The validation samples.
            sampler = SequentialSampler(dataset2), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

lis = []
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    with torch.no_grad():        
        outputs = model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask,
                              labels=None)
        logits = outputs['logits']

    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    pred_flat = np.argmax(logits, axis=1).flatten()
    lis += pred_flat.tolist()

predicted_vals = []
iter = 0

for x in lis:
    predicted_vals.append((iter, x))
    iter += 1

header = ['Id','Genre']
directory_out = '/kaggle/working/'
with open(os.path.join(directory_out,'output.csv'), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    # write a row to the csv file
    writer.writerows(predicted_vals)
