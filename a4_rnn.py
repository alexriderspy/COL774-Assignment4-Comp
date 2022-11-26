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
import csv

class TextLoader(Dataset):
    def __init__(self, dataframe_x, dataframe_y, sentence_length, embedding_dim):
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.sentence_length = sentence_length

        glove = torchtext.vocab.GloVe(name='6B', dim = embedding_dim)
        tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch

        def to_embedding(sentence):
            tokens = tokenizer(sentence)
            tokens = (tokens +[""] * (self.sentence_length-len(tokens))) if len(tokens)<self.sentence_length else tokens[:self.sentence_length] 
            return glove.get_vecs_by_tokens(tokens) 

#         to_embedding_vec = np.vectorize(to_embedding) 

        titles = dataframe_x['Title'].to_numpy()
        embeddings = torch.zeros(len(titles), self.sentence_length, embedding_dim).to(device)
        
        for i in range(len(titles)):
            embeddings[i] = to_embedding(titles[i]).to(device)

#         self.embedded_x = torch.tensor(to_embedding_vec(titles)).to(device)

        self.embedded_x = embeddings
        

    def __len__(self):
        return len(self.dataframe_x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
#         textKey = self.dataframe_x.iloc[idx,2].to(device) 
        embeddings = self.embedded_x[idx].to(device)
        labelKey = self.dataframe_y.iloc[idx, 1]
        label = (torch.tensor(int(labelKey)).to(device))

        return embeddings, label

def get_output_shape(model, input_dim):
    rand_input = torch.rand(1, input_dim).to(device)
    return model(rand_input)[0].shape

sentence_length = 15
batch_size = 200  
embedding_dim = 300
input_dim = 300 
hidden_dim = 128 
layers = 1 
outputs = 30
class RecurrentNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layers, outputs):
        super(RecurrentNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim , hidden_dim, layers, bidirectional=True, batch_first=True).to(device)
        rnn_out = get_output_shape(self.rnn, input_dim)
        flattened_size = np.prod(list(rnn_out))
        self.fc1 = nn.Linear(flattened_size,hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, outputs).to(device)
        
    def forward(self, x):
        h0 = (torch.zeros(2 * self.layers, batch_size, self.hidden_dim).to(device))
        out, hn = self.rnn(x)
        x = torch.tanh(self.fc1(out[:, -1, :])).to(device)
        x = (self.fc2(x)).to(device)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device=', device)

num_epochs = 50
learning_rate = 0.01

directory = '/kaggle/input/col774-2022/'
dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv'))
dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv'))
dataset = TextLoader(dataframe_x = dataframe_x, dataframe_y = dataframe_y, sentence_length = sentence_length, embedding_dim = embedding_dim)

model = RecurrentNet(input_dim, hidden_dim, layers, outputs).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
for epoch in tqdm.tqdm(range(num_epochs)):
    for i, (titles, labels) in enumerate(dataloader):
        titles = titles.to(device)
        labels = labels.to(device)
        outputs = model(titles).to(device)
        loss = criterion(outputs, labels).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%50 == 0:
            print(f'Epoch = {epoch}, Loss = {loss.item()}')

# tokenizer = get_tokenizer("basic_english")

# def numWords(sentence):
#     return len(tokenizer(sentence))

# numWords_vec = np.vectorize(numWords) 

# titles = dataframe_x['Title'].to_numpy()
# words = numWords_vec(titles)
# print(f'mean = {np.mean(words)}, max = {np.amax(words)}, min = {np.amin(words)}, median = {np.median(words)}' )