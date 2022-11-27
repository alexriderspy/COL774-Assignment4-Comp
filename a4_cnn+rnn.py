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
import math
import csv

directory = '/kaggle/input/col774-2022/'
dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv'))
dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device=', device)

class DatasetLoader(Dataset):
    def __init__(self, dataframe_x, dataframe_y, root_dir,sentence_length, embedding_dim,transform = None,):
        self.root_dir = root_dir
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.transform = transform
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
        
        img_name = os.path.join(self.root_dir, self.dataframe_x.iloc[idx,1])
        image = np.array(cv2.imread(img_name))/255.0
        embeddings = self.embedded_x[idx].to(device)
        labelKey = self.dataframe_y.iloc[idx, 1]
        label = torch.tensor(int(labelKey))
        if self.transform:
            image = self.transform(image)
            image = (image - torch.mean(image))/ torch.std(image)

        return image,embeddings,label

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class ImageLoader(Dataset):
    def __init__(self, dataframe_x, dataframe_y, root_dir, transform = None):
        self.root_dir = root_dir
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.transform = transform

    def __len__(self):
        return len(self.dataframe_x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.dataframe_x.iloc[idx,1])
        image = np.array(cv2.imread(img_name))/255.0
        #print("Image" + str(image))
        #image = Image.fromarray(image)
        labelKey = self.dataframe_y.iloc[idx, 1]
        label = torch.tensor(int(labelKey))

        if self.transform:
            image = self.transform(image)
            image = (image - torch.mean(image))/ torch.std(image)

        return image, label

# dataiter = iter(dataloader)
# data = dataiter.next()

# features, labels = data
#print(features, labels)

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

batch_size = 2
size = 224
class ConvNet(nn.Module):
    def __init__(self):

        super(ConvNet, self).__init__()
        #input channel, out channel, kernel size
        #batch_sz, channel, height, width
        self.expected_input_shape = (batch_size,3,size,size)
        self.conv1 = nn.Conv2d(3, 32, 5)
        #dim, stride
        #add padding to conv layer sizes are getting reduced
        self.pool1 = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2,1)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool3 = nn.MaxPool2d(2,1)
        #specify stride 
        # Calculate the input of the Linear layer
        conv1_out = get_output_shape(self.pool1, get_output_shape(self.conv1, self.expected_input_shape))
        conv2_out = get_output_shape(self.pool2, get_output_shape(self.conv2, conv1_out)) 
        conv3_out = get_output_shape(self.pool3, get_output_shape(self.conv3, conv2_out)) 
        self.fc1_in = np.prod(list(conv3_out)) # Flatten

        self.fc1 = nn.Linear(self.fc1_in//2,128)

        self.fc2 = nn.Linear(128,30)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        #print("shapes")
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool3(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(2,self.fc1_in//2)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        return x

num_epochs = 4

learning_rate = 0.001

transform = transforms.Compose(
    [#transforms.Resize((size,size)),
    transforms.ToTensor(),
    #transforms.Normalize()
    ]

)

model_cnn = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_cnn.parameters(), lr = learning_rate)

dataset = ImageLoader(dataframe_x = dataframe_x, dataframe_y = dataframe_y, root_dir = os.path.join(directory, 'images/images/'), transform = transform)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
for epoch in tqdm.tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        #print(images)
        #print(labels)
        outputs = model_cnn(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print("ok")
        if (i+1)%5000 == 0:
            print(f'Epoch = {epoch}, Loss = {loss.item()}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

num_epochs = 10
learning_rate = 0.01

dataset = TextLoader(dataframe_x = dataframe_x, dataframe_y = dataframe_y, sentence_length = sentence_length, embedding_dim = embedding_dim)

model_rnn = RecurrentNet(input_dim, hidden_dim, layers, outputs).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model_rnn.parameters(), lr = learning_rate)

dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
for epoch in tqdm.tqdm(range(num_epochs)):
    for i, (titles, labels) in enumerate(dataloader):
        titles = titles.to(device)
        labels = labels.to(device)
        outputs = model_rnn(titles).to(device)
        loss = criterion(outputs, labels).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%50 == 0:
            print(f'Epoch = {epoch}, Loss = {loss.item()}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = DatasetLoader(dataframe_x = dataframe_x, dataframe_y = dataframe_y, root_dir = os.path.join(directory, 'images/images/'), transform = transform, sentence_length = sentence_length, embedding_dim = embedding_dim)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(30)]
    n_class_samples = [0 for _ in range(30)]

    for i, (images, titles, labels) in enumerate(dataloader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
        titles = titles.to(device)
        labels = labels.to(device)
        outputs_rnn = model_rnn(titles).to(device)
        
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        images = images.to(device, dtype=torch.float)
        outputs_cnn = model_cnn(images).to(device)
        outputs = torch.mul(outputs_rnn, outputs_cnn).to(device)
        
        _, predicted = torch.max(outputs,1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(predicted)):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        
    acc = 100.0 * (n_correct/n_samples)
    print(f' Train Accuracy of network: {acc} %')
    
    for i in range(30):
        acc = 100.0 * (n_class_correct[i]/n_class_samples[i])
        print(f'Train Accuracy of classes[{i}]: {acc} %')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_test = TextLoader(dataframe_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv')), dataframe_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv')), root_dir = os.path.join(directory, 'images/images/'), transform = transform, sentence_length = sentence_length, embedding_dim = embedding_dim)
dataloader_test = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle=True, num_workers = 0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(30)]
    n_class_samples = [0 for _ in range(30)]

    for i, (images, titles, labels) in enumerate(dataloader_test):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
        titles = titles.to(device)
        labels = labels.to(device)
        outputs_rnn = model_rnn(titles).to(device)
        
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        images = images.to(device, dtype=torch.float)
        outputs_cnn = model_cnn(images).to(device)
        outputs = torch.mul(outputs_rnn, outputs_cnn).to(device)
        
        _, predicted = torch.max(outputs,1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(predicted)):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        
    acc = 100.0 * (n_correct/n_samples)
    print(f'Test Accuracy of network: {acc} %')
    
    for i in range(30):
        acc = 100.0 * (n_class_correct[i]/n_class_samples[i])
        print(f'Test Accuracy of classes[{i}]: {acc} %')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
predicted_vals = []
dataset_test_comp = TextLoader(dataframe_x = pd.read_csv(os.path.join(directory,'comp_test_x.csv')), dataframe_y = pd.read_csv(os.path.join(directory, 'sample_submission.csv')), root_dir = os.path.join(directory, 'images/images/'), transform = transform, sentence_length = sentence_length, embedding_dim = embedding_dim)
dataloader_test_comp = DataLoader(dataset = dataset_test_comp, batch_size = batch_size, shuffle = False, num_workers = 0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(30)]
    n_class_samples = [0 for _ in range(30)]

    print(f'no of training examples = {len(dataset_test_comp.dataframe_x)}')
    iters = 0
    for i, (images, titles, labels) in enumerate(dataloader_test_comp):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
        titles = titles.to(device)
        labels = labels.to(device)
        outputs_rnn = model_rnn(titles).to(device)
        
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        images = images.to(device, dtype=torch.float)
        outputs_cnn = model_cnn(images).to(device)
        outputs = torch.mul(outputs_rnn, outputs_cnn).to(device)
        
        _, predicted = torch.max(outputs,1)

        for j in range(len(predicted)):
            pred = predicted[j]
            predicted_vals.append([iters,pred.item()])
            iters += 1
            
header = ['Id','Genre']
directory_out = '/kaggle/working/'
with open(os.path.join(directory_out,'output.csv'), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    # write a row to the csv file
    writer.writerows(predicted_vals)                                        