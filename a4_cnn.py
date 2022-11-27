import sys
import torch
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

#Class for loading images

class ImageLoader(Dataset):
    def __init__(self, dataframe_x, dataframe_y, root_dir, transform = None):
        self.root_dir = root_dir
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.transform = transform

    def __len__(self):
        return len(self.dataframe_x)

    #returns item based on index
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.dataframe_x.iloc[idx,1])
        image = np.array(cv2.imread(img_name))/255.0
        
        if self.transform:
            image = self.transform(image)
            image = (image - torch.mean(image))/ torch.std(image)

        if self.dataframe_y != None:
            labelKey = self.dataframe_y.iloc[idx, 1]
            label = torch.tensor(int(labelKey))
            return image, label
        else:
            return image, None            

#auxiliary function to get the output shape of first fc layer
def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

#best batch size
batch_size = 2
size = 224

class ConvNet(nn.Module):
    def __init__(self):

        super(ConvNet, self).__init__()
        self.expected_input_shape = (batch_size,3,size,size)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2,1)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool3 = nn.MaxPool2d(2,1)
        conv1_out = get_output_shape(self.pool1, get_output_shape(self.conv1, self.expected_input_shape))
        conv2_out = get_output_shape(self.pool2, get_output_shape(self.conv2, conv1_out)) 
        conv3_out = get_output_shape(self.pool3, get_output_shape(self.conv3, conv2_out)) 
        self.fc1_in = np.prod(list(conv3_out))

        self.fc1 = nn.Linear(self.fc1_in//batch_size,128)

        self.fc2 = nn.Linear(128,30)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(batch_size,self.fc1_in//batch_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 4

learning_rate = 0.001

transform = transforms.Compose(

    [
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]

)

#directory = '/kaggle/input/col774-2022/'
directory = sys.argv[2]

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

dataset = ImageLoader(dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv')), dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv')), root_dir = os.path.join(directory, 'images/images/'), transform = transform)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
for epoch in (range(num_epochs)):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        outputs = model(images)

        #loss computed using Cross Entropy Function
        loss = criterion(outputs, labels)

        #to reinitialise the weights and biases to 0 before we do backpropagation
        optimizer.zero_grad()

        #this is the backprop step
        loss.backward()

        #updates the parameters
        optimizer.step()
        
        # if (i+1)%2000 == 0:
        #     print(f'Epoch = {epoch}, Loss = {loss.item()}')

import csv 

dataset_test = ImageLoader(dataframe_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv')), dataframe_y = None, root_dir = os.path.join(directory, 'images/images/'), transform = transform)

predicted_vals = []
        
dataloader_test = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle=False, num_workers = 0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(30)]
    n_class_samples = [0 for _ in range(30)]

    iter = 0
    for images, _ in dataloader_test:
        images = images.to(device, dtype=torch.float)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        for i in range(batch_size):
            
            pred = predicted[i]
            predicted_vals.append([iter,pred.item()])
            iter += 1
header = ['Id','Genre']
#directory_out = '/kaggle/working/'

#with open(os.path.join(directory_out,'output.csv'), 'w') as f:
with open(os.path.join(directory,'non_comp_test_pred_y.csv'), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    # write a row to the csv file
    writer.writerows(predicted_vals)