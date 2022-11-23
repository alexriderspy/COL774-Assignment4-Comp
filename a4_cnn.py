import torch
import torchvision
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
        #print(image.shape)
        if self.transform:
            image = self.transform(image)
            image = (image - torch.mean(image))/ torch.std(image)
        #print(image.shape)

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

        self.fc1 = nn.Linear(self.fc1_in//batch_size,128)

        self.fc2 = nn.Linear(128,30)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        #print("shapes")
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool3(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(batch_size,self.fc1_in//batch_size)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 4

learning_rate = 0.001

transform = transforms.Compose(
    [#transforms.Resize((size,size)),
    transforms.ToTensor(),
    #transforms.Normalize()
    ]

)

directory = '/kaggle/input/col774-2022/'

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

dataset = ImageLoader(dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv')), dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv')), root_dir = os.path.join(directory, 'images/images/'), transform = transform)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
for epoch in tqdm.tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        #print(images)
        #print(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print("ok")
        if (i+1)%2000 == 0:
            print(f'Epoch = {epoch}, Loss = {loss.item()}')