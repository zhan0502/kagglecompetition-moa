#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Note this project is runing in py38 environment 
# use conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# import library 
from math import cos, pi 
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.utils import data
from sklearn.decomposition import PCA


# In[5]:


train_features = pd.read_csv('data/train_features.csv')

train_targets_scored = pd.read_csv('data/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('data/train_targets_nonscored.csv')
sample_submissions = pd.read_csv('data/sample_submission.csv')
# preprocess data for dose, time and type 
train_features = pd.concat([train_features, pd.get_dummies(train_features.cp_type)],axis=1)
train_features = pd.concat([train_features, pd.get_dummies(train_features.cp_time)],axis=1)
train_features = pd.concat([train_features, pd.get_dummies(train_features.cp_dose)],axis=1)
train_features = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
train_dataset = train_features.iloc[:, 1:].values
train_targets = train_targets_scored.iloc[:, 1:].values
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[6]:


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.fc1 = nn.Linear(16, 2048)
        self.batch_norm1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 2048)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 206)


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x)) 
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        return x
model1 = Model1()
model1.to(device)

class Model2(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x - batch size, seq, input_size
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0=torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)

        out, _=self.lstm(x, (h0, c0))
        #out: batch_size, seq_length, hidden_size
        #out(N, 3, 128)
        out = out[:, -1, :]
        #OUT(N, 128)
        out = self.fc(out)
        #out = torch.sigmoid(out)
        return out
    
model2 = Model2(input_size=16, hidden_size=128,num_layers=3, num_classes=206)
model2.to(device)


# In[7]:


class Model3(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model3, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x - batch size, seq, input_size
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0=torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)

        out, _=self.lstm(x, (h0, c0))
        #out: batch_size, seq_length, hidden_size
        #out(N, 3, 128)
        out = out[:, -1, :]
        #OUT(N, 128)
        out = self.fc(out)
        #out = torch.sigmoid(out)
        return out
    
model3 = Model3(input_size=8, hidden_size=128,num_layers=2, num_classes=206)
model3.to(device)
    
class Model4(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model4, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x - batch size, seq, input_size
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0=torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)

        out, _=self.lstm(x, (h0, c0))
        #out: batch_size, seq_length, hidden_size
        #out(N, 3, 128)
        out = out[:, -1, :]
        #OUT(N, 128)
        out = self.fc(out)
        #out = torch.sigmoid(out)
        return out

model4 = Model4(input_size=4, hidden_size=128,num_layers=2, num_classes=206)
model4.to(device)

 


# In[8]:


class Model5(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model5, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # x - batch size, seq, input_size
        self.batch_norm1 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        h0=torch.zeros(self.num_layers*2, x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers*2, x.size(0),self.hidden_size).to(device)

        out, _=self.lstm(x, (h0, c0))
        #out: batch_size, seq_length, hidden_size
        #out(N, 3, 128)
        out = out[:, -1, :]
        #OUT(N, 128)
        out = self.batch_norm1(out)
        out = self.dropout1(out)
        out = self.fc(out)
        #out = torch.sigmoid(out)
        return out

model5 =  Model5(input_size=16, hidden_size=128, num_layers=4, num_classes=206)
model5.to(device)


class Model6(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model6, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # x - batch size, seq, input_size
        self.batch_norm1 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        h0=torch.zeros(self.num_layers*2, x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers*2, x.size(0),self.hidden_size).to(device)

        out, _=self.lstm(x, (h0, c0))
        #out: batch_size, seq_length, hidden_size
        #out(N, 3, 128)
        out = out[:, -1, :]
        #OUT(N, 128)
        out = self.batch_norm1(out)
        out = self.dropout1(out)
        out = self.fc(out)
        #out = torch.sigmoid(out)
        return out

model6 =  Model6(input_size=16, hidden_size=128, num_layers=3, num_classes=206)
model6.to(device)


# In[9]:


from torchvision.models.resnet import resnet50
model7 =  resnet50(pretrained=False)
num_features =  model7.fc.in_features
model7.fc = nn.Linear(num_features , 206)
model7.to(device)

from torchvision.models.resnet import resnet34
model8 =  resnet34(pretrained=False)
num_features =  model8.fc.in_features
model8.fc = nn.Linear(num_features, 206)
model8.to(device)


# In[10]:


MODEL = [model1,model2, model3, model4, model5, model6, model7, model8]


# In[11]:


def train_model1(model,train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w
    pca = PCA(n_components=16)
    pca.fit(train_features.iloc[:, 1:])
 
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            #train = train.reshape(train.shape[0], sequence_length,input_size)
            train, target = train.to(device), target.to(device)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            #val = val.reshape(val.shape[0], sequence_length,input_size)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
             
          

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                  f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                      , str(round(duration, 4))+"s.")

    return ave_loss, val_loss

def train_model2(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w

    pca = PCA(n_components=128)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 16
    sequence_length = 8
    num_layers = 3
    hidden_size = 128
    num_classes = 206 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], sequence_length,input_size)
            train, target = train.to(device), target.to(device)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], sequence_length,input_size)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
                
            

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                    f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                        , str(round(duration, 4))+"s.")

    return ave_loss, val_loss

# In[12]:


def train_model3(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w

    pca = PCA(n_components=32)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 8
    sequence_length = 4
    num_layers = 3
    hidden_size = 128
    num_classes = 206 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], sequence_length,input_size)
            train, target = train.to(device), target.to(device)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], sequence_length,input_size)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
                
            

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                    f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                        , str(round(duration, 4))+"s.")

    return ave_loss, val_loss


def train_model4(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w
    pca = PCA(n_components=16)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 4
    sequence_length = 4
    num_layers = 3
    hidden_size = 128
    num_classes = 206 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], sequence_length,input_size)
            train, target = train.to(device), target.to(device)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], sequence_length,input_size)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
                
            

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                    f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                        , str(round(duration, 4))+"s.")

    return ave_loss, val_loss


# In[13]:


def train_model5(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w
    pca = PCA(n_components=256)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 16
    sequence_length = 16
 
 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], sequence_length,input_size)
            train, target = train.to(device), target.to(device)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], sequence_length,input_size)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
             
          

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                  f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                      , str(round(duration, 4))+"s.")

    return ave_loss, val_loss
def train_model6(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w

    pca = PCA(n_components=128)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 16
    sequence_length = 8
 
 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], sequence_length,input_size)
            train, target = train.to(device), target.to(device)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], sequence_length,input_size)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
             
          

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                  f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                      , str(round(duration, 4))+"s.")

    return ave_loss, val_loss
 


# In[14]:


def train_model7(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w

    pca = PCA(n_components=507)
    pca.fit(train_features.iloc[:, 1:])
 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], 3,13,13)
            train, target = train.to(device), target.to(device)
            #print(train.shape)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], 3,13,13)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
             
          

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                  f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                      , str(round(duration, 4))+"s.")

    return ave_loss, val_loss

def train_model8(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate):
    class BCELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred_logits, target):
                w = torch.zeros(target.shape)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        if target[i][j] == 1:
                            w[i][j] == 1
                        elif target[i][j] == 0:
                            w[i][j]= (1-0.0034)/0.0034        
                w=w.to(device)
                ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
                return ce * w

    pca = PCA(n_components=432)
    pca.fit(train_features.iloc[:, 1:])
 
    
    x_train_fold = torch.tensor(pca.transform(train_dataset[train_idx])) 
    y_train_fold = torch.tensor(train_targets[train_idx]) 
    x_val_fold = torch.tensor(pca.transform(train_dataset[valid_idx])) 
    y_val_fold = torch.tensor(train_targets[valid_idx]) 

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    train_len=train_idx.shape[0]
    valid_len=valid_idx.shape[0]

        # create a new optimizer at the beginning of each epoch: give the current learning rate.  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
    criterion = BCELoss()
    criterion_val = BCELoss()
    
    loss_stopper=[]
    running_loss = 0
    running_val_loss= 0 
    start = time.time()
    for train, target in train_loader:
            model.zero_grad() #wrong
            train = train.reshape(train.shape[0], 3,12,12)
            train, target = train.to(device), target.to(device)
            #print(train.shape)
            train = train.float()
            target =  target.float()

            output = model.forward(train)
            loss = criterion(output, target)    
            loss.sum().backward()
            optimizer.step() 
            running_loss = running_loss + loss.sum().detach().item()

    for val, val_target in valid_loader:
            val = val.reshape(val.shape[0], 3,12,12)
            val, val_target = val.to(device), val_target.to(device)
            val = val.float()
            val_target = val_target.float()
            output_val = model.forward(val)
            val_loss = criterion_val(output_val,val_target)   
            running_val_loss = running_val_loss + val_loss.sum().detach().item()
             
          

    stop = time.time()
    duration = stop-start

    ave_loss = running_loss/(train_len*206) 
    val_loss = running_val_loss/(valid_len*206) 
    
    print(f"Epoch {epoch+1}/{num_epochs}.. "
                        f"Train loss: {ave_loss:.10f}.. "
                                  f"validation loss: {val_loss:.10f}.. "
                            #  f"Epoch {acc}.. "

                      , str(round(duration, 4))+"s.")

    return ave_loss, val_loss


# In[15]:


TRAIN_MODEL = [train_model1, train_model2, train_model3, train_model4, train_model5,train_model6, train_model7,
              train_model8]
import os
import random
 


# In[16]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
n_splits=len(MODEL)
 
mskf = MultilabelStratifiedKFold(n_splits) 
batch_size = 512
shuffled_indices=torch.randperm(len(MODEL))
num_epochs = 50


lr=0.0002    
for i, (train_idx, valid_idx) in enumerate(mskf.split(X=train_dataset,y=train_targets)):
 
    train_loss_epoch =[]
    valid_loss_epoch =[]
    
    for epoch in range(num_epochs):
        if epoch < 30:
            lrs = np.linspace(0, lr, 31)[1:]
            learning_rate = lrs[epoch]
        else:
            learning_rate= (1+ cos(epoch*pi/num_epochs))*1/2*lr
    
        x_train_fold = train_dataset[train_idx]
        y_train_fold = train_targets[train_idx] 
        x_val_fold = train_dataset[valid_idx] 
        y_val_fold = train_targets[valid_idx]
        
        model_idx = shuffled_indices[i].item()
        print(f'Fold {i + 1}',"training model", model_idx)
      
        
        model = MODEL[model_idx]
        train_model = TRAIN_MODEL[model_idx]
        
 
        ave_loss, val_loss=  train_model(model, train_features, x_train_fold, y_train_fold, x_val_fold, y_val_fold, learning_rate)
 
        valid_loss_epoch.append(val_loss)
        train_loss_epoch.append(ave_loss)
        


        if epoch> 4 and valid_loss_epoch[epoch] > valid_loss_epoch[epoch-1] and valid_loss_epoch[epoch-1] > valid_loss_epoch[epoch- 2]:
            torch.save(model, str(model_idx) +'_test87.pth')
            break
        
        if epoch == num_epochs-1:
            torch.save(model, str(model_idx) +'_test87.pth')
    plt.figure(figsize=(5,5))
    plt.xlim(0, epoch)
    plt.ylim(0, 1) 
    plt.title("Train Loss vs. Validation Loss")
    plt.plot(train_loss_epoch,label="Training") 
    plt.plot(valid_loss_epoch,label="Validation") 
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.savefig(str(model_idx) + '_test87.png', dpi=300, bbox_inches='tight')
    plt.show()
    #print(valid_loss/num_epochs)


# In[ ]:


# preprocess data for dose, time and type 
test_features = pd.read_csv('data/test_features.csv')
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_type)],axis=1)
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_time)],axis=1)
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_dose)],axis=1)
test_features = test_features.drop(['cp_type','cp_time','cp_dose'], axis =1)


# In[ ]:


def test_model1(model,test_features):
    
    pca = PCA(n_components=16)
    pca.fit(train_features.iloc[:, 1:])
  
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )   
  
    ########output result
    predictions = np.zeros([test_features.shape[0], 206])
 
    #######################
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
        predictions[i*512:(i+1)*512] = p
    
    print("Model1 inferrence done!")

    return predictions

def test_model2(model,test_features):
    
    pca = PCA(n_components=128)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 16
    sequence_length = 8
    num_layers = 3
    hidden_size = 128
    num_classes = 206 
  
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )   
  
    ########output result
    predictions = np.zeros([test_features.shape[0], 206])
 
    #######################
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], sequence_length,input_size)
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
        predictions[i*512:(i+1)*512] = p
    
    print("Model2 inferrence done!")

    return predictions

def test_model3(model,test_features):
     
    pca = PCA(n_components=32)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 8
    sequence_length = 4
    
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )  
     ########output result
    predictions = np.zeros([test_features.shape[0], 206])
     #######################
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], sequence_length,input_size)
        #print(testdata.shape)
        test = testdata.to(device)
        
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
        predictions[i*512:(i+1)*512] = p
    print("Model3 inferrence done!")        
    return predictions
 

def test_model4(model,test_features):

    pca = PCA(n_components=16)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 4
    sequence_length = 4
    
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )  
     ########output result
    predictions = np.zeros([test_features.shape[0], 206])
     #######################
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], sequence_length,input_size)
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
        predictions[i*512:(i+1)*512] = p
    print("Model4 inferrence done!")        
    return predictions

def test_model5(model,test_features):

    pca = PCA(n_components=256)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 16
    sequence_length = 16
    
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )  
     ########output result
    predictions = np.zeros([test_features.shape[0], 206])
     #######################
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], sequence_length,input_size)
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
        predictions[i*512:(i+1)*512] = p
    print("Model5 inferrence done!")        
    return predictions

def test_model6(model,test_features):

    pca = PCA(n_components=128)
    pca.fit(train_features.iloc[:, 1:])
    input_size = 16
    sequence_length = 8
 
    
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )  
     ########output result
    predictions = np.zeros([test_features.shape[0], 206])
     #######################
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], sequence_length,input_size)
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
        predictions[i*512:(i+1)*512] = p
    print("Model6 inferrence done!")        
    return predictions

 
def test_model7(model,test_features):
    

    pca = PCA(n_components=507)
    pca.fit(train_features.iloc[:, 1:])
  
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )   
    ########output result
    predictions = np.zeros([test_features.shape[0], 206])
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], 3, 13, 13)
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
       #p_back = np.where(p<0.5,p-eps, p+eps)
        predictions[i*512:(i+1)*512] = p 
    print("Model7 inferrence done!")    
    return predictions

def test_model8(model,test_features):
    

    pca = PCA(n_components=432)
    pca.fit(train_features.iloc[:, 1:])
  
    test_dataset_tensor = torch.Tensor(pca.transform(test_features.iloc[:, 1:].values))
    test_dataset = data.TensorDataset(test_dataset_tensor)
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 512, num_workers =2, shuffle = False, 
                                    )   
    ########output result
    predictions = np.zeros([test_features.shape[0], 206])
    for i, testdata in enumerate(test_dataloader):
        #note here the testdata is a list element and the element in the lsit is the tensor
        testdata = torch.Tensor(testdata[0])
        testdata = testdata.reshape(testdata.shape[0], 3, 12, 12)
        test = testdata.to(device)
        p = model(test)
        p = torch.sigmoid(p).detach().cpu().numpy()
       #p_back = np.where(p<0.5,p-eps, p+eps)
        predictions[i*512:(i+1)*512] = p 
    print("Model8 inferrence done!")    
    return predictions

 
prediction1=test_model1(model1,test_features)
prediction2=test_model2(model2,test_features)
prediction3=test_model3(model3,test_features)
prediction4=test_model4(model4,test_features)
prediction5=test_model5(model5,test_features)
prediction6=test_model6(model6,test_features)
prediction7=test_model7(model7,test_features)
prediction8=test_model8(model8,test_features)
 

predictions = [prediction1,prediction2,prediction3,prediction4,prediction5,prediction6,prediction7,prediction8]
predictions_ave =  sum(predictions)/len(predictions )

test_features1 = pd.read_csv('data/test_features.csv')
output_frame = pd.DataFrame({'sig_id': test_features1.iloc[:,0].values})

for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= predictions_ave
output_frame.to_csv('test87_ave.csv', index = False)

 
for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction1
output_frame.to_csv('test87_model1.csv', index = False)
 

for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction2
output_frame.to_csv('test87_model2.csv', index = False)

for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction3
output_frame.to_csv('test87_model3.csv', index = False)
 
for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction4
output_frame.to_csv('test87_model4.csv', index = False)

for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction5
output_frame.to_csv('test87_model5.csv', index = False)
 
for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction6
output_frame.to_csv('test87_model6.csv', index = False)

for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction7
output_frame.to_csv('test87_model7.csv', index = False)

for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= prediction8
output_frame.to_csv('test87_model8.csv', index = False)
 


# In[ ]:


####################weighted average###############################
#ave= 0.1*out0 + 0.2*out1+ 0.4*out2+ 0.3*out3


# In[ ]:





# In[ ]:




