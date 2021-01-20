#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorch_tabnet.tab_model import TabNetRegressor
import torch.nn as nn
import torch
import torch.nn.functional as F
 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
   
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from matplotlib import pyplot as plt
device=torch.device('cuda')


# In[2]:


train_features = pd.read_csv('data/train_features.csv')

train_targets_scored = pd.read_csv('data/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('data/train_targets_nonscored.csv')
sample_submissions = pd.read_csv('data/sample_submission.csv')

# preprocess data for dose, time and type 
train_features = pd.concat([train_features, pd.get_dummies(train_features.cp_type)],axis=1)
train_features = pd.concat([train_features, pd.get_dummies(train_features.cp_time)],axis=1)
train_features = pd.concat([train_features, pd.get_dummies(train_features.cp_dose)],axis=1)
train_features = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
train_features = train_features.iloc[:, 1:].values
#train_features
train_target =train_targets_scored.iloc[:, 1:].values


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_features, train_target, test_size = 0.1, 
                                                                                   random_state=1)
y_train_1 = y_train[:, 1].reshape(-1, 1)
y_val_1 = y_val[:, 1].reshape(-1, 1)


# In[4]:


test_features = pd.read_csv('data/test_features.csv')
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_type)],axis=1)
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_time)],axis=1)
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_dose)],axis=1)
test_features = test_features.drop(['cp_type','cp_time','cp_dose'], axis =1)
test_features = test_features.iloc[:, 1:].values
pred = np.empty([test_features.shape[0],train_target.shape[1]])


# In[ ]:

import time
max_epochs = 80
for i in range(train_target.shape[1]):
    start = time.time()
    train_target_s = y_train[:,i].reshape(-1, 1)
    valid_target_s = y_val[:,i].reshape(-1, 1)
    clf = TabNetRegressor()
    clf.fit(
        X_train=X_train, y_train=train_target_s,
        eval_set=[(X_train, train_target_s), (X_val, valid_target_s)],
        eval_name=['train', 'valid'],
        max_epochs=max_epochs,
        patience=80,
        batch_size=4096, virtual_batch_size=256,
        num_workers=0,
        #loss_fn=my_loss_fn,
        drop_last=False

    ) 
    p = clf.predict(test_features)
    pred[:, i]=p.reshape(-1) 

    stop = time.time()
    print(stop - start)
# In[61]:
 
test_features1 = pd.read_csv('data/test_features.csv')
output_frame = pd.DataFrame({'sig_id': test_features1.iloc[:,0].values})
for col in train_targets_scored.columns[1:]:
    output_frame[col] =0
output_frame.iloc[:,1:]= abs(pred) 
output_frame.to_csv('test74_tabnet.csv', index = False)
torch.save(clf, 'test74_tabnet.pth')


# In[ ]:




