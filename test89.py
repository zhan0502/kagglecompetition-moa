#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

 
import pandas as pd
import numpy as np
 
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection, decomposition, preprocessing, pipeline


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
X = train_features
y = train_target 
test_features = pd.read_csv('data/test_features.csv')
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_type)],axis=1)
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_time)],axis=1)
test_features = pd.concat([test_features, pd.get_dummies(test_features.cp_dose)],axis=1)
test_features = test_features.drop(['cp_type','cp_time','cp_dose'], axis =1)
test_features = test_features.iloc[:, 1:].values
pred = np.empty([test_features.shape[0],train_target.shape[1]])


# In[3]:



scl = preprocessing.StandardScaler()
pca = decomposition.PCA()
rf = ensemble.RandomForestClassifier(n_jobs=-1)
import time

classifier = pipeline.Pipeline([ ("scaling", scl), ("pca", pca), ("rf",rf) ])
total_duration = 0
for i in range(train_target.shape[1]):
    if 71<= i <=75 or 111<=i<=113 or 149<=i<=151  or 185<=i<=189:

        start = time.time()
        y_s = y[:,i]
        classifier = ensemble.RandomForestClassifier(n_jobs=-1)
        param_grid = {
            "n_estimators": np.arange(100, 500),
            "max_depth": np.arange(2,20),
            "criterion": ["gini", "entropy"],
        }

        model = model_selection.RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_grid,
            n_iter = 2,
            scoring="accuracy",
            verbose=10,
            n_jobs=1,
            cv=5,
        )
        #model.fit(X, y)
        model.fit(X,y_s)
        stop = time.time()
        duration = stop - start

        total_duration = total_duration + duration
        print(i, duration/60, model.best_score_)
        print(model.best_estimator_.get_params())
        # In[ ]:

        p = model.best_estimator_.predict_proba(test_features)
        pred[:, i]=p[:,1]

        if i == 189 or total_duration/60 > 450:
            print(i, 'reaching time line, please continue next time')
            test_features1 = pd.read_csv('data/test_features.csv')
            output_frame = pd.DataFrame({'sig_id': test_features1.iloc[:,0].values})
            for col in train_targets_scored.columns[1:]:
                output_frame[col] =0
            output_frame.iloc[:,1:]= pred 
            output_frame.to_csv('test82_randomforest_p7.csv', index = False)
            break
test_features1 = pd.read_csv('data/test_features.csv')
output_frame = pd.DataFrame({'sig_id': test_features1.iloc[:,0].values})
for col in train_targets_scored.columns[1:]:
        output_frame[col] =0
output_frame.iloc[:,1:]= pred 
output_frame.to_csv('test82_randomforest_p7.csv', index = False)
             
   


# In[ ]:


import pickle
with open('random_forest.pickle', 'wb') as f:
    pickle.dump(model, f)
 


# In[ ]:




