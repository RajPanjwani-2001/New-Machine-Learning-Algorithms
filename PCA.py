#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:07:19 2021

@author: raj
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')
df['gender'] = df['gender'].map({'Male': 0,'Female': 1})
df['is_patient'] = df['is_patient'].map({1: 0,2: 1})

df.isna()
df = df.fillna(0)

x = df.drop(['is_patient'],axis='columns')
y = df['is_patient']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

scaled_data = scaler.transform(x)

from sklearn.decomposition import PCA

pca_obj = PCA(n_components=2)
pca_obj.fit(scaled_data)
x_pca = pca_obj.transform(scaled_data)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(x_pca,y,test_size=0.2)

import xgboost as xgb
#converting to DMatrix

DTrain = xgb.DMatrix(X_train,label=Y_train)
DTest = xgb.DMatrix(X_test,label=Y_test)
(DTrain)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 2} 

'''multi:softprob: 
same as softmax, but output a vector of 
ndata * nclass, which can be further reshaped to ndata * nclass matrix. 
The result contains predicted probability of each data point belonging to each class.
'''

steps = 100 

model = xgb.train(param, DTrain, steps)
xgb.plot_importance(model)
xgb.plot_tree(model)

import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score

y_pred = model.predict(DTest)
best_pred = np.asarray([np.argmax(line) for line in y_pred])

print("Precision = {}".format(precision_score(Y_test, best_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_pred)))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, best_pred)
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
cmd_obj.plot()
cmd_obj.ax_.set(
                title='XGBoost', 
                xlabel='Predicted', 
                ylabel='Actual')
plt.show()



