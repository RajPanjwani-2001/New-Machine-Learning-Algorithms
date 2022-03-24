#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:18:29 2021

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

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(x,y,test_size=0.2)

import lightgbm as lgb

model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(X_train,Y_train)
lgb.plot_importance(model)

from sklearn.metrics import precision_score,recall_score,accuracy_score
#Checking OverFitting
print('Training accuracy {:.4f}'.format(model.score(X_train,Y_train)))
print('Testing accuracy {:.4f}'.format(model.score(X_test,Y_test)))

y_pred = model.predict(X_test)

print("Precision = {}".format(precision_score(Y_test, y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, y_pred)))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, y_pred)
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
cmd_obj.plot()
cmd_obj.ax_.set(
                title='LightGBM', 
                xlabel='Predicted', 
                ylabel='Actual')
plt.show()

lgb.plot_tree(model,figsize=(30,40))
