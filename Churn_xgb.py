#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 00:16:22 2021

@author: raj
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Telco-Customer-Churn.csv')

df = df.drop(['customerID'],axis = 'columns')

df.gender.unique()

df.isna().sum()

df.SeniorCitizen.unique()

df.Partner.unique()

#df.Partner = df.Partner.map({'Yes': 1, 'No' : 0})

df.Dependents.unique()

#df.Dependents = df.Dependents.map({'Yes' : 1, 'No' : 0})

df.PhoneService.unique()

#df.PhoneService = df.PhoneService.map({'Yes' : 1, 'No' : 0})

df.TotalCharges = pd.to_numeric(df.TotalCharges,errors='coerce')

x = df.drop(['Churn'],axis = 'columns')
y = df['Churn']

df.columns

x_encoded = pd.get_dummies(x, columns=['gender','SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       ])

y = y.map({'Yes':1,'No':0})

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_encoded,y,random_state=42,stratify=y)

sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

import xgboost as xgb

clf_xgb = xgb.XGBClassifier(objective='binary:logistic',missing = None, seed = 42)
clf_xgb.fit(X_train,y_train,
            verbose = True,
            early_stopping_rounds = 20,
            eval_metric = 'aucpr',
            eval_set = [(X_test,y_test)]
            )


