import numpy as np
from numpy import inf, nan
import pandas as pd
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score
from MachineLearning import FeatureEngineering
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')
obj = FeatureEngineering()
df = obj.impute_mean(df)
df = pd.get_dummies(df, drop_first=True)

data = df.values
features = data[:, :-1]
cls_labels = data[:, -1]

avg_acc = 0
degree = 3
n_components = 20

X_train, X_test, Y_train, Y_test = train_test_split(features, cls_labels, test_size=0.3, shuffle=True)
for i in range(30):
    kern_obj = KernelPCA(n_components=n_components+i, degree=degree, kernel='poly')
    kern_obj.fit(X_train)
    X_train = kern_obj.transform(X_train)
    X_test = kern_obj.transform(X_test)

    X_train = np.nan_to_num(X_train, copy=True, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, copy=True, posinf=0, neginf=0)

    xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1, nthread=-1)
    xgb.fit(X_train, Y_train)
    Y_pred = xgb.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(Y_test, Y_pred)
    acc = accuracy_score(Y_test, Y_pred)
    avg_acc += acc
    print(p, r, f1, s, acc)