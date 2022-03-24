from MachineLearning import FeatureEngineering
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif,SelectKBest,SelectPercentile,RFE,chi2
import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')
obj = FeatureEngineering()

df = obj.data_handling(df)
data = df.values

x = data[:,:-1]
y = data[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2)

bn = 24
X_train = obj.binning_quatile(X_train,bn)
X_test = obj.binning_quatile(X_test,bn)

'''info_gain = np.zeros(x.shape[1])
for i in range(500):
    info_gain = info_gain + mutual_info_classif(x,y)
    info_gain_bin = mutual_info_classif(x1,y)

info_gain = info_gain/500
print('Before binning: ',info_gain)

info_gain_bin = info_gain_bin/500
print('After binning: ',info_gain_bin)'''


sel_col = SelectKBest(chi2,k= 1).fit(X_train,Y_train)
#sel_col = RFE(estimator=xgboost,n_features_to_select=3,step=1)
X_train_f = sel_col.transform(X_train)
X_test_f = sel_col.transform(X_test)

xgb = xgboost.XGBClassifier(use_label_encoder = True,eval_metric='logloss')
xgb.fit(X_train_f,Y_train)

y_pred = xgb.predict(X_test_f)

print(accuracy_score(Y_test,y_pred))





    

