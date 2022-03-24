from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest

df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')
lab_enc = LabelEncoder()
feat_sel = True

for i in range(df.shape[1]):
    if(df.iloc[:, i].isna().sum() != 0):
        print("Column contains NA value : " + str(i))
        df.iloc[:, i].fillna(df.iloc[:, i].mean(), inplace=True)

    if df.iloc[:, i].dtype == object:
        print("Column contains a categorical value: ", i)
        df.iloc[:, i] = lab_enc.fit_transform(df.iloc[:, i])
        print('Categories: ', df.iloc[:, i].unique())


x = df.drop(df.columns[-1], axis='columns')
y = df[df.columns[-1]]

for i in range(x.shape[1]):
        x.iloc[:, i] = (x.iloc[:, i] - x.iloc[:, i].mean())/ (x.iloc[:, i].std()**2)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape)


if feat_sel == True:
    sel_col = SelectKBest(mutual_info_classif, k=5).fit(X_train, Y_train)
    X_train_f = sel_col.transform(X_train)
    X_test_f = sel_col.transform(X_test)
    print('X_train:', X_train_f)
    print('X_test: ', X_test_f)
    print(sel_col.get_support(indices=True))






parameters_rs = {
    "n_estimators" : [10,20,50,100,150,200,250,300],
    "learning_rate" : [0.4,0.5,0.6,0.7],
     "max_depth"        : [ 3, 4, 5, 6],
     #"num_feature" : [5,6,11],
     "gamma"            : [0.1, 0.4,0.7,1.1],
     "reg_lambda" : [0,0.5,1,1.5,2,2.5,3],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }


model = xgb.XGBClassifier(use_label_encoder = True,eval_metric='logloss')

rs = RandomizedSearchCV(model, param_distributions= parameters_rs,n_iter=20,
                                   scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)
rs.fit(X_trainY_train)
score = rs.score(X_test_f, Y_test)
print(score)
new_params = rs.best_params_
print(new_params)
