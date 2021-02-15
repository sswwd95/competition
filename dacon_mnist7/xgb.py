#xgb 로 기준치 잡고 딥러닝으로 완성

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('../dacon7/train.csv')
test = pd.read_csv('../dacon7/test.csv')
sub = pd.read_csv('../dacon7/submission.csv')

x_train = train.drop(['id', 'digit', 'letter'],axis=1)
X_test = test.drop(['id', 'letter'],axis=1)
y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):
    y_train[i, digit] = 1
print(y_train.shape)# (2048,10)

x_train = x_train.values
X_test =X_test.values

print(x_train.shape) #(2048, 784)
print(X_test.shape) #(20480, 784)
print(y.shape) #(2048,)

'''
x= np.append(x_train, X_test, axis=0)
print(x.shape) # (22528, 784)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum>=0.99)+1
print('cumsum >=0.99', cumsum>=0.99)
print('d : ', d) #d :  305
'''
pca = PCA(n_components=305)
x2_train =pca.fit_transform(x_train)
print(x2_train.shape) #(22528, 305)

x_train, x_test, y_train, y_test = train_test_split(
    x2_train, y, random_state=77, shuffle=True, train_size=0.8
)

# parameters = [
#     {'n_estimators' : [500,800,1000],
#     'learning_rate' : [0.1,0.01,0.001],
#     'max_depth' : [6,8,10],
#     'colsample_bytree' : [0.6,0.8,1]},
# ]

model =XGBClassifier(n_jobs=8,  cv=5)

model.fit(x_train,y_train, eval_metric='mlogloss')

y_pred = model.predict(x_test)
print('y_pred : ', model.score(x_test, y_test))


