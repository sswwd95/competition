import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./solar/csv/train.csv')
print(train.tail())
'''
        Day  Hour  Minute  DHI  DNI   WS     RH  T  TARGET
52555  1094    21      30    0    0  2.4  70.70 -4     0.0     
52556  1094    22       0    0    0  2.4  66.79 -4     0.0     
52557  1094    22      30    0    0  2.2  66.78 -4     0.0     
52558  1094    23       0    0    0  2.1  67.72 -4     0.0     
52559  1094    23      30    0    0  2.1  67.70 -4     0.0
'''
sub = pd.read_csv('./solar/csv/sample_submission.csv')

def preprocess_data(data, is_train=True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12-6)/6*np.pi/2) 
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])

    temp = data.copy()
    temp = temp[['TARGET','GHI','DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False:
        temp = temp[['TARGET','GHI','DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:,:]
df_train = preprocess_data(train)
x_train = df_train.to_numpy()

###### test파일 합치기############
df_test = []

for i in range(81):
    file_path = '../solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
print(X_test.shape) #(3888, 6)
print(X_test.head(48))

##################################

from sklearn.model_selection import train_test_split
X_train_1, X_val_1, Y_train_1, Y_val_1 = train_test_split(
    df_train.iloc[:,:-2],df_train.iloc[:,-2], test_size=0.2, random_state=0)
X_train_2, X_val_2, Y_train_2, Y_val_2 = train_test_split(
    df_train.iloc[:,:-2],df_train.iloc[:,-1], test_size=0.2, random_state=0)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

from lightgbm import LGBMRegressor

def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):

  
    # (a) Modeling  
    model = LGBMRegressor(alpha=q,num_leaves=140, bagging_fraction=0.8,
                         learning_rate= 0.001,n_estimators=20000, subsample=0.8) 

     
    # LGBMRegressor 회귀모델              
    # bagging_fraction, subsample = 데이터를 랜덤 샘플링하여 학습에 사용 
    # alpha = q -> 제공되는 값 넣는 것. 
    # n_estimators = 트리갯수                
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=100)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model

def train_data(X_train, Y_train, X_val, Y_val, X_test):
    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()
    for q in quantiles:
        print(q)
        pred, model = LGBM(q, X_train, Y_train, X_val, Y_val, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred], axis=1)
    LGBM_actual_pred.columns=quantiles
    return LGBM_models, LGBM_actual_pred

####DAY7####
models_1, results_1 = train_data(X_train_1, Y_train_1, X_val_1, Y_val_1, X_test)
# print(results_1.sort_index()[:48])

####DAY8####
models_2, results_2 = train_data(X_train_2, Y_train_2, X_val_2, Y_val_2, X_test)
# print(results_2.sort_index()[:48])

print(results_1.shape,results_2.shape) #(3888, 9) (3888, 9)

##### sub 파일에 q0.1~0.9까지 값 넣기 ######
sub.loc[sub.id.str.contains('Day7'), 'q_0.1':] = results_1.sort_index().values
sub.loc[sub.id.str.contains('Day8'), 'q_0.1':] = results_2.sort_index().values
# print(sub.iloc[:48])

sub.to_csv('./solar/csv/lgbm_sub_3.csv', index=False)


# 점수 : 1.9507407941 