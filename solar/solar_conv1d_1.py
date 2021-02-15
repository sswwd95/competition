import pandas as pd
import numpy as np
import os
import glob
import random
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./solar/csv/train.csv')
sub = pd.read_csv('./solar/csv/sample_submission.csv')

# Hour - 시간
# Minute - 분
# DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))
# DNI - 직달일사량(Direct Normal Irradiance (W/m2))
# WS - 풍속(Wind Speed (m/s))
# RH - 상대습도(Relative Humidity (%))
# T - 기온(Temperature (Degree C))
# Target - 태양광 발전량 (kW)

# axis = 0은 행렬에서 행의 원소를 다 더함, 1은 열의 원소를 다 더함

# 1. 데이터

#DHI, DNI 보다 더 직관적인 GHI 열 추가.
def preprocess_data(data, is_train=True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12-6)/6*np.pi/2) 
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # day7
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # day8
        temp = temp.dropna()
        return temp.iloc[:-96] # day8에서 2일치 땡겨서 올라갔기 때문에 마지막 2일 빼주기

    elif is_train==False:
        temp = temp[['Hour','TARGET','GHI','DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:,:] # 트레인데이터가 아니면 마지막 하루만 리턴시킴

df_train = preprocess_data(train)
x_train = df_train.to_numpy()

print(x_train)
print(x_train.shape) #(52464, 10) day7,8일 추가해서 컬럼 10개

###### test파일 합치기############
df_test = []

for i in range(81):
    file_path = '../solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) # 위에서 명시한 False => 마지막 하루만 리턴
    df_test.append(temp)   # 마지막 하루 값들만 전부 붙여주기 

x_test = pd.concat(df_test)
print(x_test.shape) #(3888, 8) -> (81, 48,8) 81일, 하루(24*2(30분단위)=48), 8개 컬럼
x_test = x_test.to_numpy()

##################################

# 정규화 (데이터가 0으로 많이 쏠려있어서 standardscaler 사용)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train[:,:-2])  # day7,8일을 빼고 나머지 컬럼들을 train
x_train[:,:-2] = scaler.transform(x_train[:,:-2])
x_test = scaler.transform(x_test)

######## train데이터 분리 ###########
def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]  # x_train 
        tmp_y1 = data[x_end-1:x_end,-2]  # day7 / x_end-1:x_end => i:x_end와 같은 위치로 맞춰주기
        tmp_y2 = data[x_end-1:x_end,-1]  # day8
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x), np.array(y1), np.array(y2))

x, y1, y2 = split_xy(x_train,1) # x_train을 한 행씩 자른다. (30분 단위로 보면서 day7,8의 같은 시간대 예측)
print(x.shape) #(52464, 1, 8)
print(y1.shape) #(52464, 1)
print(y2.shape) #(52464, 1)

########## test 데이터를 train 데이터와 같게 분리 ######
def split_x(data, timestep) : 
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))
x_test = split_x(x_test,1)
######################################################

from sklearn.model_selection import train_test_split
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    x, y1, y2, train_size = 0.8, random_state=0)

print(x_train.shape) #(41971, 1, 8)

def quantile_loss(q, y_true, y_pred):
    e = (y_true - y_pred)  # 원래값에서 예측값 뺀 것
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1) 

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout

def Model():
    model = Sequential()
    model.add(Conv1D(128,2,padding='same',activation='relu', input_shape = (1,8)))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,2,padding='same', activation='relu'))
    model.add(Conv1D(64,2,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
modelpath = '../solar/check/solar0121_{epoch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

bs = 16
epochs = 1

######day7######
x=[]
for q in quantiles:
    model = Model()
    modelpath = '../solar/check/solar_0121_day7_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    model.compile(loss=lambda y_true,y_pred: quantile_loss(q,y_true, y_pred),
                optimizer='adam', metrics = [lambda y, y_pred: quantile_loss(q, y, y_pred)])
    model.fit(x_train,y1_train, batch_size = bs, callbacks=[es, cp, lr], epochs=epochs, validation_data=(x_val, y1_val))
    pred = pd.DataFrame(model.predict(x_test).round(2)) # round는 반올림 (2)는 . 뒤의 자리수 -> ex) 0.xx를 반올림
    x.append(pred)
df_temp1 = pd.concat(x, axis=1)
df_temp1[df_temp1<0] = 0   # 0보다 작으면 0로 한다. 
num_temp1 = df_temp1.to_numpy()
sub.loc[sub.id.str.contains('Day7'), 'q_0.1':] = num_temp1

######day8#######
x = []
for q in quantiles:
    model = Model()
    modelpath = '../solar/check/solar_0121_day8_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    model.compile(loss=lambda y_true,y_pred: quantile_loss(q,y_true, y_pred), 
                optimizer='adam', metrics = [lambda y, y_pred: quantile_loss(q, y, y_pred)])
    model.fit(x_train,y2_train, batch_size = bs, callbacks=[es, cp, lr], epochs=epochs, validation_data=(x_val, y2_val))
    pred = pd.DataFrame(model.predict(x_test).round(2)) # round는 반올림 (2)는 . 뒤의 자리수 -> ex) 0.xx를 반올림
    x.append(pred)
df_temp2 = pd.concat(x, axis=1)
df_temp2[df_temp2<0] = 0  
num_temp2 = df_temp2.to_numpy()
sub.loc[sub.id.str.contains('Day8'), 'q_0.1':] = num_temp2

sub.to_csv('./solar/csv/sub_0121.csv', index=False)








