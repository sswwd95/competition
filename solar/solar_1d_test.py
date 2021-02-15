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

# data = pd.DataFrame(x_test)
# print(data.duplicated()) #중복된 행인지 점검하기(동일한 데이터가 있는지)
# print(data[data.duplicated()]) #중복된 행의 데이터만 표시하기

'''
     Hour  TARGET  GHI  DHI  DNI   WS     RH     T
335    23     0.0  0.0    0    0  3.4  51.96  -8.4
333    22     0.0  0.0    0    0  1.5  40.62  -4.9
289     0     0.0  0.0    0    0  2.7  57.37  -6.7
299     5     0.0  0.0    0    0  2.7  59.46  -7.3
335    23     0.0  0.0    0    0  2.3  63.30  -6.3
293     2     0.0  0.0    0    0  2.7  42.35  15.5
291     1     0.0  0.0    0    0  2.4  52.59  -5.1
331    21     0.0  0.0    0    0  2.8  54.96  -2.4
289     0     0.0  0.0    0    0  1.8  69.92  -1.9
289     0     0.0  0.0    0    0  1.0  64.94  -5.1
291     1     0.0  0.0    0    0  1.0  64.80  -5.2
295     3     0.0  0.0    0    0  0.8  63.26  -5.2
295     3     0.0  0.0    0    0  0.7  73.63  -2.4
299     5     0.0  0.0    0    0  0.4  74.38  -2.3
293     2     0.0  0.0    0    0  2.1  27.85   7.1
295     3     0.0  0.0    0    0  2.0  27.41   7.1
289     0     0.0  0.0    0    0  3.2  40.45   0.4
293     2     0.0  0.0    0    0  3.4  41.48   0.6
329    20     0.0  0.0    0    0  2.5  58.76  -0.1
331    21     0.0  0.0    0    0  2.5  56.99  -0.2
295     3     0.0  0.0    0    0  1.1  75.91  -3.0
333    22     0.0  0.0    0    0  1.9  76.83   0.9
293     2     0.0  0.0    0    0  2.9  51.31 -12.2
333    22     0.0  0.0    0    0  2.7  53.97  -9.7
299     5     0.0  0.0    0    0  1.3  64.51  -0.1
331    21     0.0  0.0    0    0  2.1  38.23  -4.7
299     5     0.0  0.0    0    0  3.0  43.40 -14.4
291     1     0.0  0.0    0    0  2.0  52.73  -6.7
293     2     0.0  0.0    0    0  1.9  51.83  -6.7
297     4     0.0  0.0    0    0  2.2  48.83  -1.8
291     1     0.0  0.0    0    0  1.2  79.31  -0.8
327    19     0.0  0.0    0    0  0.9  80.09  -3.9
329    20     0.0  0.0    0    0  0.9  54.19  -7.8
289     0     0.0  0.0    0    0  3.8  48.38  -0.5
291     1     0.0  0.0    0    0  3.7  47.71  -0.5
293     2     0.0  0.0    0    0  1.4  60.45  13.9
295     3     0.0  0.0    0    0  1.6  32.63  17.3
289     0     0.0  0.0    0    0  2.1  48.55  -5.2
293     2     0.0  0.0    0    0  1.3  56.98   7.0
333    22     0.0  0.0    0    0  2.5  67.00   8.7
293     2     0.0  0.0    0    0  3.4  35.98  -1.5
291     1     0.0  0.0    0    0  3.0  55.47   4.1
297     4     0.0  0.0    0    0  3.1  58.80   4.3
'''
# print(data.duplicated().sum()) #43 중복된 행이 적기 때문에 데이터에 문제 없음. 
# data.drop_duplicates(inplace=True) 중복된 행은 하나만 남기고 제거하기

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
print(x_test.shape) #(2888,1,8)

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
    model.add(Conv1D(256,2,padding='same',activation='relu', input_shape = (1,8)))
    model.add(Dropout(0.2))
    model.add(Conv1D(128,2,padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,2,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
modelpath = '../solar/check/solar0121_{epoch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

bs = 64
epochs = 200

######day7######
x=[]
for q in quantiles:
    model = Model()
    modelpath = '../solar/check/test1_0121_day7_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    model.compile(loss=lambda y_true,y_pred: quantile_loss(q,y_true, y_pred), optimizer='adam')
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
    modelpath = '../solar/check/test1_0121_day8_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    model.compile(loss=lambda y_true,y_pred: quantile_loss(q,y_true, y_pred), optimizer='adam')
    model.fit(x_train,y2_train, batch_size = bs, callbacks=[es, cp, lr], epochs=epochs, validation_data=(x_val, y2_val))
    pred = pd.DataFrame(model.predict(x_test).round(2)) # round는 반올림 (2)는 . 뒤의 자리수 -> ex) 0.xx를 반올림
    x.append(pred)
df_temp2 = pd.concat(x, axis=1)
df_temp2[df_temp2<0] = 0  
num_temp2 = df_temp2.to_numpy()
sub.loc[sub.id.str.contains('Day8'), 'q_0.1':] = num_temp2

sub.to_csv('./solar/csv/test1_0121.csv', index=False)

# epoch = 200
# 7시 45분 시작- 8시 끝









