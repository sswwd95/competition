# td,rh 제거
# 점수 : 2.0774143574

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

# 1. 데이터

# def Add_features(data):
#      c = 243.12
#      b = 17.62
#      gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
#      dp = ( c * gamma) / (b - gamma)
#      data.insert(1,'Td',dp)
#      data.insert(1,'T-Td',data['T']-data['Td'])
#      data.insert(1,'GHI',data['DNI']+data['DHI'])
#      return data

def preprocess_data(data, is_train=True):
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']+data['DHI'])

    temp = data.copy()
    temp = temp[['TARGET','GHI','DHI', 'DNI', 'WS', 'T','T-Td']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # day7
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # day8
        temp = temp.dropna()
        return temp.iloc[:-96] # day8에서 2일치 땡겨서 올라갔기 때문에 마지막 2일 빼주기

    elif is_train==False:
        temp = temp[['TARGET','GHI','DHI', 'DNI', 'WS', 'T','T-Td']]
        return temp.iloc[-48:,:] # 트레인데이터가 아니면 마지막 하루(day6)만 리턴시킴

df_train = preprocess_data(train)


print(df_train.columns)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.9) # 폰트크기 0.9
sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True)
# sns.heatmap(data=df.corr(), square=정사각형으로, annot=글씨 , cbar=오른쪽에 있는 bar)
# plt.show()

x_train = df_train.to_numpy()

print(x_train)
print(x_train.shape) #(52464, 9) day7,8일 추가해서 컬럼 10개

###### test파일 합치기############
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

# print(data.duplicated().sum()) #43 중복된 행이 적기 때문에 데이터에 문제 없음. 
# data.drop_duplicates(inplace=True) 중복된 행은 하나만 남기고 제거하기

##################################

# 정규화 (데이터가 0으로 많이 쏠려있어서 standardscaler 사용)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train[:,:-2])  # day7,8일을 빼고 나머지 컬럼들을 train
x_train[:,:-2] = scaler.transform(x_train[:,:-2])
x_test = scaler.transform(x_test)

x_test = x_test.reshape(81,48,7)

######## train데이터 분리 ###########
def split_xy(data,timestep):
    x,y = [],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]  # x_train 
        tmp_y = data[i:x_end,-2:]  # day7 / x_end-1:x_end => i:x_end와 같은 위치로 맞춰주기
        x.append(tmp_x)
        y.append(tmp_y)
    return(np.array(x), np.array(y))

x, y = split_xy(x_train,48)  # 하루씩 잘라서 예측하기
print(x.shape) #(52417, 48, 8)
print(y.shape) # (52417, 48, 2)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size = 0.8, random_state=0)

print(x_train.shape) 
print(x_test.shape) 
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

# (41933, 48, 8)
# (3841, 48, 8)
# (10484, 48, 8)
# (41933, 1, 2)
# (10484, 1, 2)

x_train = x_train.reshape(x_train.shape[0],1,48,x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], 1,48,x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], 1,48,x_val.shape[2])

y_train = y_train.reshape(y_train.shape[0],48,2 )
y_val = y_val.reshape(y_val.shape[0],48,2 )

print(x_train.shape)
print(x_val.shape)
print(x_test.shape) #(81, 1, 48, 8)
print(y_train.shape)
print(y_val.shape)

def quantile_loss(q, y_true, y_pred):
    e = (y_true - y_pred)  # 원래값에서 예측값 뺀 것
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1) 

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Flatten, Dropout, Reshape

# relu 넣으면 값 더 떨어짐
def Model():
    model = Sequential()
    model.add(Conv2D(128, 2, padding='same',input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(Conv2D(64, 2, padding='same'))
    # model.add(Conv2D(64, 2, padding='same'))
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Reshape((48,2)))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(2))
    return model
    

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

bs = 64
epochs = 200


############
for q in quantiles:
    model = Model()
    modelpath = '../solar/check/_conv2d_3_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    model.compile(loss=lambda y_true,y_pred: quantile_loss(q,y_true, y_pred), optimizer='adam')
    model.fit(x_train,y_train, batch_size = bs, callbacks=[es, cp, lr], epochs=epochs, validation_data=(x_val, y_val))
    
    target = model.predict(x_test)
    
    target = pd.DataFrame(target.reshape(target.shape[0]*target.shape[1],target.shape[2]))
    target1 = pd.concat([target], axis=1)
    target1[target<0] = 0
    target2 = target1.to_numpy()
        
    print(str(q)+'번째 지정')
    sub.loc[sub.id.str.contains('Day7'), 'q_' + str(q)] = target2[:,0].round(2)
    sub.loc[sub.id.str.contains('Day8'), 'q_' + str(q)] = target2[:,1].round(2)

sub.to_csv('./solar/csv/sub_conv2d_3.csv',index=False)
