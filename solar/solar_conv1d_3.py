
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
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']+data['DHI'])

    temp = data.copy()
    temp = temp[['TARGET','GHI','DHI', 'DNI','WS','RH', 'T']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # day7
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # day8
        temp = temp.dropna()
        return temp.iloc[:-96] # day8에서 2일치 땡겨서 올라갔기 때문에 마지막 2일 빼주기

    elif is_train==False:
        temp = temp[['TARGET','GHI','DHI', 'DNI','WS', 'RH', 'T']]
        return temp.iloc[-48:,:] # 트레인데이터가 아니면 마지막 하루(day6)만 리턴시킴

df_train = preprocess_data(train)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.9) # 폰트크기 0.9
sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True)
# sns.heatmap(data=df.corr(), square=정사각형으로, annot=글씨 , cbar=오른쪽에 있는 bar)
# plt.show()

print(df_train.columns)

train = df_train.to_numpy()

print(train)
print(train.shape) #(52464,9)

###### test파일 합치기############
df_test = []

for i in range(81):
    file_path = '../solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) # 위에서 명시한 False => 마지막 하루만 리턴
    df_test.append(temp)   # 마지막 하루 값들만 전부 붙여주기 

target1 = pd.concat(df_test)
print(target1.shape) #(3888, 7) -> (81, 48,8) 81일, 하루(24*2(30분단위)=48), 8개 컬럼
target = target1.to_numpy()

# data = pd.DataFrame(x_test)
# print(data.duplicated()) #중복된 행인지 점검하기(동일한 데이터가 있는지)
# print(data[data.duplicated()]) #중복된 행의 데이터만 표시하기

# print(data.duplicated().sum()) #43 중복된 행이 적기 때문에 데이터에 문제 없음. 
# data.drop_duplicates(inplace=True) 중복된 행은 하나만 남기고 제거하기

######## train데이터 분리 ###########
def split_xy(data,timestep):
    x,y = [],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]  
        tmp_y = data[x_end-1:x_end,-2:]  
        x.append(tmp_x)
        y.append(tmp_y)
    return(np.array(x), np.array(y))

x, y = split_xy(train,48)  # 하루씩 잘라서 예측하기
print(x.shape)
print(y.shape) 
# (52417, 48, 7)
# (52417, 1, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=0)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, random_state=0)

print(x_train.shape) 
print(x_test.shape) 
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

# (33546, 48, 7)
# (10484, 48, 7)
# (8387, 48, 7)
# (33546, 48, 2)
# (8387, 48, 2)

# 정규화하기 위해 2차원만들기
x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
target = scaler.transform(target)

x_train = x_train.reshape(int(x_train.shape[0]/48),48,x_train.shape[1])
x_test = x_test.reshape(int(x_test.shape[0]/48),48,x_test.shape[1])
x_val = x_val.reshape(int(x_val.shape[0]/48),48,x_val.shape[1])
target = target.reshape(int(target.shape[0]/48),48,target.shape[1])

print(x_train.shape)
print(x_val.shape)
print(x_test.shape) 
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
    model.add(Conv1D(256, 2, activation='relu',padding='same',input_shape=( x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 2, activation='relu', padding='same'))
    model.add(Conv1D(128, 2, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(96,))
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
epochs = 1

loss_list = list()
############
for q in quantiles:
    model = Model()
    modelpath = '../solar/check/_conv2d_3_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    model.compile(loss=lambda y_true,y_pred: quantile_loss(q,y_true, y_pred), optimizer='adam',metrics=['mse'])
    model.fit(x_train,y_train, batch_size = bs, callbacks=[es, cp, lr], epochs=epochs, validation_data=(x_val, y_val))
   
    result = model.evaluate(x_test, y_test, batch_size=32)
    # print('loss : ', result[0])
    loss_list.append(result[0])
    y_pred = model.predict(target)
    print(y_pred.shape) #(81,48,2)
    

    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2]))
    y_pred1 = pd.concat([y_pred], axis=1)
    y_pred1[y_pred<0] = 0
    y_pred2 = y_pred1.to_numpy()
        
    print(str(q)+'번째 지정')
    sub.loc[sub.id.str.contains('Day7'), 'q_' + str(q)] = y_pred2[:,0].round(2)
    sub.loc[sub.id.str.contains('Day8'), 'q_' + str(q)] = y_pred2[:,1].round(2)

sub.to_csv('./solar/csv/sub_conv1d_3.csv',index=False)

loss_mean = sum(loss_list) / len(loss_list)
print('loss_mean : ', loss_mean)

#epoch=1 :-> loss_mean :  2.4424050649007163
# 노드 200->100 : loss_mean :  2.4398615757624307
#loss_mean :  2.410707950592041

# loss_mean :  2.272169404559665

# loss_mean :  2.207289112938775 hour 추가
