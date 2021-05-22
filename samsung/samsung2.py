import numpy as np
import pandas as pd

data = np.load('./samsung/npy/samsung2_data.npy')

def split_x(seq, size, col) : 
    dataset = []
    for i in range(len(seq) - size +1) : 
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    return np.array(dataset)
size = 5 # 며칠씩 자를건가
col = 6 # 열

dataset = split_x(data, size, col)
print(dataset)
print(dataset.shape) #(2395, 5, 6)

x = dataset[:-1,:,:7]
print(x.shape) #(2394, 5, 6)
print(x)
y = dataset[1:,:1,-1:]
print(y.shape) #(2394, 1, 1)
print(y)
x_pred = dataset[-1:,:,:]
print(x_pred.shape) #(1, 5, 6)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
x_val =  x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
x_val = scaler.transform(x_val)

print(x_train.shape) #(1915, 25)
print(x_test.shape) # (479, 25)
print(x_pred.shape) #(1,25)
print(x_val.shape)

x_train = x_train.reshape(x_train.shape[0],5,6)
x_test = x_test.reshape(x_test.shape[0], 5,6)
x_pred = x_pred.reshape(x_pred.shape[0],5,6)
x_val =  x_val.reshape(x_val.shape[0], 5,6)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

np.save('./samsung/npy/samsung2.npy',arr=[x_train, x_test, x_val, y_train, y_test,y_val,x_pred])

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(1000,2,activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(800,2,activation='relu'))
model.add(Conv1D(500,2,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../data/modelcheckpoint/samsung2_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'loss', patience=20, mode='min')
model.fit(x_train, y_train, batch_size = 16, callbacks=[es, cp], epochs=1000, validation_data=(x_val,y_val))

# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=16)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict): 
    return np.array(mean_squared_error(y_test, y_predict))
print ('RMSE : ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)

pred = model.predict(x_pred)
print('1월 15일 : ', pred)


# loss, mae :  133903.0625 269.1882629394531
# RMSE :  133903.05
# R2 :  0.999177756043082
# 1월 15일 :  [[89990.15]]

# loss, mae :  156557.625 292.8653259277344
# RMSE :  156557.64
# R2 :  0.9990386435381821
# 1월 15일 :  [[88948.586]]

#1월 15일 실제 주가 : 88,000원