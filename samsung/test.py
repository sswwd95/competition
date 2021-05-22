import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,Activation
import  datetime

data=pd.read_csv('005930.KS_5y.csv')
data.head()

#결측치 제거
dataset=data.dropna()

high_prices=data['High'].values
low_prices=data['Low'].values
mid_prices=(high_prices+low_prices)/2

mid_prices

plt.plot(mid_prices)

#create window
seq_len=50 #최근 50일 data를 사용(window size=50)
#한칸씩 밀어서 window를 생성
sequence_length=seq_len+1

result=[]
for index in range(len(mid_prices)-sequence_length):
    result.append(mid_prices[index:index+sequence_length])
  
  
normalized_data = []
window_mean=[]
window_std=[]
for window in result:
    normalized_window = [((p-np.mean(window))/np.std(window)) for p in window]
    normalized_data.append(normalized_window)
    window_mean.append(np.mean(window))
    window_std.append(np.std(window))

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
#50일(x)로 1일(y)예측
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=50)
#validation loss가 작을수록 학습이 잘된 것

pred = model.predict(x_test)

#복원
pred_result=[]
pred_y=[]
for i in range(len(pred)):
      n1=(pred[i]*window_std[i])+window_mean[i]
      n2=(y_test[i]*window_std[i])+window_mean[i]
      pred_result.append(n1)
      pred_y.append(n2)
     
fig = plt.figure(facecolor='white',figsize=(20,10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()