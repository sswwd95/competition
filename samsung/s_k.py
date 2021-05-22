import numpy as np

x_train=np.load('./samsung/npy/sam.npy',allow_pickle=True)[0]
x_test=np.load('./samsung/npy/sam.npy',allow_pickle=True)[1]
x_val=np.load('./samsung/npy/sam.npy',allow_pickle=True)[2]
y_train=np.load('./samsung/npy/sam.npy',allow_pickle=True)[3]
y_test=np.load('./samsung/npy/sam.npy',allow_pickle=True)[4]
y_val=np.load('./samsung/npy/sam.npy',allow_pickle=True)[5]
x_pred=np.load('./samsung/npy/sam.npy',allow_pickle=True)[6]

print(x_train.shape)

x1_train=np.load('./samsung/npy/ko.npy',allow_pickle=True)[0]
x1_test=np.load('./samsung/npy/ko.npy',allow_pickle=True)[1]
x1_val=np.load('./samsung/npy/ko.npy',allow_pickle=True)[2]
x1_pred=np.load('./samsung/npy/ko.npy',allow_pickle=True)[3]


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, concatenate,Dropout,Input

#samsung
input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))
dense1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)

#kodex
input2 = Input(shape=(x_train.shape[1],x_train.shape[2]))
dense2 = LSTM(64, activation='relu')(input2)
dense2 = Dense(32, activation='relu')(dense2)
dense2 = Dense(32, activation='relu')(dense2)
dense2 = Dense(16, activation='relu')(dense2)

# 합치기
merge1 = concatenate([dense1, dense2])
middle1 = Dense(20, activation='relu')(merge1)     
middle1 = Dense(16, activation='relu')(middle1)
middle1 = Dense(8, activation='relu')(middle1)
middle1 = Dense(8, activation='relu')(middle1)
output1 = Dense(2, activation='relu')(middle1)

model = Model(inputs=[input1, input2],
              outputs = output1)

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../data/modelcheckpoint/s_k_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'val_loss', patience=20, mode='min')
model.fit([x_train,x1_train], y_train, batch_size = 16, callbacks=[es, cp], epochs=500, validation_data=([x_val,x1_val],y_val))

# 4. 평가 예측
loss,mae = model.evaluate([x_test,x1_test], y_test, batch_size=16)
print("loss, mae : ", loss, mae)

y_predict = model.predict([x_test,x1_test])


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict): 
    return np.array(mean_squared_error(y_test, y_predict))
print ('RMSE : ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)

predict = model.predict([x_pred,x1_pred])
print('1월 18일(시가), 1월 19일(시가): ', predict)

