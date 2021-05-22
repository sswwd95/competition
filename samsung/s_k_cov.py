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
from tensorflow.keras.layers import Dense,Conv1D,concatenate,Dropout,Flatten,Input

#samsung
input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))
c1 = Conv1D(128,2,padding='same', activation='relu')(input1)
c1 = Conv1D(128,2,padding='same', activation='relu')(c1)
c1 = Conv1D(32,2,padding='same', activation='relu')(c1)
c1 = Flatten()(c1)
c1 = Dense(50, activation='relu')(c1)

#kodex
input2 = Input(shape=(x_train.shape[1],x_train.shape[2]))
c2 = Conv1D(128,2,padding='same', activation='relu')(input1)
c2 = Conv1D(128,2,padding='same', activation='relu')(c2)
c2 = Conv1D(32,2,padding='same', activation='relu')(c2)
c2 = Flatten()(c2)
c2 = Dense(50, activation='relu')(c2)

# 합치기
merge1 = concatenate([c1, c2])
middle1 = Dense(50, activation='relu')(merge1)     
# middle1 = Dense(8, activation='relu')(middle1)
# middle1 = Dense(8, activation='relu')(middle1)
# middle1 = Dense(8, activation='relu')(middle1)
output1 = Dense(2, activation='relu')(middle1)

model = Model(inputs=[input1, input2],
              outputs = output1)

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/s_k_cov_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'val_loss', patience=20, mode='min')

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit([x_train,x1_train], y_train, batch_size = 16, callbacks=[es, cp], epochs=1000, validation_data=([x_val,x1_val],y_val))

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

# s_k_cov_420021.2812.hdf5
# loss, mae :  582009.25 517.2100219726562
# RMSE :  582009.25
# R2 :  0.9907168382421194
# 1월 18일(시가), 1월 19일(시가):  [[89458.805 89562.97 ]]

# s_k_cov_347614.5938.hdf5
# loss, mae :  491052.5625 494.7571105957031
# RMSE :  491052.7
# R2 :  0.9921571536105421
# 1월 18일(시가), 1월 19일(시가):  [[88931.414 88908.35 ]]

# s_k_cov_374025.5000.hdf5
# loss, mae :  530065.625 505.1334533691406
# RMSE :  530065.25
# R2 :  0.9915368752439169
# 1월 18일(시가), 1월 19일(시가):  [[88949.5  89530.48]]


# lr 하기 전 
# s_k_cov_388846.9688.hdf5
# loss, mae :  544370.75 523.5440673828125
# RMSE :  544371.0
# R2 :  0.9912770307606085
# 1월 18일(시가), 1월 19일(시가):  [[89452.52 89700.46]]


