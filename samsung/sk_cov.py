import numpy as np
from tensorflow.keras.models import load_model

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

# 모델
model=load_model('../data/modelcheckpoint/s_k_cov_347614.5938.hdf5') 
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

# loss, mae :  491052.5625 494.7571105957031
# RMSE :  491052.7
# R2 :  0.9921571536105421
# 1월 18일(시가), 1월 19일(시가):  [[88931.414 88908.35 ]]