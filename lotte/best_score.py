# 전이황 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 

tf.config.experimental.set_memory_growth(physical_devices[0], True)
#데이터 지정 및 전처리
# x = np.load('../../data/npy/train_data_x9.npy',allow_pickle=True)
# x_pred = np.load('../../data/npy/predict_data9.npy',allow_pickle=True)
# y = np.load("../../data/npy/P_project_y5.npy",allow_pickle=True)
# y1 = np.zeros((len(y), len(y.unique())))
# for i, digit in enumerate(y):
#     y1[i, digit] = 1

x = np.load("../data/lotte/npy/P_project_x200.npy",allow_pickle=True)
y = np.load("../data/lotte/npy/P_project_y200.npy",allow_pickle=True)
x_pred = np.load('../data/lotte/npy/test200.npy',allow_pickle=True)


x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=45, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=32, seed = 42)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
# test_generator = idg2.flow(x_pred, shuffle=False)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers

efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
a = efficientnet.output
a = Conv2D(filters = 1792,kernel_size=(12,12), strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-5)) (a)
a = BatchNormalization() (a)
a = Activation('swish') (a)
a = GlobalAveragePooling2D() (a)
# a = Flatten() (a)
a = Dense(512, activation= 'swish') (a)
a = Dropout(0.5) (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnet.input, outputs = a)

efficientnet.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 10)
lr = ReduceLROnPlateau(patience= 5, factor=0.5)
mc = ModelCheckpoint('../data/lotte/mc/lotte_b4.h5',save_best_only=True, verbose=1)

model.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['acc'])
learning_history = model.fit_generator (train_generator,epochs=300, steps_per_epoch= len(x_train) / 32,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

model.save('../data/lotte/h5/b4_weight_model.h5')

# predict
model.load_weights('../data/lotte/mc/lotte_b4.h5')
result = model.predict(x_pred,verbose=True)

# tta_steps = 30
# # tta : 증강된 이미지를 여러번 보여준 다음 각각의 단계에 대해서 prediction을 평균하고 이 결과를 최종값으로 사용하는 것
# predictions = []

# for i in tqdm(range(tta_steps)):
#    # generator 초기화
#     test_generator.reset()
#     preds = model.predict_generator(generator = test_generator, verbose = 1)
#     predictions.append(preds)

# 평균을 통한 final prediction
# pred = np.mean(predictions, axis=0)

sub = pd.read_csv('../lotte/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../data/lotte/answer_b4.csv',index=False)

# size 200x200
# Adam(learning_rate=0.0005)
# 85.172