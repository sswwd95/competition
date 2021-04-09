
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

x = np.load("../data/lotte/npy/lt_x225.npy",allow_pickle=True)
y = np.load("../data/lotte/npy/lt_y225.npy",allow_pickle=True)
x_pred = np.load('../data/lotte/npy/test225.npy',allow_pickle=True)

x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

idg = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=45, 
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

print(x.shape, y.shape, x_pred.shape)
# print(y[0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.9, shuffle = True, random_state=42)

train_generator = idg.flow(x_train, y_train,batch_size=32, seed = 42)
# seed => random_state
test_generator = idg2.flow(x_test, y_test)
# pred_generator = x_pred
print(x_train.shape, y_train.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet, EfficientNetB4,EfficientNetB7


ef = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(250,250, 3))
top_model = ef.output
top_model = Flatten()(top_model)
top_model = Dense(1000, activation="softmax")(top_model)

model = Model(inputs=ef.input, outputs = top_model)

model.summary()

import time
import datetime
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(patience= 10)
lr = ReduceLROnPlateau(patience= 5, factor=0.5)
mc = ModelCheckpoint('../data/lotte/mc/lotte_225_adam.h5',save_best_only=True, verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss = 'categorical_crossentropy', metrics=['accuracy'])

learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 32,
    validation_data=test_generator, callbacks=[es,lr,mc])

model.save('../data/lotte/h5/b4_225_weight_model.h5')

# predict
model.load_weights('../data/lotte/h5/b4_225_weight_model.h5')
result = model.predict(x_pred,verbose=True)
    
print(result.shape)
sub = pd.read_csv('../lotte/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../data/lotte/answer_225adam.csv',index=False)

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print('실행 시간  : ', times)

# b4 파일(adam lr=1e-4)
# Epoch 47/200
# Epoch 00047: val_loss did not improve from 0.00564
# 1800/1800 [==============================] - ETA: 0s - loss: 0.0016 - accuracy: 0.9997   
# score 73


# b4_sgd 파일(sgd lr=1e-2)
# Epoch 00043: val_loss did not improve from 0.00800
# 675/675 [==============================] - 217s 321ms/step - loss: 1.3907e-04 - accuracy: 1.0000 - val_loss: 0.0080 - val_accuracy: 0.9979
# score = 74

# b4_sgd_2파일(sgd lr=0.05)
# Epoch 00067: val_loss did not improve from 0.00187
# 675/675 [==============================] - 215s 319ms/step - loss: 1.4300e-04 - accuracy: 1.0000 - val_loss: 0.0027 - val_accuracy: 0.9994
# score = 71

# b4_sgd_2파일(sgd lr=0.05) img 제너레이터 수정
