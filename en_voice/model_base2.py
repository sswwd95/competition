from pickle import load
from librosa.filters import mel
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import librosa
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

sample_submission = pd.read_csv("A:\study\en_voice\sample_submission.csv")

africa_train_paths = glob("A:\study\en_voice/train/africa/*.wav")
australia_train_paths = glob("A:\study\en_voice/train/australia/*.wav")
canada_train_paths = glob("A:\study\en_voice/train/canada/*.wav")
england_train_paths = glob("A:\study\en_voice/train/england/*.wav")
hongkong_train_paths = glob("A:\study\en_voice/train/hongkong/*.wav")
us_train_paths = glob("A:\study\en_voice/train/us/*.wav")

path_list = [africa_train_paths, australia_train_paths, canada_train_paths,
             england_train_paths, hongkong_train_paths, us_train_paths]


# glob로 test data의 path를 불러올때 순서대로 로드되지 않을 경우를 주의해야 합니다.
# test_ 데이터 프레임을 만들어서 나중에 sample_submission과 id를 기준으로 merge시킬 준비를 합니다.

def get_id(data):
    return np.int(data.split("\\")[4].split(".")[0])
test = 'A:\\study\\en_voice\\test\\*.wav'
print(test.split("\\")[4].split(".")[0])

test_ = pd.DataFrame(index = range(0, 6100), columns = ["path", "id"])
test_["path"] = glob("A:\\study\\en_voice\\test\\*.wav")
print('path : ',test_["path"])
test_["id"] = test_["path"].apply(lambda x : get_id(x))
print('id : ',test_["id"])

start = datetime.now()

# npy파일로 저장된 데이터를 불러옵니다.
africa_train_data = np.load("A:\study\en_voice/npy_data/africa18_npy.npy", allow_pickle = True)
australia_train_data = np.load("A:\study\en_voice/npy_data/australia18_npy.npy", allow_pickle = True)
canada_train_data = np.load("A:\study\en_voice/npy_data/australia18_npy.npy", allow_pickle = True)
england_train_data = np.load("A:\study\en_voice/npy_data/england18_npy.npy", allow_pickle = True)
hongkong_train_data = np.load("A:\study\en_voice/npy_data/hongkong18_npy.npy", allow_pickle = True)
us_train_data = np.load("A:\study\en_voice/npy_data/us18_npy.npy", allow_pickle = True)

test_data = np.load("A:\study\en_voice/npy_data/test18_npy.npy", allow_pickle = True)

train_data_list = [africa_train_data, australia_train_data, canada_train_data, england_train_data, hongkong_train_data, us_train_data]

# 이번 대회에서 음성은 각각 다른 길이를 갖고 있습니다.
# baseline 코드에서는 음성 중 길이가 가장 작은 길이의 데이터를 기준으로 데이터를 잘라서 사용합니다.
# for i in range(6) : 
#     print(len(train_data_list[i]))
# 2500
# 1000 
# 1000 
# 10000
# 1020
# 10000  

def get_mini(data):

    mini = 9999999
    for i in data:
        if len(i) < mini:
            mini = len(i)

    return mini

#음성들의 길이를 맞춰줍니다.

def set_length(data, d_mini):

    result = []
    for i in data:
        result.append(i[:d_mini])
    result = np.array(result)

    return result

#feature를 생성합니다.

def get_feature(data, sr = 18000, n_fft = 512, hop_length = 128, n_mels = 128):
    mel = []
    for i in data:
        # win_length 는 음성을 작은 조각으로 자를때 작은 조각의 크기입니다.
        # hop_length 는 음성을 작은 조각으로 자를때 자르는 간격을 의미합니다.
        # n_mels 는 적용할 mel filter의 개수입니다.
        mel_ = librosa.feature.melspectrogram(i, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)
        mel.append(mel_)
    mel = np.array(mel)
    mel = librosa.power_to_db(mel, ref = np.max)
    print(np.max(mel), np.min(mel))
    # 0.0 -80.0

    print("================================")
    mel_mean = mel.mean()
    print(mel_mean)
    # -69.2475

    print("================================")
    mel_std = mel.std()
    print(mel_std)
    # 14.347334

    print("================================")
    mel = (mel - mel_mean) / mel_std
    print(mel)
    print("================================")

    return mel

train_x = np.concatenate(train_data_list, axis= 0)
test_x = np.array(test_data)

# 음성의 길이 중 가장 작은 길이를 구합니다.

train_mini = get_mini(train_x)
test_mini = get_mini(test_x)

mini = np.min([train_mini, test_mini])


# data의 길이를 가장 작은 길이에 맞춰 잘라줍니다.

train_x = set_length(train_x, mini)
test_x = set_length(test_x, mini)

# librosa를 이용해 feature를 추출합니다.

train_x = get_feature(data = train_x)
test_x = get_feature(data = test_x)

train_x = train_x.reshape(-1, train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(-1, test_x.shape[1], test_x.shape[2], 1)
print(train_x.shape)
print(test_x.shape)
# (25520, 128, 704, 1)
# (6100, 128, 704, 1)

# train_data의 label을 생성해 줍니다.


train_y = np.concatenate((np.zeros(len(africa_train_data), dtype = np.int),
                        np.ones(len(australia_train_data), dtype = np.int),
                         np.ones(len(canada_train_data), dtype = np.int) * 2,
                         np.ones(len(england_train_data), dtype = np.int) * 3,
                         np.ones(len(hongkong_train_data), dtype = np.int) * 4,
                         np.ones(len(us_train_data), dtype = np.int) * 5), axis = 0)


print(train_x.shape, train_y.shape, test_x.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Convolution2D, BatchNormalization, Flatten,
                                     Dropout, Dense, AveragePooling2D, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
def block(input_, units= 8, dropout_rate=0.2):
    
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(input_)
    x = Dropout(rate=dropout_rate)(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = Dropout(rate=dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = Dropout(rate=dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)
    return x


def build_fn():
    dropout_rate = 0.2
    
    in_ = Input(shape = (train_x.shape[1:]))
    
    block_01 = block(in_, units = 8, dropout_rate = dropout_rate)
    block_02 = block(block_01, units = 16, dropout_rate = dropout_rate)
    block_03 = block(block_02, units = 32, dropout_rate = dropout_rate)

    x = Flatten()(block_03)

    x = Dense(units = 64, activation = "relu")(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Dropout(rate = dropout_rate)(x)

    x = Dense(units = 128, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Add()([x_res, x])
    x = Dropout(rate = dropout_rate)(x)

    model_out = Dense(units = 6, activation = 'softmax')(x)
    model = Model(in_, model_out)
    return model

split = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 10)
es = EarlyStopping(patience= 10, monitor= 'val_loss', verbose= 1)
lr = ReduceLROnPlateau(patience=5, monitor= 'val_loss', factor=0.5, verbose=1)
path = 'A:\\study\\en_voice\\h5\\en_voice_base2.h5'
mc = ModelCheckpoint(path, monitor='val_loss', save_best_only=True)

pred = []
pred_ = []
for train_idx, val_idx in split.split(train_x, train_y):
    x_train, y_train = train_x[train_idx], train_y[train_idx]
    x_val, y_val = train_x[val_idx], train_y[val_idx]

    model = build_fn()
    model.summary()

    model.compile(optimizer = keras.optimizers.Adam(0.001),
                 loss = keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ['acc'])

    history = model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), epochs = 100, callbacks=[es,lr,mc])
    print("*******************************************************************")
    pred.append(model.predict(test_x))
    pred_.append(np.argmax(model.predict(test_x), axis = 1))
    print("*******************************************************************")

def cov_type(data):
    return np.int(data)

# 처음에 살펴본 것처럼 glob로 test data의 path는 sample_submission의 id와 같이 1,2,3,4,5.....으로 정렬 되어있지 않습니다.
# 만들어둔 test_ 데이터프레임을 이용하여 sample_submission과 predict값의 id를 맞춰줍니다.

result = pd.concat([test_, pd.DataFrame(np.mean(pred, axis = 0))], axis = 1).iloc[:, 1:]
print(result)
result["id"] = result["id"].apply(lambda x : cov_type(x))

result = pd.merge(sample_submission["id"], result)
result.columns = sample_submission.columns

result.to_csv('A:/study/en_voice/csv/baseline2.csv',index=False)



end = datetime.now()
time = end-start
print('시간 : ', time)
'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 128, 704, 1) 0
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 704, 16) 160         input_1[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 128, 704, 16) 64          conv2d[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 704, 16) 2320        batch_normalization[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 128, 704, 16) 64          conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 704, 16) 2320        batch_normalization_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 128, 704, 16) 64          conv2d_2[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 704, 16) 0           batch_normalization_2[0][0]
                                                                 batch_normalization[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 64, 352, 16)  0           add[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 64, 352, 16)  0           average_pooling2d[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 352, 32)  4640        dropout[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 352, 32)  128         conv2d_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 352, 32)  9248        batch_normalization_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 64, 352, 32)  128         conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 352, 32)  9248        batch_normalization_4[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 64, 352, 32)  128         conv2d_5[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 352, 32)  0           batch_normalization_5[0][0]
                                                                 batch_normalization_3[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 32, 176, 32)  0           add_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 32, 176, 32)  0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 176, 64)  18496       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 176, 64)  256         conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 176, 64)  36928       batch_normalization_6[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 176, 64)  256         conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 176, 64)  36928       batch_normalization_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 176, 64)  256         conv2d_8[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 176, 64)  0           batch_normalization_8[0][0]
                                                                 batch_normalization_6[0][0]
__________________________________________________________________________________________________
average_pooling2d_2 (AveragePoo (None, 16, 88, 64)   0           add_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 16, 88, 64)   0           average_pooling2d_2[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 90112)        0           dropout_2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          11534464    flatten[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 128)          512         dense[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 128)          0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          16512       dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 128)          512         dense_1[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 128)          0           batch_normalization_9[0][0]
                                                                 batch_normalization_10[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 128)          0           add_3[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 6)            774         dropout_4[0][0]
==================================================================================================
Total params: 11,674,406
Trainable params: 11,673,222
Non-trainable params: 1,184
__________________________________________________________________________________________________
'''