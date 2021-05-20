from pickle import load
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

start = datetime.now()

# npy파일로 저장된 데이터를 불러옵니다.
africa_train_data = np.load("A:\study\en_voice/npy_data/africa_npy.npy", allow_pickle = True)
australia_train_data = np.load("A:\study\en_voice/npy_data/australia_npy.npy", allow_pickle = True)
canada_train_data = np.load("A:\study\en_voice/npy_data/australia_npy.npy", allow_pickle = True)
england_train_data = np.load("A:\study\en_voice/npy_data/england_npy.npy", allow_pickle = True)
hongkong_train_data = np.load("A:\study\en_voice/npy_data/hongkong_npy.npy", allow_pickle = True)
us_train_data = np.load("A:\study\en_voice/npy_data/us_npy.npy", allow_pickle = True)

test_data = np.load("A:\study\en_voice/npy_data/test_npy.npy", allow_pickle = True)

train_data_list = [africa_train_data, australia_train_data, canada_train_data, england_train_data, hongkong_train_data, us_train_data]

# 이번 대회에서 음성은 각각 다른 길이를 갖고 있습니다.
# baseline 코드에서는 음성 중 길이가 가장 작은 길이의 데이터를 기준으로 데이터를 잘라서 사용합니다.

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

def get_feature(data, sr = 22050, n_fft = 512, hop_length = 128, n_mels = 128):
    mel = []
    for i in data:
        # win_length 는 음성을 작은 조각으로 자를때 작은 조각의 크기입니다.
        # hop_length 는 음성을 작은 조각으로 자를때 자르는 간격을 의미합니다.
        # n_mels 는 적용할 mel filter의 개수입니다.
        mel_ = librosa.feature.melspectrogram(i, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)
        mel.append(mel_)
    mel = np.array(mel)
    mel = librosa.power_to_db(mel, ref = np.max)

    mel_mean = mel.mean()
    mel_std = mel.std()
    mel = (mel - mel_mean) / mel_std

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
# (25520, 128, 863, 1)
# (6100, 128, 863, 1)

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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Convolution2D, BatchNormalization, Flatten,
                                     Dropout, Dense, AveragePooling2D, Add)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

nmb = load_model('A:\\study\\en_voice\\mobilenet_rmsprop_1.h5')

split = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 10)

pred = []
pred_ = []

for train_idx, val_idx in split.split(train_x, train_y):
    x_train, y_train = train_x[train_idx], train_y[train_idx]
    x_val, y_val = train_x[val_idx], train_y[val_idx]

    model = nmb()
    model.compile(optimizer = keras.optimizers.RMSprop(0.001),
                 loss = keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ['acc'])

    history = model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), epochs = 8)
    print("*******************************************************************")
    pred.append(model.predict(test_x))
    pred_.append(np.argmax(model.predict(test_x), axis = 1))
    print("*******************************************************************")


def cov_type(data):
    return np.int(data)

# 처음에 살펴본 것처럼 glob로 test data의 path는 sample_submission의 id와 같이 1,2,3,4,5.....으로 정렬 되어있지 않습니다.
# 만들어둔 test_ 데이터프레임을 이용하여 sample_submission과 predict값의 id를 맞춰줍니다.

result = pd.concat([test_, pd.DataFrame(np.mean(pred, axis = 0))], axis = 1).iloc[:, 1:]
result["id"] = result["id"].apply(lambda x : cov_type(x))

result = pd.merge(sample_submission["id"], result)
result.columns = sample_submission.columns

result.to_csv("DACON.csv", index = False)

end = datetime.now()

time = end - start
print('시간 : ', time)