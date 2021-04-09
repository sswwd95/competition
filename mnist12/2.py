import numpy as np
import pandas as pd
import glob
import datetime
import cv2

import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dense, BatchNormalization, Activation, Reshape, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
    
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

from PIL import Image

# c:/data/dacon/data2/dirty_mnist/
# c:/data/dacon/data2/test_dirty_mnist
# c:/data/dacon/data2/dirty_mnist_answer.csv


kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=23
)

# 이미지 로드 / npy 저장

str_time=datetime.datetime.now()

# train_list=glob.glob('c:/data/dacon/data2/dirty_mnist/*.png')
# test_list=glob.glob('c:/data/dacon/data2/test_dirty_mnist/*.png')
# answer_csv=pd.read_csv('c:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0, header=0)

df_train=pd.read_csv('../dacon12/mnist_data/train.csv', index_col=0, header=0)
dftrain=df_train.iloc[:, 2:].values # mnist1 x
dfanswer=df_train.iloc[:, 1].values # mnist1 y
df_test=pd.read_csv('../dacon12/mnist_data/test.csv', index_col=0, header=0)
dftest=df_test.iloc[:, 1:].values
df_answer=df_test.iloc[:, 0].values

answer=list()
for i in range(len(dfanswer)):
    label=dfanswer[i]
    answer.append(label)

for i in range(len(df_answer)):
    label=df_answer[i]
    answer.append(label)

answer=np.array(answer)

print(answer[:10])
print(answer.shape)


dftrain=dftrain.reshape(-1, 28, 28)
dftest=dftest.reshape(-1, 28, 28)

count=2048
for i in range(20480):
    img=dftest[i]
    img=np.where((img<65)&(img!=0), 0, img)
    img=np.where((img>65)&(img!=0), 255, img)
    img=Image.fromarray(img.astype('uint8'))
    img.save('../dacon12/mnist_data/' + str(count) + '.png')
    count+=1

dftrain=list()
for i in range(22528):
    img=cv2.imread('../dacon12/mnist_data/')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, dsize=None, fx=9, fy=9, interpolation=cv2.INTER_CUBIC)
    img=np.array(img)
    dftrain.append(img)

print(dftrain[0].shape)
print(dfanswer[0])

dftrain=np.array(dftrain)

# plt.imshow(dftrain[0])
# plt.show()
# img=Image.fromarray(img.astype('uint8'), 'RGB')
# img.save('c:/data/dacon/data2/mnist_data/train.png')

'''
# digit OneHotEncoding
label=list()
for i in range(len(answer)):
    if answer[i]=='A':
        digit=0
        label.append(digit)
    elif answer[i]=='B':
        digit=1
        label.append(digit)
    elif answer[i]=='C':
        digit=2
        label.append(digit)
    elif answer[i]=='D':
        digit=3
        label.append(digit)
    elif answer[i]=='E':
        digit=4
        label.append(digit)
    elif answer[i]=='F':
        digit=5
        label.append(digit)
    elif answer[i]=='G':
        digit=6
        label.append(digit)
    elif answer[i]=='H':
        digit=7
        label.append(digit)
    elif answer[i]=='I':
        digit=8
        label.append(digit)
    elif answer[i]=='J':
        digit=9
        label.append(digit)
    elif answer[i]=='K':
        digit=10
        label.append(digit)
    elif answer[i]=='L':
        digit=11
        label.append(digit)
    elif answer[i]=='M':
        digit=12
        label.append(digit)
    elif answer[i]=='N':
        digit=13
        label.append(digit)
    elif answer[i]=='O':
        digit=14
        label.append(digit)
    elif answer[i]=='P':
        digit=15
        label.append(digit)
    elif answer[i]=='Q':
        digit=16
        label.append(digit)
    elif answer[i]=='R':
        digit=17
        label.append(digit)
    elif answer[i]=='S':
        digit=18
        label.append(digit)
    elif answer[i]=='T':
        digit=19
        label.append(digit)
    elif answer[i]=='U':
        digit=20
        label.append(digit)
    elif answer[i]=='V':
        digit=21
        label.append(digit)
    elif answer[i]=='W':
        digit=22
        label.append(digit)
    elif answer[i]=='X':
        digit=23
        label.append(digit)
    elif answer[i]=='Y':
        digit=24
        label.append(digit)
    elif answer[i]=='Z':
        digit=25
        label.append(digit)
    else:
        pass

label=to_categorical(label)

print(dftrain[:5])
print(label[:5])

# print(len(train_list)) # 50000
# print(len(test_list)) # 5000
# print(train_list_numpy.shape) # (2048, 28, 28)
# print(train_list_answer[0]) # L
# print(train_list_answer.shape) # (2048, )

# img2=cv2.imread(train_list_numpy[0], cv2.IMREAD_GRAYSCALE)
# img2=cv2.resize(img2, (128, 128))

x=dftrain
y=label

x=x.reshape(-1, 252, 252, 1)/255.

print('x : ', x.shape)
print('y : ', y.shape)

# print(y.shape)

# x_train, x_val, y_train, y_val=train_test_split(
#     x, y,
#     train_size=0.8,
#     random_state=23
# )

# x_train, x_test, y_train, y_test=train_test_split(
#     x_train, y_train,
#     train_size=0.9,
#     random_state=23
# )

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=0.1
)

batch=16

datagen2=ImageDataGenerator()

# train_set=datagen.flow(
#     x_train, y_train,
#     batch_size=batch,
#     seed=23
# )

# val_set=datagen2.flow(
#     x_val, y_val,
#     seed=23
# )

# test_set=datagen2.flow(
#     x_test, y_test,
#     seed=23
# )

# print(x_test[:5])
# print(y_test)

# print(x_train.shape) # (1474, 28, 28, 1)
# print(x_val.shape) # (410, 28, 28, 1)
# print(x_test.shape) # (164, 28, 28, 1)
# print(y_train.shape) # (1474, )

pred=list()
pred=0
kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=23
)

i=1

for train_index, val_index in kf.split(x, y):
    x_train=x[train_index]
    y_train=y[train_index]
    x_val=x[val_index]
    y_val=y[val_index]

    x_train, x_test, y_train, y_test=train_test_split(
        x_train, y_train,
        train_size=0.9
    )

    train_set=datagen.flow(
        x_train, y_train,
        seed=23,
        batch_size=batch
    )

    val_set=datagen2.flow(
        x_val, y_val,
        seed=23
    )

    test_set=datagen2.flow(
        x_test, y_test,
        seed=23
    )

    model=Sequential()
    model.add(Conv2D(128, 2, padding='same', input_shape=(252, 252, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 2, padding='same'))
    model.add(Conv2D(32, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(26, activation='softmax'))

    es=EarlyStopping(
        patience=200,
        verbose=1
    )

    mc=ModelCheckpoint(
        'c:/data/modelcheckpoint/dacon2.hdf5',
        verbose=1,
        save_best_only=True
    )

    rl=ReduceLROnPlateau(
        verbose=1,
        factor=0.5,
        patience=50
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics='acc'
    )

    model.fit(
        train_set,
        validation_data=(x_val, y_val),
        steps_per_epoch=len(x_train)//batch,
        epochs=500,
        callbacks=[es, mc, rl]
    )

    # model.fit_generator(
    #     train_set,
    #     validation_data=val_set,
    #     steps_per_epoch=len(x_train)//batch,
    #     epochs=500,
    #     callbacks=[es, mc, rl]
    # )

    # loss=model.evaluate(
    #     test_set
    # )

    model.load_weights('c:/data/modelcheckpoint/dacon2.hdf5')

    pred=model.predict(
        test_set
    )

    pred=np.argmax(pred, axis=-1)


    results=list()

    for i in range(len(pred)):
        if pred[i]==0:
            img='A'
            results.append(img)
        elif pred[i]==1:
            img='B'
            results.append(img)
        elif pred[i]==2:
            img='C'
            results.append(img)
        elif pred[i]==3:
            img='D'
            results.append(img)
        elif pred[i]==4:
            img='E'
            results.append(img)
        elif pred[i]==5:
            img='F'
            results.append(img)
        elif pred[i]==6:
            img='G'
            results.append(img)
        elif pred[i]==7:
            img='H'
            results.append(img)
        elif pred[i]==8:
            img='I'
            results.append(img)
        elif pred[i]==9:
            img='J'
            results.append(img)
        elif pred[i]==10:
            img='K'
            results.append(img)
        elif pred[i]==11:
            img='L'
            results.append(img)
        elif pred[i]==12:
            img='M'
            results.append(img)
        elif pred[i]==13:
            img='N'
            results.append(img)
        elif pred[i]==14:
            img='O'
            results.append(img)
        elif pred[i]==15:
            img='P'
            results.append(img)
        elif pred[i]==16:
            img='Q'
            results.append(img)
        elif pred[i]==17:
            img='R'
            results.append(img)
        elif pred[i]==18:
            img='S'
            results.append(img)
        elif pred[i]==19:
            img='T'
            results.append(img)
        elif pred[i]==20:
            img='U'
            results.append(img)
        elif pred[i]==21:
            img='V'
            results.append(img)
        elif pred[i]==22:
            img='W'
            results.append(img)
        elif pred[i]==23:
            img='X'
            results.append(img)
        elif pred[i]==24:
            img='Y'
            results.append(img)
        elif pred[i]==25:
            img='Z'
            results.append(img)

    print('results : ', results[:10])
    '''