import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
#########데이터 로드

train_dir =  'A:\\study\\data\\lotte\\train'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)
print(categories)

nb_classes = len(categories)
print(nb_classes)

image_w = 225
image_h = 225

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    print(label)
    image_dir = train_dir + "/" + cat
    print(image_dir)
    print(cat)
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("../data/lotte/npy/lt_x225.npy", arr=X)
np.save("../data/lotte/npy/lt_y225.npy", arr=y)
x = np.load("../data/lotte/npy/lt_x225.npy",allow_pickle=True)
y = np.load("../data/lotte/npy/lt_y225.npy",allow_pickle=True)

print(x.shape)
print(y.shape)
# (48000, 255, 255, 3)
# (48000, 1000)


img1=[]
for i in range(0,72000):
    filepath='../lotte/test/%d.jpg'%i
    image2=Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((image_w, image_h))
    image_data2=asarray(image2)
    img1.append(image_data2)    

np.save("../data/lotte/npy/test225.npy", arr=img1)

x_pred = np.load('../data/lotte/npy/test225.npy',allow_pickle=True)

print(x_pred.shape)



