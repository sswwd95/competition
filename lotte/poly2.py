import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
'''
test_img = load_img('C:/lotte_data/LPD_competition/train/14/7.jpg')
plt.imshow(test_img)
plt.show()
# 어레이로 바꿔 크기 확인
arr_img = img_to_array(test_img)
print(arr_img.shape)
# (256, 256, 3)
arr_img = arr_img.reshape(1, arr_img.shape[0]*arr_img.shape[1]*arr_img.shape[2])
# 폴리노미날피텨 적용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_img = poly.fit_transform(arr_img)
print(poly_img.shape)
plt.imshow(poly_img)
plt.show()
'''
# --------------------------------------

test_img = cv2.imread('C:/lotte_data/LPD_competition/train/32/2.jpg')#, cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (100, 100))/255.

# 원본
arr_img = img_to_array(test_img)
print(arr_img.shape)
# 32,32

# 원본에서 특성 강한 것
arr_img3 = np.where((arr_img <= 160/255.), 0, arr_img)
arr_img2 = arr_img.reshape(100, 100*3)
arr_img4 = arr_img3.reshape(100, 100*3)



arr_img = arr_img.reshape(100, 100*3)


print(arr_img.shape)

print('2차원이다\n: ', arr_img)

# 폴리노미날피텨 적용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_img = poly.fit_transform(arr_img)
poly_img2 = poly.fit_transform(arr_img4)


print(poly_img.shape)
# (32, 4752)
# plt.imshow(poly_img)
# plt.show()

# (32, 32, 3)
# # 2 차원 만들기
# (32, 96)
# # 폴리
# (32, 4752)


print('폴리이다\n: ', poly_img)

poly_img = poly_img.reshape(100*5, int(45450/15), 3)
poly_img2 = poly_img2.reshape(100*5, int(45450/15), 3)

# 특성 센거만 남아라 ㅡㅡ 딱남아
# 원본 -> 폴리
poly_img = np.where((poly_img <= 160/255.), 0, poly_img)

# 원본 ->  특성 -> 폴리


# # 이미지 팽창
# poly_img = cv2.dilate(poly_img, kernel=np.ones((2, 2), np.uint8), iterations=1)
# poly_img2 = cv2.dilate(poly_img2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
# poly_img = cv2.medianBlur(src=poly_img, ksize= 5)
# poly_img2 = cv2.medianBlur(src=poly_img2, ksize= 5)


plt.figure(figsize=(15,15))

# 원본
plt.subplot(4,1,1)
plt.imshow(test_img)

# 원본 -> 특성
plt.subplot(4,1,2)
plt.imshow(arr_img3)

# 원본 -> 폴리
plt.subplot(4,1,3)
plt.imshow(poly_img)

# 원본 -> 특성 -> 폴리
plt.subplot(4,1,4)
plt.imshow(poly_img2)

plt.show()

# 질문: 그럼 용량이 너무너무 커지는데 어떡해요?
# 샘왈: 그것만 쓰면 안되고 다른 거랑 엮어야지!
# pca인가 ?

# 결론: 원본 > 특성 강하게 > 폴리돌려 > 쉐잎 바꿔서 모델에 넣어 > acc 비교