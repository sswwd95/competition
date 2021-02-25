import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
# stratifiedkfold = k-fold가 label을 데이터와 학습에 올바르게 분배하지 못하는 경우를 해결해준다.  
from keras import Sequential
from keras.layers import *
# math 모듈의 모든 변수, 함수, 클래스 가져온다.
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD

train = pd.read_csv('../dacon7/train.csv')
test = pd.read_csv('../dacon7/test.csv')
sub = pd.read_csv('../dacon7/submission.csv')

#distribution of label('digit') 
train['digit'].value_counts()

# drop columns
train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values

plt.imshow(train2[100].reshape(28,28))

# reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# data normalization
train2 = train2/255.0
test2 = test2/255.0

# imagedatagenerator로 data 늘리기,부풀리기
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
'''
idg = ImageDataGenerator(rotation_range=회전하는 범위(단위: degree),
                            width_shift_range=수평 이동하는 범위(이미지 가로폭에 대한 비율),
                            height_shift_range=수직 이동하는 범위(이미지의 세로폭에 대한 비율),
                            shear_range=전단(shearing)범위. 크게 하면 더 비스듬하게 찌그러진 이미지가 됨(단위:degree),
                            zoom_range=이미지를 확대/축소시키는 비율(최소:1-zoom_range, 최대:1+zoom_range),
                            channel_shift_range=입력이 RGB3채널인 이미지의 경우 R,G,B각각에 임이의 값을 더하거나 뺄 수 있음(0~255),
                            horizontal_flip=True로 설정 시 가로로 반전,
                            vertical_flip=True로 설정 시 세로로 반전,
                            brightness_range = 이미지의 밝기를 랜덤으로 다르게 준다)

# 문자랑 숫자는 방향성이 있기 때문에 큰 변화 주지 않는다. 
# idg = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
# 표준화는 개별 특징에 대해 평균을 0으로, 분산을 1로 하여 특징별 데이터 분포를 좁히는 방법
# idg = ImageDataGenerator(zca_whitening=True)
# 백색화는 데이터 성분 사이의 상관관계를 없애는 방법.
# 백색화를 수행하면 전체적으로 어두워지고 가장자리가 강조된 것처럼 보이지만 이는 백색화가
# 주위의 픽셀 정보로부터 쉽게 상정되는 색상은 무시하는 효과가 있기 때문. 정보량이 많은 가장자리 등을 강조함으로써 학습 효율을 높일 수 있다.

'''
idg2 = ImageDataGenerator()
'''
# show augmented image data 
sample_data = train2[100].copy()
sample = expand_dims(sample_data,0)
sample_datagen = ImageDataGenerator(rotation_range=5,height_shift_range=(-1,1), width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) : 
    plt.subplot(3,3,i+1)
    sample_batch = sample_generator.next()
    sample_image=sample_batch[0]
    plt.imshow(sample_image.reshape(28,28))
'''
# Validation

# cross validation
skf = StratifiedKFold(n_splits=80, random_state=42, shuffle=True)

# Modeling
# %%time

reLR = ReduceLROnPlateau(patience=100,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=160, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(train2,train['digit']) :
    
    mc = ModelCheckpoint('../dacon7/check/best_cvision.h5',save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train,batch_size=16,seed=7)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(test2,shuffle=False)
    
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    # 배치 정규화 :  미니배치 학습을 통해 배치마다 표준화를 수행하는 것
    # 활성화 함수 relu 등 출력값의 범위가 한정되지 않은 함수의 출력에 배치 정규화를 사용하면 학습이 원활하게 진행되어 큰 효과 발휘
    # 올바르게 정규화하면 활성화 함수에 sigmoid가 아닌 relu함수를 사용해도 좋은 학습 결과 나온다.

    model.add(Dropout(0.2))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    #sparse_categorical_crossentropy : 다중분류 손실 함수. categorical_crossentropy와 동일하지만 원핫인코딩안해도 된다. 
    # epsilon : 0으로 나누어지는 것을 방지

    learning_history = model.fit_generator(train_generator,epochs=1000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('../dacon7/check/best_cvision.h5')
    result += model.predict_generator(test_generator,verbose=True)/80
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')

print(val_loss_min, np.mean(val_loss_min))

model.summary()

# Submission

sub['digit'] = result.argmax(1)
sub.to_csv('../dacon7/sub/my_last.csv',index=False)


# [0.0018692457815632224, 0.22330547869205475, 0.22682319581508636, 0.1364862024784088, 0.1243099793791771, 
# .0013704715529456735, 0.11754651367664337, 0.11301590502262115, 0.20533983409404755, 0.04607833921909332, 
# 0.23159566521644592, 0.06503696739673615, 0.07219216972589493, 0.0027585176285356283, 0.0035736015997827053, 
# 0.04797627404332161, 0.26872676610946655, 0.0016134700272232294, 8.60580439621117e-06, 0.18866358697414398, 
# 0.0006911106174811721, 0.01920921728014946, 0.07500430941581726, 0.016810225322842598, 0.05403481051325798, 
# 0.0015547852963209152, 0.1442171037197113, 0.177638441324234, 0.22273430228233337, 0.053504034876823425,
#  0.10526036471128464, 0.00010383558401372284, 2.86098702417803e-06, 0.18142269551753998, 0.3452763557434082,
#  0.08483167737722397, 0.08852493762969971, 0.07368792593479156, 0.14506791532039642, 0.07617022097110748,
#  0.2802917957305908, 0.24388617277145386, 3.433772508287802e-05, 0.15745480358600616, 0.24472159147262573,
#  0.06629257649183273, 0.0009186813258565962, 0.13506601750850677, 8.456425712211058e-05, 0.07034020870923996,
#  0.08151176571846008, 0.06635554134845734, 0.0012418099213391542, 0.03519897535443306, 0.011998157948255539, 
# 0.000503463321365416, 0.03487606346607208, 0.0008711389382369816, 0.41332998871803284, 9.665128345659468e-06,
#  1.730908707031631e-06, 0.1333596557378769, 0.002676163800060749, 0.023635411635041237, 0.0027228640392422676,
#  0.2863043546676636, 9.069227962754667e-05, 0.1966668665409088, 0.0016968795098364353, 0.04442853108048439,
#  0.11156415194272995, 0.014731531031429768, 0.31445151567459106, 0.34944072365760803, 0.37953007221221924, 
# 0.01418767124414444, 0.00048405781853944063, 0.06805093586444855, 0.1739005297422409, 0.19765827059745789] 
# 0.1013075981261153

'''
# validation 생성
x_train, x_val, y_train, y_val = train_test_split(
    train2, train['digit'], test_size=0.2, random_state=77, stratify=train['digit'])
# stratify : default=none. target으로 지정해주면 각각의 class비율(ratio)을 train/validation에 유지
# 한 쪽에 쏠려서 분배되는 것을 막아주기 때문에 이 옵션을 사용하지 않으면 classification 문제일 경우 성능차이가 많이 난다.
'''