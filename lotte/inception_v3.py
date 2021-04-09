import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop,Adam,SGD
from sklearn.model_selection import train_test_split


x = np.load("../data/lotte/npy/P_project_x.npy",allow_pickle=True)
y = np.load("../data/lotte/npy/P_project_y.npy",allow_pickle=True)
x_pred = np.load('../data/lotte/npy/test.npy',allow_pickle=True)

print(x.shape, y.shape, x_pred.shape)
# (48000, 150, 150, 3) (48000, 1000) (72000, 150, 150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

local_weights_file = '../data/lotte/model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
     layer.trainable = False
        
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(1000, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience= 15)
lr = ReduceLROnPlateau(patience= 7, factor=0.6)
mc = ModelCheckpoint('../data/lotte/mc/lotte_v3_3.h5',save_best_only=True, verbose=1)

model.compile(optimizer = RMSprop(lr=1e-6), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

history=model.fit(x_train,y_train,callbacks=[es,lr],epochs=500,validation_split=0.2)

loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)

result = model.predict(x_pred, verbose=True)

import pandas as pd
submission = pd.read_csv('../lotte/sample.csv')
submission['prediction'] = result.argmax(1)
submission.to_csv('../data/lotte/answer_v3_3.csv', index=False)

# v3_2파일
# lr=1e-5
# score = 34.250

# v3_3 파일
# lr=1e-6
# score = 32