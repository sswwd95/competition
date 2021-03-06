import pandas as pd
import numpy as np
import os
import glob
import random
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

train = pd.read_csv('./solar/csv/train.csv')
# Hour - 시간
# Minute - 분
# DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))
# DNI - 직달일사량(Direct Normal Irradiance (W/m2))
# WS - 풍속(Wind Speed (m/s))
# RH - 상대습도(Relative Humidity (%))
# T - 기온(Temperature (Degree C))
# Target - 태양광 발전량 (kW)

sub = pd.read_csv('./solar/csv/sample_submission.csv')

def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:, :]   #테스트 파일의 day6부분만 자르기

######### test파일 하나로 합치기 #####
df_test = []
for i in range(81):
      file_path = '../solar/test/' + str(i) + '.csv'
      temp = pd.read_csv(file_path)
      temp = preprocess_data(temp)  #day6만 모아서 붙이기
      df_test.append(temp)

X_test = pd.concat(df_test)
#Attach padding dummy time series
X_test = X_test.append(X_test[-96:])  # 왜??? 왜 5,6일을 추가할까??
print(X_test.shape) #(3984, 9)

# Td, T-Td, GHI 피처 추가
# def Add_features(data):
#      c = 243.12
#      b = 17.62
#      gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
#      dp = ( c * gamma) / (b - gamma)
#      data.insert(1,'Td',dp)
#      data.insert(1,'T-Td',data['T']-data['Td'])
#      data.insert(1,'GHI',data['DNI']+data['DHI'])
#      return data

# 직관적인 GHI 피처 추가 (?)
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

train = Add_features(train)
X_test = Add_features(X_test)

print(train.columns)
print(X_test.shape) #(3984, 10)

df_train = train.drop(['Day','Hour','Minute'],axis=1)  
df_test  = X_test.drop(['Day','Hour','Minute'],axis=1)

print(df_train.columns)

# indices : 인덱스들의 복수 묶음, enumerate : 반복문 사용 시 몇 번째 반복문인지 확인
column_indices = {name: i for i, name in enumerate(df_train.columns)}

print(column_indices) # {'GHI': 0, 'DHI': 1, 'DNI': 2, 'WS': 3, 'RH': 4, 'T': 5, 'TARGET': 6}

#Train and Validation split
n = len(train)
train_df = df_train[0:int(n*0.8)]  # ==train_size = 0.8
val_df   = df_train[int(n*0.8):]   # ==val_size = 0.2
test_df = df_test  # test파일

print(train_df.shape) #(42048, 7)
print(val_df.shape) #(10512, 7)

# Normalization(정규화)
num_features = train_df.shape[1]

train_mean = train_df.mean() # mean() = 평균계산
train_std  = train_df.std()  # std() = 표준 편차 계산

train_df = (train_df - train_mean) / train_std
val_df   =  (val_df - train_mean) / train_std
test_df  = (test_df - train_mean) / train_std

print(train_df.shape) #(42048, 7)
print(val_df.shape) #(10512, 7)


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=None):
        # 행 데이터 저장
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # label column indices 작성
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        #Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self) :
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
    # Slicing doesn't preserve static shape information, so set the shapes
    # Manually. This way the tf.data.Datasets' are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels
    
WindowGenerator.split_window = split_window

def make_dataset(self, data,is_train=True):
    data = np.array(data, dtype=np.float32)
    if is_train==True:
    	ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=256,)
    else:
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=False, batch_size=256,)
    ds = ds.map(self.split_window)
    return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_df,is_train=True)

@property
def val(self):
    return self.make_dataset(self.val_df,is_train=True)

@property
def test(self):
    return self.make_dataset(self.test_df,is_train=False)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self,'_example', None)
    if result is None:
        #No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

def plot(self, model=None, plot_col='TARGET', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12,8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
            label='Inputs', marker='.',zorder=-10)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else : 
            label_col_index = plot_col_index
        if label_col_index is None:
            continue 
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
            edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions',  c='#ff7f0e', s=64)
    if n == 0:
        plt.legend()
    plt.xlabel('Time[30m]')
WindowGenerator.plot = plot
#Set the data-set 24 hours input -> 48 hours output
w1 = WindowGenerator(input_width=48, label_width=96, shift = 96)
w1.plot()
# print(plt.show())
print(w1.train.element_spec)
# (TensorSpec(shape=(None, 48, 7), dtype=tf.float32, name=None), TensorSpec(shape=(None, 96, 7), dtype=tf.float32, name=None))

for example_inputs, example_labels in w1.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')  # Inputs shape (batch, time, features): (256, 48, 7)
    print(f'Labels shape (batch, time, features): {example_labels.shape}')  # Labels shape (batch, time, features): (256, 96, 7)

################# Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#########################
OUT_STEPS = 96

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

########## quantile plot definition
def quantile_plot(self, model=None, plot_col='TARGET', max_subplots=3, quantile=None):
    inputs, labels = self.example
    if quantile == 0.1:
        plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        if quantile == 0.1:
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.',zorder=-10)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        if label_col_index is None:
            continue
        if quantile == 0.1:
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=20)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='none', label=f'Predictions(q={quantile})', s=15)
        if quantile == 0.9 and n==0:
            plt.legend()
    plt.xlabel('Time [30m]')

WindowGenerator.quantile_plot = quantile_plot


def DenseModel():
    model = tf.keras.Sequential()
    model.add(L.Lambda(lambda x: x[:, -1:, :]))
    model.add(L.Dense(512, activation='relu'))
    model.add(L.Dense(256, activation='relu'))
    model.add(L.Dense(128, activation='relu'))
    model.add(L.Dense(64, activation='relu'))
    model.add(L.Dense(32, activation='relu'))
    model.add(L.Dense(16, activation='relu'))
    model.add(L.Dense(8, activation='relu'))
    model.add(L.Dense(4, activation='relu'))
    model.add(L.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros))
    model.add(L.Reshape([OUT_STEPS, num_features]))
    return model

Dense_actual_pred = pd.DataFrame()
Dense_val_score = pd.DataFrame()

for q in quantiles:
    model = DenseModel()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    history = model.fit(w1.train, validation_data=w1.val, epochs=300, callbacks=[early_stopping])
    pred = model.predict(w1.test, verbose=0)
    target_pred = pd.Series(pred[::48][:,:,6].reshape(7776)) #Save predicted value (striding=48 step, 9 = TARGET) 
    Dense_actual_pred = pd.concat([Dense_actual_pred,target_pred],axis=1)
    Dense_val_score[f'{q}'] = model.evaluate(w1.val)
    w1.quantile_plot(model, quantile=q)

Dense_actual_pred.columns = quantiles
#Denormalizing TARGET values
Dense_actual_pred_denorm = Dense_actual_pred*train_std['TARGET'] + train_mean['TARGET']
#Replace Negative value to Zero
Dense_actual_pred_nn = np.where(Dense_actual_pred_denorm<0, 0, Dense_actual_pred_denorm)

sub.iloc[:,1:] = Dense_actual_pred_nn
sub.to_csv('.solar/csv/sub_0122_dense.csv',index=False)


'''
### AutoRegressive LSTM
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)
    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(w1.example[0])
prediction.shape

def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the lstm state
    prediction, state = self.warmup(inputs)
    # Insert the first prediction
    predictions.append(prediction)
    # Run the rest of the prediction steps
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                              training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)
    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions

FeedBack.call = call

AR_Lstm_actual_pred = pd.DataFrame()
AR_Lstm_val_score = pd.DataFrame()

for q in quantiles:
    model = feedback_model
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    history = model.fit(w1.train, validation_data=w1.val, epochs=20, callbacks=[early_stopping])
    pred = model.predict(w1.test, verbose=0)
    target_pred = pd.Series(pred[::48][:,:,9].reshape(7776))
    AR_Lstm_actual_pred = pd.concat([AR_Lstm_actual_pred,target_pred],axis=1)
    AR_Lstm_val_score[f'{q}'] = model.evaluate(w1.val)
    w1.quantile_plot(model, quantile=q)

AR_Lstm_actual_pred.columns = quantiles

AR_Lstm_actual_pred_denorm = AR_Lstm_actual_pred*train_std['TARGET'] + train_mean['TARGET']
AR_Lstm_actual_pred_nn = np.where(AR_Lstm_actual_pred_denorm<0, 0, AR_Lstm_actual_pred_denorm)

sub.iloc[:,1:] = AR_Lstm_actual_pred_nn
sub.to_csv("../data/DACON_0126/submission/submission_210118_quantile_ar_lstm_nn.csv",index=False)
'''