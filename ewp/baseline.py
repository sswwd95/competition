'''
NeuralProphet 설치
> git clone https://github.com/ourownstory/neural_prophet.git
> cd neural_prophet
> pip install .
'''

import pandas as pd
import numpy as np
from datetime import datetime
from neuralprophet import NeuralProphet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def convert_time(x):
    Ymd, HMS = x.split(' ')
    H, M, S = HMS.split(':')
    H = str(int(H)-1)
    HMS = ':'.join([H, M, S])
    return ' '.join([Ymd, HMS])


# 데이터 불러오기
train_data = pd.read_csv('A:\\study\\data\\ewp\\energy.csv')
# 시간 변환
train_data['time'] = train_data['time'].apply(lambda x:convert_time(x))

energy_list = ['dangjin_floating','dangjin_warehouse','dangjin','ulsan']
submission = pd.read_csv('A:\\study\\data\\ewp\\sample_submission.csv')

for name in energy_list : 
    print(name)    
    column = name
    df = pd.DataFrame()
    df['ds'] = train_data['time']
    df['y'] = train_data[column]

    # 모델 설정
    model = NeuralProphet()
    # 훈련
    loss = model.fit(df, freq="H")
    # 예측용 데이터 프레임 만들기
    df_pred = model.make_future_dataframe(df, periods=18000)
    # 예측
    predict = model.predict(df_pred)
    print(predict)

    # 2021-02-01 ~ 2021-03-01
    predict_1 = predict.copy()
    predict_1 = predict_1.query('ds >= "2021-02-01 00:00:00"')
    predict_1 = predict_1.query('ds < "2021-03-01 00:00:00"')

    # 2021-06-09 ~ 2021-07-09
    predict_2 = predict.copy()
    predict_2 = predict_2.query('ds >= "2021-06-09 00:00:00"')
    predict_2 = predict_2.query('ds < "2021-07-09 00:00:00"')

    # 제출 파일 업데이트
    submission[column] = list(predict_1['yhat1']) + list(predict_2['yhat1'])

    print(submission.head())

    submission.to_csv('A:\\study\\data\\ewp\\submission_0410.csv', index=False)

# 계정 hyurstar(집)
# 파일명 : submission_0410.csv
# score : 13.71979