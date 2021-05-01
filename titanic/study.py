# https://www.kaggle.com/napetrov/tps04-svm-with-scikit-learn-intelex
# pip install scikit-learn-intelex
# pip install optuna

from sklearnex import patch_sklearn
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import optuna

RANDOM_SEED = 2021
PROBAS = True
FOLDS = 5
N_ESTIMATORS = 1000

TARGET = 'Survived'

# Loading data===========================
# test 데이터 Pseudo Labels추가
train = pd.read_csv("A:\\study\\data\\titanic\\train.csv")
test = pd.read_csv("A:\\study\\data\\titanic\\test.csv")
submission = pd.read_csv("A:\\study\\data\\titanic\\sample_submission.csv", index_col='PassengerId')
# Pseudo labels taken from great BIZEN notebook: https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble
pseudo_labels = pd.read_csv("A:\\study\\data\\titanic\\pseudo_label.csv")
print(pseudo_labels)

test[TARGET] = pseudo_labels[TARGET]
print(test[TARGET])
'''
0        0
1        1
2        1
3        0
4        1
        ..
99995    1
99996    0
99997    0
99998    1
99999    1
'''
print('==================')
print(train)
'''
       PassengerId  Survived  Pclass                Name     Sex    Age  SibSp  Parch     Ticket   Fare   Cabin Embarked
0                0         1       1    Oconnor, Frankie    male    NaN      2      0     209245  27.14  C12239        S
1                1         0       3         Bryan, Drew    male    NaN      0      0      27323  13.35     NaN        S
2                2         0       3      Owens, Kenneth    male   0.33      1      2  CA 457703  71.29     NaN        S
3                3         0       3       Kramer, James    male  19.00      0      0   A. 10866  13.04     NaN        S
4                4         1       3       Bond, Michael    male  25.00      0      0     427635   7.76     NaN        S
...            ...       ...     ...                 ...     ...    ...    ...    ...        ...    ...     ...      ...
99995        99995         1       2         Bell, Adele  female  62.00      0      0   PC 15008  14.86  D17243        C
99996        99996         0       2       Brown, Herman    male  66.00      0      0      13273  11.15     NaN        S
99997        99997         0       3  Childress, Charles    male  37.00      0      0        NaN   9.95     NaN        S
99998        99998         0       3    Caughlin, Thomas    male  51.00      0      1     458654  30.92     NaN        S
99999        99999         0       3       Enciso, Tyler    male  55.00      0      0     458074  13.96     NaN        S
'''
print('==================')
print(test)
'''
       PassengerId  Pclass                Name     Sex   Age  SibSp  Parch    Ticket    Fare   Cabin Embarked  Survived
0           100000       3    Holliday, Daniel    male  19.0      0      0     24745   63.01     NaN        S         0
1           100001       3    Nguyen, Lorraine  female  53.0      0      0     13264    5.81     NaN        S         1
2           100002       1     Harris, Heather  female  19.0      0      0     25990   38.91  B15315        C         1
3           100003       2        Larsen, Eric    male  25.0      0      0    314011   12.93     NaN        S         0
4           100004       1       Cleary, Sarah  female  17.0      0      2     26203   26.89  B22515        C         1
...            ...     ...                 ...     ...   ...    ...    ...       ...     ...     ...      ...       ...
99995       199995       3       Cash, Cheryle  female  27.0      0      0      7686   10.12     NaN        Q         1
99996       199996       1       Brown, Howard    male  59.0      1      0     13004   68.31     NaN        S         0
99997       199997       3  Lightfoot, Cameron    male  47.0      0      0   4383317   10.87     NaN        S         0
99998       199998       1  Jacobsen, Margaret  female  49.0      1      2  PC 26988   29.68  B20828        C         1
99999       199999       1    Fishback, Joanna  female  41.0      0      2  PC 41824  195.41  E13345        C         1
'''
print('==================')

# concat만 하면 행 인덱스가 달라진다.트레인 10만까지 기록하고, 다시 테스트 0부터 10만까지 기록된다.
all_df = pd.concat([train, test]).reset_index(drop=True)
# reset_index(drop=True)로 하면 기존 인덱스를 버리고 재배열해준다.
print(all_df.isnull().sum())
# PassengerId         0
# Survived            0
# Pclass              0
# Name                0
# Sex                 0
# Age              6779
# SibSp               0
# Parch               0
# Ticket           9804
# Fare              267
# Cabin          138697
# Embarked          527

# train의 survived 열을 pop()으로 요소 삭제해주고, 꺼낸 survived는 target으로 설정한다.
target = train.pop('Survived')
print(target)

# Feature engeenring==============================
# 결측값을 제거 안하고 fillna()이용하여 평균값으로 채워준다
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

# Cabin, fillna with 'X' and take first letter
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())
# 객실 열들의 결측값에는 'X' 로 채우고 값이 있으면 0번째인 알파벳만 불러온다.

# Ticket, fillna with 'X', split string and take first split 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')
# 티켓 열의 결측값에 'X'로 채우고 값이 있는 경우, 공백 기준으로 문자를 나누었을 때 
# 나눈 str형태의 문자의 길이가 1보다 크면 str의 (공백 기준으로) 0번째만 지정하고 아니면 'X'로 바꾼다. 

# Fare, fillna with mean value
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
# 운임승차료와 티겟 클래스의 데이터프레임을 리스트로 만들어주고 dropna()를 통해 결측치를 삭제한다.
# groupby('Pclass')는 'Pclass'를 제외한 컬럼에 함수 적용하게 해준다.
# Fare(운임승차료)의 중앙값을 Pclass에 맞춰서 뽑아내고, dictionary형태로 만들어준다.
# 실행 결과 : {'Fare': {1: 71.81, 2: 21.7, 3: 11.33}}


all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
# 'Fare'의 결측값을 위의 fare_map에서 처리한 Fare의 값으로 채워준다.

all_df['Fare'] = np.log1p(all_df['Fare'])
# np.log가 아닌 log1p를 하는 이유는  price가 0이 되는 경우 y 값이 무한대가 되버리기 때문에 정규화가 힘들어진다.
# 그래서 log1p를 사용하는 것이다. = log(1+ax)

# Embarked, fillna with 'X' value
all_df['Embarked'] = all_df['Embarked'].fillna('X')

# Name, take only surnames
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])
print(all_df[:10])   #[5 rows x 12 columns]
print(all_df.isnull().sum())  #결측값 제거 완료

# # 컬럼별로 다르게 적용
# label_cols = ['Name', 'Ticket', 'Sex']
# onehot_cols = ['Cabin', 'Embarked']
# numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# def label_encoder(c):
#     le = LabelEncoder()
#     return le.fit_transform(c)

# scaler = StandardScaler()

# onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
# label_encoded_df = all_df[label_cols].apply(label_encoder)
# numerical_df = pd.DataFrame(scaler.fit_transform(all_df[numerical_cols]), columns=numerical_cols)
# target_df = all_df[TARGET]

# all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df, target_df], axis=1)
# # print(all_df.head(5))   #[5 rows x 22 columns]

# all_df_scaled = all_df.drop([TARGET], axis = 1).copy()
# scaler = StandardScaler()
# scaler.fit(all_df.drop([TARGET], axis = 1))
# all_df_scaled = scaler.transform(all_df_scaled)

# all_df_scaled = pd.DataFrame(all_df_scaled, columns=all_df.drop([TARGET], axis = 1).columns)
# # print(all_df_scaled.head(5))  #[5 rows x 21 columns]

# X = all_df_scaled
# y = all_df[TARGET]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = RANDOM_SEED)

# test = all_df_scaled[len(train):]

# # print (f'X:{X.shape} y: {y.shape} \n')  #X:(200000, 21) y: (200000,) 
# # print (f'X_train:{X_train.shape} y_train: {y_train.shape}')
# # print (f'X_test:{X_test.shape} y_test: {y_test.shape}')
# print (f'test:{test.shape}')
# # X:(200000, 21) y: (200000,) 
# # X_train:(160000, 21) y_train: (160000,)
# # X_test:(40000, 21) y_test: (40000,)
# # test:(100000, 21)  


# # Single SVM run ==================================== 
# svc_kernel_rbf = SVC(kernel='rbf', random_state=0, C=1.3040348958661234, gamma=0.11195797734572176, verbose=True)
# svc_kernel_rbf.fit(X_train, y_train)
# y_pred = svc_kernel_rbf.predict(X_test)
# accuracy_score(y_pred, y_test)

# final_pred = svc_kernel_rbf.predict(test)

# submission['Survived'] = np.round(final_pred).astype(int)
# submission.to_csv('A:\\study\\data\\titanic\\svc_kernel_rbf.csv')



# # Hyperparams selection and Kfolds==================
# def objective(trial):
#     from sklearn.svm import SVC
#     params = {
#         'C': trial.suggest_loguniform('C', 0.01, 0.1),
#         'gamma': trial.suggest_categorical('gamma', ["auto"]),
#         'kernel': trial.suggest_categorical("kernel", ["rbf"])
#     }

#     svc = SVC(**params, verbose=True)
#     svc.fit(X_train, y_train)
#     return svc.score(X_test, y_test)

# study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=123),
#                             direction="maximize",
#                             pruner=optuna.pruners.MedianPruner())
# study.optimize(objective, n_trials=5, show_progress_bar=True)

# print(f"Best Value from optune: {study.best_trial.value}")
# print(f"Best Params from optune: {study.best_params}")

# if study.best_trial.value >= 0.8851:
#     best_value = study.best_params
# else:
#     best_value = {'C': 0.9284115572652722, 'gamma': 0.1234156796521313, 'kernel': 'rbf'}
#     print(f"Using precalculated best params instead: {best_value}")

# n_folds = 20
# kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
# y_pred = np.zeros(test.shape[0])

# for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
#     print("Running Fold {}".format(fold + 1))
#     X_train, X_valid = pd.DataFrame(X.iloc[train_index]), pd.DataFrame(X.iloc[valid_index])
#     y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#     svc_kernel_rbf = SVC(**best_value)
#     svc_kernel_rbf.fit(X_train, y_train)
#     print("  Accuracy: {}".format(accuracy_score(y_valid, svc_kernel_rbf.predict(X_valid))))
#     y_pred += svc_kernel_rbf.predict(test)

# y_pred /= n_folds

# print("")
# print("Done!")

# submission['Survived'] = np.round(y_pred).astype(int)
# submission.to_csv('A:\\study\\titanic\\sub\\voting_submission3.csv')

# # 04/30 143등
# # 0.81714


# # 5/1 최종 16등
# # 0.81309

