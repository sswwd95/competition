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
test[TARGET] = pseudo_labels[TARGET]
all_df = pd.concat([train, test]).reset_index(drop=True)

target = train.pop('Survived')
# print(test.head())

# Feature engeenring==============================
# Age fillna with mean age for each class
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

# Cabin, fillna with 'X' and take first letter
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())

# Ticket, fillna with 'X', split string and take first split 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
all_df['Fare'] = np.log1p(all_df['Fare'])

# Embarked, fillna with 'X' value
all_df['Embarked'] = all_df['Embarked'].fillna('X')

# Name, take only surnames
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])
# print(all_df.head(5))   #[5 rows x 12 columns]
# print(all_df.isnull().sum())  #결측값 제거 완료

# 컬럼별로 다르게 적용
label_cols = ['Name', 'Ticket', 'Sex']
onehot_cols = ['Cabin', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)

scaler = StandardScaler()

onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = pd.DataFrame(scaler.fit_transform(all_df[numerical_cols]), columns=numerical_cols)
target_df = all_df[TARGET]

all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df, target_df], axis=1)
# print(all_df.head(5))   #[5 rows x 22 columns]

all_df_scaled = all_df.drop([TARGET], axis = 1).copy()
scaler = StandardScaler()
scaler.fit(all_df.drop([TARGET], axis = 1))
all_df_scaled = scaler.transform(all_df_scaled)

all_df_scaled = pd.DataFrame(all_df_scaled, columns=all_df.drop([TARGET], axis = 1).columns)
# print(all_df_scaled.head(5))  #[5 rows x 21 columns]

X = all_df_scaled
y = all_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = RANDOM_SEED)

test = all_df_scaled[len(train):]

# print (f'X:{X.shape} y: {y.shape} \n')  #X:(200000, 21) y: (200000,) 
# print (f'X_train:{X_train.shape} y_train: {y_train.shape}')
# print (f'X_test:{X_test.shape} y_test: {y_test.shape}')
print (f'test:{test.shape}')
# X:(200000, 21) y: (200000,) 
# X_train:(160000, 21) y_train: (160000,)
# X_test:(40000, 21) y_test: (40000,)
# test:(100000, 21)  


# Single SVM run ==================================== 
svc_kernel_rbf = SVC(kernel='rbf', random_state=0, C=1.3040348958661234, gamma=0.11195797734572176, verbose=True)
svc_kernel_rbf.fit(X_train, y_train)
y_pred = svc_kernel_rbf.predict(X_test)
accuracy_score(y_pred, y_test)

final_pred = svc_kernel_rbf.predict(test)

submission['Survived'] = np.round(final_pred).astype(int)
submission.to_csv('A:\\study\\data\\titanic\\svc_kernel_rbf.csv')



# Hyperparams selection and Kfolds==================
def objective(trial):
    from sklearn.svm import SVC
    params = {
        'C': trial.suggest_loguniform('C', 0.01, 0.1),
        'gamma': trial.suggest_categorical('gamma', ["auto"]),
        'kernel': trial.suggest_categorical("kernel", ["rbf"])
    }

    svc = SVC(**params, verbose=True)
    svc.fit(X_train, y_train)
    return svc.score(X_test, y_test)

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=123),
                            direction="maximize",
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=5, show_progress_bar=True)

print(f"Best Value from optune: {study.best_trial.value}")
print(f"Best Params from optune: {study.best_params}")

if study.best_trial.value >= 0.8851:
    best_value = study.best_params
else:
    best_value = {'C': 0.9284115572652722, 'gamma': 0.1234156796521313, 'kernel': 'rbf'}
    print(f"Using precalculated best params instead: {best_value}")

n_folds = 20
kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
y_pred = np.zeros(test.shape[0])

for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
    print("Running Fold {}".format(fold + 1))
    X_train, X_valid = pd.DataFrame(X.iloc[train_index]), pd.DataFrame(X.iloc[valid_index])
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    svc_kernel_rbf = SVC(**best_value)
    svc_kernel_rbf.fit(X_train, y_train)
    print("  Accuracy: {}".format(accuracy_score(y_valid, svc_kernel_rbf.predict(X_valid))))
    y_pred += svc_kernel_rbf.predict(test)

y_pred /= n_folds

print("")
print("Done!")

submission['Survived'] = np.round(y_pred).astype(int)
submission.to_csv('A:\\study\\titanic\\sub\\voting_submission3.csv')

# 04/30 143등
# 0.81714


# 5/1 최종 16등
# 0.81309

