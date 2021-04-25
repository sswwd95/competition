# https://www.kaggle.com/hiro5299834/tps-apr-2021-7-models-voting-ensemble

import pandas as pd 
import numpy as np

import lightgbm as lgb
import catboost as ctb

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("A:\\study\\data\\titanic\\train.csv")
test = pd.read_csv("A:\\study\\data\\titanic\\test.csv")
submission = pd.read_csv("A:\\study\\data\\titanic\\sample_submission.csv")

TARGET = 'Survived'

print(train.head())
'''
   PassengerId  Survived  Pclass              Name   Sex  ...  Parch     Ticket   Fare   Cabin  Embarked
0            0         1       1  Oconnor, Frankie  male  ...      0     209245  27.14  C12239         S       
1            1         0       3       Bryan, Drew  male  ...      0      27323  13.35     NaN         S       
2            2         0       3    Owens, Kenneth  male  ...      2  CA 457703  71.29     NaN         S       
3            3         0       3     Kramer, James  male  ...      0   A. 10866  13.04     NaN         S       
4            4         1       3     Bond, Michael  male  ...      0     427635   7.76     NaN         S 

'''
print(train.describe())
'''
[5 rows x 12 columns]
         PassengerId       Survived         Pclass           Age          SibSp          Parch         Fare
count  100000.000000  100000.000000  100000.000000  96708.000000  100000.000000  100000.000000  99866.00000    
mean    49999.500000       0.427740       2.106910     38.355472       0.397690       0.454560     43.92933    
std     28867.657797       0.494753       0.837727     18.313556       0.862566       0.950076     69.58882    
min         0.000000       0.000000       1.000000      0.080000       0.000000       0.000000      0.68000    
25%     24999.750000       0.000000       1.000000     25.000000       0.000000       0.000000     10.04000    
50%     49999.500000       0.000000       2.000000     39.000000       0.000000       0.000000     24.46000    
75%     74999.250000       1.000000       3.000000     53.000000       1.000000       1.000000     33.50000    
max     99999.000000       1.000000       3.000000     87.000000       8.000000       9.000000    744.66000    
'''
print(train.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 12 columns):
 #   Column       Non-Null Count   Dtype
---  ------       --------------   -----
 0   PassengerId  100000 non-null  int64
 1   Survived     100000 non-null  int64
 2   Pclass       100000 non-null  int64
 3   Name         100000 non-null  object
 4   Sex          100000 non-null  object
 5   Age          96708 non-null   float64
 6   SibSp        100000 non-null  int64
 7   Parch        100000 non-null  int64
 8   Ticket       95377 non-null   object
 9   Fare         99866 non-null   float64
 10  Cabin        32134 non-null   object
 11  Embarked     99750 non-null   object
dtypes: float64(2), int64(5), object(5)
memory usage: 9.2+ MB
None
'''
print(test.head())
'''
   PassengerId  Pclass              Name     Sex   Age  SibSp  Parch  Ticket   Fare   Cabin Embarked
0       100000       3  Holliday, Daniel    male  19.0      0      0   24745  63.01     NaN        S
1       100001       3  Nguyen, Lorraine  female  53.0      0      0   13264   5.81     NaN        S
2       100002       1   Harris, Heather  female  19.0      0      0   25990  38.91  B15315        C
3       100003       2      Larsen, Eric    male  25.0      0      0  314011  12.93     NaN        S
4       100004       1     Cleary, Sarah  female  17.0      0      2   26203  26.89  B22515        C
'''
print(test.describe())
'''
         PassengerId         Pclass           Age          SibSp         Parch          Fare
count  100000.000000  100000.000000  96513.000000  100000.000000  100000.00000  99867.000000
mean   149999.500000       2.368930     30.565796       0.486550       0.49283     45.374804
std     28867.657797       0.878458     14.054634       0.771262       0.92360     65.204725
min    100000.000000       1.000000      0.080000       0.000000       0.00000      0.050000
25%    124999.750000       1.000000     21.000000       0.000000       0.00000     10.130000
50%    149999.500000       3.000000     27.000000       0.000000       0.00000     13.980000
75%    174999.250000       3.000000     40.000000       1.000000       1.00000     37.390000
max    199999.000000       3.000000     81.000000       8.000000       9.00000    680.700000
'''
print(test.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 11 columns):
 #   Column       Non-Null Count   Dtype
---  ------       --------------   -----
 0   PassengerId  100000 non-null  int64
 1   Pclass       100000 non-null  int64
 2   Name         100000 non-null  object
 3   Sex          100000 non-null  object
 4   Age          96513 non-null   float64
 5   SibSp        100000 non-null  int64
 6   Parch        100000 non-null  int64
 7   Ticket       94819 non-null   object
 8   Fare         99867 non-null   float64
 9   Cabin        29169 non-null   object
 10  Embarked     99723 non-null   object
dtypes: float64(2), int64(4), object(5)
memory usage: 8.4+ MB
None
'''
all_df = pd.concat([train, test]).reset_index(drop=True)
all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] 

all_df['Age'] = all_df['Age'].fillna(all_df['Age'].median())
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')
all_df['Embarked'] = all_df['Embarked'].fillna('X')

fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
all_df['Fare'] = np.log1p(all_df['Fare'])

label_cols = ['Age', 'Ticket', 'Sex', 'Cabin', 'Embarked', 'Pclass', 'SibSp', 'Parch', 'FamilySize']
numerical_cols = ['Fare']

def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)

label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = all_df[numerical_cols]
target_df = all_df[TARGET]

all_df = pd.concat([numerical_df, label_encoded_df, target_df], axis=1)

X_train = all_df[:train.shape[0]].drop(TARGET, axis=1)
X_test = all_df[train.shape[0]:].drop(TARGET, axis=1).reset_index(drop=True)
y_train = train[TARGET]

print(X_train,'\n',X_test)
'''
           Fare  Age  Ticket  Sex  Cabin  Embarked  Pclass  SibSp  Parch  FamilySize
0      3.337192   71      49    1      2         2       0      2      0           2
1      2.663750   71      49    1      8         2       2      0      0           0
2      4.280686    3      14    1      8         2       2      1      2           3
3      2.641910   47       0    1      8         2       2      0      0           0
4      2.170196   59      49    1      8         2       2      0      0           0
...         ...  ...     ...  ...    ...       ...     ...    ...    ...         ...
99995  2.763800  133      21    0      3         0       1      0      0           0
99996  2.497329  141      49    1      8         2       1      0      0           0
99997  2.393339   83      49    1      8         2       2      0      0           0
99998  3.463233  111      49    1      8         2       2      0      1           1
99999  2.705380  119      49    1      8         2       2      0      0           0
[100000 rows x 10 columns] 

            Fare  Age  Ticket  Sex  Cabin  Embarked  Pclass  SibSp  Parch  FamilySize
0      4.159039   47      49    1      8         2       2      0      0           0
1      1.918392  115      49    0      8         2       2      0      0           0
2      3.686627   47      49    0      1         0       0      0      0           0
3      2.634045   59      49    1      8         2       1      0      0           0
4      3.328268   43      49    0      1         0       0      0      2           2
...         ...  ...     ...  ...    ...       ...     ...    ...    ...         ...
99995  2.408745   63      49    0      8         1       2      0      0           0
99996  4.238589  127      49    1      8         2       0      1      0           1
99997  2.474014  103      49    1      8         2       2      0      0           0
99998  3.423611  107      21    0      1         0       0      1      2           3
99999  5.280204   91      21    0      4         0       0      0      2           2

[100000 rows x 10 columns]
'''

N_ESTIMATORS = 1000
N_SPLITS = 10
SEED = 2021
EARLY_STOPPING_ROUNDS = 100
VERBOSE = False

parameters = {
    'max_depth': np.arange(2, 5, dtype=int),
    'min_samples_leaf':  np.arange(2, 5, dtype=int),
}

# DecisionTreeClassifier
model = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=SEED),
    param_grid=parameters,
    scoring='accuracy',
    cv=10,
    n_jobs=-1)
model.fit(X_train, y_train)
dtm_best_parameters = model.best_params_

# RandomForestClassifier
model = GridSearchCV(
    estimator=RandomForestClassifier(random_state=SEED),
    param_grid=parameters,
    scoring='accuracy',
    cv=10,
    n_jobs=-1)
model.fit(X_train, y_train)
rfc_best_parameters = model.best_params_

lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': N_ESTIMATORS,
    'objective': 'binary',
    'random_state': SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}

ctb_params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': SEED,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': N_ESTIMATORS,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

lor_params = {
    'max_iter': 300,
    'n_jobs': -1,
    'random_state': SEED
}

dtc_params = {
    'max_depth': dtm_best_parameters['max_depth'],
    'min_samples_leaf': dtm_best_parameters['min_samples_leaf'],
    'class_weight': 'balanced',
    'random_state': SEED
}

rfc_params = {
    'max_depth': rfc_best_parameters['max_depth'],
    'min_samples_leaf': rfc_best_parameters['min_samples_leaf'],
    'random_state': SEED
}


hgb_params = {
    'random_state': SEED
}

svc_params = {
    'dual': False,
    'random_state': SEED
}

lgb_oof, lgb_preds = np.zeros(train.shape[0]), 0
ctb_oof, ctb_preds = np.zeros(train.shape[0]), 0
lor_oof, lor_preds = np.zeros(train.shape[0]), 0
dtc_oof, dtc_preds = np.zeros(train.shape[0]), 0
rfc_oof, rfc_preds = np.zeros(train.shape[0]), 0
hgb_oof, hgb_preds = np.zeros(train.shape[0]), 0
svc_oof, svc_preds = np.zeros(train.shape[0]), 0

cv_score = pd.DataFrame(np.zeros((N_SPLITS+1) * 7).reshape(N_SPLITS+1, -1),
                        columns=['lgb', 'ctb', 'lor', 'dtc', 'rfc', 'hgb', 'svc'])
    
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    
    # LGBMRegressor()
    pre_model = lgb.LGBMRegressor(**lgb_params)
    pre_model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr),(X_va, y_va)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE
    )

    lgb_params2 = lgb_params.copy()
    lgb_params2['learning_rate'] = lgb_params['learning_rate'] * 0.1
    model = lgb.LGBMRegressor(**lgb_params2)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr),(X_va, y_va)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE,
        init_model=pre_model
    )
    
    lgb_oof[valid_idx] = model.predict(X_va)
    lgb_preds += model.predict(X_test) / N_SPLITS
    cv_score.loc[fold, 'lgb'] = accuracy_score(y_va, np.where(lgb_oof[valid_idx]>0.5, 1, 0))
    
    # CatBoostClassifier()
    model = ctb.CatBoostClassifier(**ctb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              use_best_model=True,
              early_stopping_rounds=EARLY_STOPPING_ROUNDS,
              verbose=VERBOSE
              )
    
    ctb_oof[valid_idx] = model.predict(X_va)
    ctb_preds += model.predict(X_test) / N_SPLITS
    cv_score.loc[fold, 'ctb'] = accuracy_score(y_va, np.where(ctb_oof[valid_idx]>0.5, 1, 0))

    # LogisticRegression()
    model = LogisticRegression(**lor_params)
    model.fit(X_tr, y_tr)
    
    lor_oof[valid_idx] = model.predict_proba(X_va)[:, 1]
    lor_preds += model.predict_proba(X_test)[: ,1] / N_SPLITS
    cv_score.loc[fold, 'lor'] = accuracy_score(y_va, np.where(lor_oof[valid_idx]>0.5, 1, 0))

    # DecisionTreeClassifier
    model = DecisionTreeClassifier(**dtc_params)
    model.fit(X_tr, y_tr)
    
    dtc_oof[valid_idx] = model.predict_proba(X_va)[:, 1]
    dtc_preds += model.predict_proba(X_test)[: ,1] / N_SPLITS
    cv_score.loc[fold, 'dtc'] = accuracy_score(y_va, np.where(dtc_oof[valid_idx]>0.5, 1, 0))

    # RandomForestClassifier()
    model = RandomForestClassifier(**rfc_params)
    model.fit(X_tr, y_tr)
    
    rfc_oof[valid_idx] = model.predict_proba(X_va)[:, 1]
    rfc_preds += model.predict_proba(X_test)[: ,1] / N_SPLITS
    cv_score.loc[fold, 'rfc'] = accuracy_score(y_va, np.where(rfc_oof[valid_idx]>0.5, 1, 0))

    # HistGradientBoostingClassifier()
    model = HistGradientBoostingClassifier(**hgb_params)
    model.fit(X_tr, y_tr)
    
    hgb_oof[valid_idx] = model.predict_proba(X_va)[:, 1]
    hgb_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    cv_score.loc[fold, 'hgb'] = accuracy_score(y_va, np.where(hgb_oof[valid_idx]>0.5, 1, 0))

    # LinearSVC()
    model = LinearSVC(**svc_params)
    model.fit(X_tr, y_tr)
    
    svc_oof[valid_idx] = model.predict(X_va)
    svc_preds += model.predict(X_test) / N_SPLITS
    cv_score.loc[fold, 'svc'] = accuracy_score(y_va, np.where(svc_oof[valid_idx]>0.5, 1, 0))

    print(f"FOLD {fold:1d}: "\
          f"lgb {cv_score.loc[fold, 'lgb']:.4f}, "\
          f"ctb {cv_score.loc[fold, 'ctb']:.4f}, "\
          f"lor {cv_score.loc[fold, 'lor']:.4f}, "\
          f"dtc {cv_score.loc[fold, 'dtc']:.4f}, "\
          f"rfc {cv_score.loc[fold, 'rfc']:.4f}, "\
          f"hgb {cv_score.loc[fold, 'hgb']:.4f}, "\
          f"svc {cv_score.loc[fold, 'svc']:.4f}")

cv_score.loc[N_SPLITS, 'lgb'] = accuracy_score(y_train, np.where(lgb_oof>0.5, 1, 0))
cv_score.loc[N_SPLITS, 'ctb'] = accuracy_score(y_train, np.where(ctb_oof>0.5, 1, 0))
cv_score.loc[N_SPLITS, 'lor'] = accuracy_score(y_train, np.where(lor_oof>0.5, 1, 0))
cv_score.loc[N_SPLITS, 'dtc'] = accuracy_score(y_train, np.where(dtc_oof>0.5, 1, 0))
cv_score.loc[N_SPLITS, 'rfc'] = accuracy_score(y_train, np.where(rfc_oof>0.5, 1, 0))
cv_score.loc[N_SPLITS, 'hgb'] = accuracy_score(y_train, np.where(hgb_oof>0.5, 1, 0))
cv_score.loc[N_SPLITS, 'svc'] = accuracy_score(y_train, np.where(svc_oof>0.5, 1, 0))
print(f"lgb {cv_score.loc[N_SPLITS, 'lgb']:.6f}, "\
      f"ctb {cv_score.loc[N_SPLITS, 'ctb']:.6f}, "\
      f"lor {cv_score.loc[N_SPLITS, 'lor']:.6f}, "\
      f"dtc {cv_score.loc[N_SPLITS, 'dtc']:.6f}, "\
      f"rfc {cv_score.loc[N_SPLITS, 'rfc']:.6f}, "\
      f"hgb {cv_score.loc[N_SPLITS, 'hgb']:.6f}, "\
      f"svc {cv_score.loc[N_SPLITS, 'svc']:.6f}")

plt.figure(figsize=(16, 10))
plt.plot(range(N_SPLITS+1), cv_score['lgb'], label="LGBMRegressor")
plt.plot(range(N_SPLITS+1), cv_score['ctb'], label="CatBoostClassifier")
plt.plot(range(N_SPLITS+1), cv_score['lor'], label="LogisticRegression")
plt.plot(range(N_SPLITS+1), cv_score['dtc'], label="DecisionTreeClassifier")
plt.plot(range(N_SPLITS+1), cv_score['rfc'], label="RandomForestClassifier")
plt.plot(range(N_SPLITS+1), cv_score['hgb'], label="HistGradientBoostingClassifier")
plt.plot(range(N_SPLITS+1), cv_score['svc'], label="LinearSVC")

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=16)
plt.title("CV: each fold", fontsize=24)
plt.show()


oof = pd.DataFrame()
oof['oof_lgb'] = np.where(lgb_oof>0.5, 1, 0)
oof['oof_ctb'] = np.where(ctb_oof>0.5, 1, 0)
oof['oof_lor'] = np.where(lor_oof>0.5, 1, 0)
oof['oof_dtc'] = np.where(dtc_oof>0.5, 1, 0)
oof['oof_rfc'] = np.where(rfc_oof>0.5, 1, 0)
oof['oof_hgb'] = np.where(hgb_oof>0.5, 1, 0)
oof['oof_svc'] = np.where(svc_oof>0.5, 1, 0)

div_threshold = int((len([col for col in oof.columns if col.startswith('oof_')])+1) / 2)
oof['voting'] = (oof[[col for col in oof.columns if col.startswith('oof_')]].sum(axis=1) >= div_threshold).astype(int)
voting_cv = accuracy_score(y_train, oof['voting'])
print(f"Voting ensemble CV {voting_cv:.6f}")

submission['submit_lgb'] = np.where(lgb_preds>0.5, 1, 0)
submission['submit_ctb'] = np.where(ctb_preds>0.5, 1, 0)
submission['submit_lor'] = np.where(lor_preds>0.5, 1, 0)
submission['submit_dtc'] = np.where(dtc_preds>0.5, 1, 0)
submission['submit_rfc'] = np.where(rfc_preds>0.5, 1, 0)
submission['submit_hgb'] = np.where(hgb_preds>0.5, 1, 0)
submission['submit_svc'] = np.where(svc_preds>0.5, 1, 0)

submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis = 1).value_counts()

div_threshold = int((len([col for col in submission.columns if col.startswith('submit_')])+1) / 2)

submission[TARGET] = (submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis=1) >= div_threshold).astype(int)
submission[['PassengerId', TARGET]].to_csv("A:\\study\\titanic\\sub\\voting_submission.csv", index=False)

'''
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
FOLD 3: lgb 0.7826, ctb 0.7833, lor 0.7649, dtc 0.7714, rfc 0.7734, hgb 0.7830, svc 0.7651
FOLD 4: lgb 0.7800, ctb 0.7830, lor 0.7660, dtc 0.7715, rfc 0.7756, hgb 0.7808, svc 0.7654
FOLD 5: lgb 0.7832, ctb 0.7828, lor 0.7687, dtc 0.7730, rfc 0.7750, hgb 0.7831, svc 0.7681
FOLD 6: lgb 0.7863, ctb 0.7844, lor 0.7700, dtc 0.7784, rfc 0.7782, hgb 0.7866, svc 0.7700
C:\Users\sswwd\anaconda3\envs\kaggle\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
FOLD 7: lgb 0.7888, ctb 0.7890, lor 0.7732, dtc 0.7817, rfc 0.7816, hgb 0.7882, svc 0.7721
FOLD 8: lgb 0.7796, ctb 0.7759, lor 0.7634, dtc 0.7708, rfc 0.7733, hgb 0.7761, svc 0.7616
FOLD 9: lgb 0.7784, ctb 0.7784, lor 0.7623, dtc 0.7722, rfc 0.7704, hgb 0.7780, svc 0.7630
lgb 0.783090, ctb 0.782640, lor 0.767390, dtc 0.774220, rfc 0.776120, hgb 0.782650, svc 0.767200
Voting ensemble CV 0.780220
'''

# 04/25 270등
# score : 0.80245 
