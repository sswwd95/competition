import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats


x = []
for i in range(1,11):
    if i != 10:
        df = pd.read_csv(f'../data/lotte/csv/answer ({i}).csv', index_col=0, header=0)
        data = df.to_numpy()
        x.append(data)

x = np.array(x)

# print(x.shape)
a= []
df = pd.read_csv(f'../data/lotte/csv/answer ({i}).csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(9):
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

sub = pd.read_csv('../lotte/sample.csv')
sub['prediction'] = np.array(a)
sub.to_csv('../data/lotte/answer_add10.csv',index=False)

# add_all  85.181
# 1 84.043
# 2 84.993
# 3 85.093
# 4 83.410
# 5 84.853
# 6 84.994
# 7 85.108
# 8 85.094
# 9 85.178
# 10 85.286(모바일넷 뺀 파일)
