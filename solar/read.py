import numpy as np
import pandas as pd




x = []
for i in range(0,6):
    df = pd.read_csv(f'./solar/csv/best_{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'./solar/csv/best_{i}.csv', index_col=0, header=0)
for i in range(7776):
    for j in range(9):
        a = []
        for k in range(6):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('./solar/csv/last.csv')        