import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/csv/samsung.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df = df.iloc[:662,[0,1,2,3,5,6]]
df= df.sort_index(ascending=True) # 번호 오름차순
y= df.iloc[:,0]
del df['시가']
df['시가'] = y
df1 = df.dropna(axis=0)
print(df1.columns)
print(df1.head())
print(df)
print(df.shape)

#14일 추가
df2 = pd.read_csv('./samsung/csv/samsung2.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df2 = df2.iloc[[0],[0,1,2,3,7,8]]
y= df2.iloc[:,0]
del df2['시가']
df2['시가'] = y
print(df2)
df3 = pd.concat([df,df2])
print(df3)

# #15일 추가
df4 = pd.read_csv('./samsung/csv/samsung3.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df4 = df4.iloc[[0],[0,1,2,3,7,8]]
y= df4.iloc[:,0]
del df4['시가']
df4['시가'] = y

f_data = pd.concat([df3,df4])
print(f_data)  
print(f_data.isnull().sum())
print(f_data)
print(f_data.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.9) # 폰트크기 0.9
# sns.heatmap(data=f_data.corr(), square=True, annot=True, cbar=True)
# # sns.heatmap(data=df.corr(), square=정사각형으로, annot=글씨 , cbar=오른쪽에 있는 bar)
# plt.show()


sam= f_data.to_numpy()

print(sam.shape) #(664, 7)
print(sam)
print(type(sam)) #<class 'numpy.ndarray'>

def split_x(seq, size, col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size),0:col].astype('float32')
        aaa.append(subset)
    return np.array(aaa)

size=5
col=6

sam=split_x(sam, size, col)
print(sam.shape)
print(sam)

x = sam[:-1,:,:-1] 
print(x)
print(x.shape) 
y = sam[1:,-2:,-1:] 
print(y.shape)
x_pred = sam[-1:,:,:-1]
print(x_pred.shape) 

# (658, 5, 5)
# (658, 2, 1)
# (1, 5, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=50)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=50)

y_train = y_train.reshape(y_train.shape[0],2)
y_test = y_test.reshape(y_test.shape[0],2)
y_val = y_val.reshape(y_val.shape[0],2)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
x_val =  x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],5,5)
x_test = x_test.reshape(x_test.shape[0],5,5)
x_pred = x_pred.reshape(x_pred.shape[0],5,5)
x_val =  x_val.reshape(x_val.shape[0],5,5)

np.save('./samsung/npy/sam.npy',arr=[x_train, x_test, x_val, y_train, y_test,y_val, x_pred])
