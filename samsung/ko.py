import numpy as np
import pandas as pd

ko = pd.read_csv('./samsung/csv/kodex.csv',encoding='cp949', thousands=',',index_col=0, header=0)

ko = ko.iloc[:664,[0,1,2,3,7,8]]


ko = ko.sort_index(ascending=True)
y=ko.iloc[:,0:1]
del ko['시가']
ko['시가'] =y
print(ko)
print(ko.isnull())
print(ko.isnull().sum())
ko = ko.dropna(axis=0)
print(ko)
print(ko.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.9) # 폰트크기 0.9
# sns.heatmap(data=ko.corr(), square=True, annot=True, cbar=True)
# # sns.heatmap(data=df.corr(), square=정사각형으로, annot=글씨 , cbar=오른쪽에 있는 bar)
# plt.show()

ko = ko.to_numpy()
# print(type(ko)) #<class 'numpy.ndarray'>

print(ko.shape) #(664, 6)
print(ko)
print(type(ko)) #<class 'numpy.ndarray'>

def split_x(seq, size, col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size),0:col].astype('float32')
        aaa.append(subset)
    return np.array(aaa)

size=5
col=6

ko=split_x(ko, size, col)

print(ko.shape) 

x1 = ko[:-1,:,:-1] 
print(x1)
print(x1.shape) 
y1 = ko[1:,-2:,-1:] 
print(y.shape)
x1_pred = ko[-1:,:,:-1]
print(x1_pred.shape) 
# (660, 5, 6)
# (658, 5, 6)
# (1, 5, 6)


from sklearn.model_selection import train_test_split
x1_train, x1_test,y1_train, y1_test= train_test_split(
    x1,y1, train_size = 0.8, random_state=50)
x1_train, x1_val,y1_train, y1_val = train_test_split(
    x1_train, y1_train, train_size=0.8, random_state=50)

y1_train = y1_train.reshape(y1_train.shape[0],2)
y1_test = y1_test.reshape(y1_test.shape[0],2)
y1_val = y1_val.reshape(y1_val.shape[0],2)

x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])
x1_pred = x1_pred.reshape(x1_pred.shape[0], x1_pred.shape[1]*x1_pred.shape[2])
x1_val =  x1_val.reshape(x1_val.shape[0], x1_val.shape[1]*x1_val.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_pred = scaler.transform(x1_pred)
x1_val = scaler.transform(x1_val)

print(x1_train.shape) #(420, 36)
print(x1_test.shape) # (132, 36)
print(x1_pred.shape) #(1,36)
print(x1_val.shape)#(105,36)

x1_train = x1_train.reshape(x1_train.shape[0],5,5)
x1_test = x1_test.reshape(x1_test.shape[0],5,5)
x1_pred = x1_pred.reshape(x1_pred.shape[0],5,5)
x1_val =  x1_val.reshape(x1_val.shape[0],5,5)



np.save('./samsung/npy/ko.npy',arr=[x1_train, x1_test, x1_val,x1_pred])
