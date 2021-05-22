
import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/csv/samsung.csv',encoding='cp949',thousands=',',index_col=0,header=0)
# df1 = df.iloc[:662,[0,1,2,3,5,6]]
df1 = df.iloc[:662,[0,1,2,3,5,6,10,11,12]]

df1= df1.sort_index(ascending=True) # 번호 오름차순
# y= df1.iloc[:,3:4]
# del df1['종가']
# df1['종가'] = y
# df1 = df1.dropna(axis=0)

print(df1)
print(df1.shape)
print(df1.corr())

# df2= df.iloc[662:,[0,1,2,3,5,6]] /50
# df2= df2.sort_index(ascending=True) # 번호 오름차순
# y2= df2.iloc[:,3:4]
# del df2['종가']
# df2['종가'] = y2
# df2 = df2.dropna(axis=0)
# print(df2)
# print(df2.shape)
# print(df2.corr())
# print(df2.isnull())
# print(df2.isnull().sum())

# df3 = pd.concat([df2,df1])

# print(df3)
# print(df3.info())
# print(df3.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.9) # 폰트크기 0.9
sns.heatmap(data=df1.corr(), square=True, annot=True, cbar=True)
plt.show()
# f_df = df1.to_numpy()
# print(f_df)
# print(type(f_df)) #<class 'numpy.ndarray'>
# print(f_df.shape) #(2397, 6)

# np.save('./samsung/npy/samsung_data.npy', arr = f_df)


# 분할 x
# loss, mae :  566593.3125 588.6598510742188
# RMSE :  566593.2
# R2 :  0.9963602791366756
# 1월 14일 :  [[90417.69]]

'''
import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/samsung.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df1 = df.iloc[:662,[0,1,2,3,5,6]]
df1= df1.sort_index(ascending=True) # 번호 오름차순
y= df1.iloc[:,3:4]
del df1['종가']
df1['종가'] = y
df1 = df1.dropna(axis=0)

print(df1)
print(df1.shape)
print(df1.corr())
f_df = df1.to_numpy()
print(f_df)
print(type(f_df)) #<class 'numpy.ndarray'>
print(f_df.shape) #(2397, 6)

np.save('./samsung/samsung1_data.npy', arr = f_df)

# 데이터 분할
# loss, mae :  425237.25 500.985551855469
# RMSE :  425237.28
# R2 :  0.9927418904822783
# 1월 14일 :  [[86000.836]]
'''
