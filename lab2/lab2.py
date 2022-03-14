import imp
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import datetime as dt
import os

df=pd.read_csv("./autos.csv")
print(df.columns)

#manipulowanie danymi

A=pd.DataFrame({'data':np.arange(dt.datetime(2020,3,1),dt.datetime(2020,3,6),dt.timedelta(days=1)),
                'A':np.random.randn(5),
               'B':np.random.randn(5),
               'C':np.random.randn(5),
               })
A.set_index(['data'],inplace=True)
print(A)

#generowanie
A=pd.DataFrame(data=np.random.randint(-10,11,size=(20,3)),
               index=np.arange(1,21),
               columns=['A','B','C']
               )
A.index.name='id'
print(A.iloc[:3])
print(A.iloc[-3:])
print(A.index)
print(A.columns)

print(A.iloc[:,:])

print(A.sample(5,axis=0))
print(A['A'])
print(A.iloc[:3,:][['A','B']])
print(A.iloc[5,:])
print(A.iloc[[0,5,6,7],[1,2]])

#describe
print(A.describe())
print(A>0)