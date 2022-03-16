import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats


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
print(A[A>0])
print(A.mean())
print(A.mean(axis=1))

#concat
A=pd.DataFrame(data=np.random.randint(-10,11,size=(5,2)),
               columns=['A','B']
               )
B=pd.DataFrame(data=np.random.randint(-10,11,size=(5,1)),
               columns=['C']
               )
C=np.concatenate([A,B],axis=1)
print(C.T)

#sortowanie
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 
                   "y": ["a", "b", "a", "b", "b"]},
                  index=np.arange(5))
df.index.name='id'
df.sort_index(inplace=True)
print(df)
df.sort_values(['y'],ascending=False,inplace=True)
print(df)

#grupowanie
slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'],
            'Fruit': ['Apple','Apple', 'Banana', 'Banana', 'Apple'], 
            'Pound': [10, 15, 50, 40, 5],
            'Profit':[20, 30, 25, 20, 10]
}
df3 = pd.DataFrame(slownik)
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day','Fruit']).sum())

#wypelnianie
df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df.index.name='id'
print(df)
df['B']=1
print(df)
df.iloc[1,2]=20
df[df<0]=-df
print(df)

#uzupelnianie
df.iloc[[0,3],1]=np.nan
print(df)
df.fillna(0,inplace=True)
print(df)
df.iloc[[0,3],1]=np.nan
df=df.replace(to_replace=np.nan,value=-9999)
print(df)
df.iloc[[0,3],1]=np.nan
print(pd.isnull(df))

##############################
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 
                   "y": ["a", "b", "a", "b", "b"]})

#zad1
print(df.groupby('y').mean(),'\n')

#zad2
print(pd.DataFrame({col : df[col].value_counts() for col in df.columns}))
print(df.value_counts())

#zad3
#df=np.loadtxt("./lab2/autos.csv")
#print(df)
df=pd.read_csv("./lab2/autos.csv")
print(df.head())

#Zad4
print(df.groupby('make').mean()[['highway-mpg','city-mpg']])

#zad5
print(df.groupby('make')['fuel-type'].value_counts())

#zad6

p1=np.polyfit(df['length'],df['city-mpg'],deg=1)
p2=np.polyfit(df['length'],df['city-mpg'],deg=2)

#zad7
corr=scipy.stats.pearsonr(df['length'],df['city-mpg'])
print("Pearson correlation coefficient:",corr)

#zad8
x=np.linspace(df['length'].min(),df['length'].max(),100)
fig,axs=plt.subplots(1,1)
fig.suptitle('Zad8')

axs.scatter(df['length'],df['city-mpg'],s=30,alpha=0.5,label='sample')
axs.set_xlabel('length')
axs.set_ylabel('city-mpg')
#p1[0]*x+p1[1]
axs.plot(x,np.polyval(p1,x),label='deg 1',color='r')
#p2[0]*x**2+p2[1]*x+p2[2]
axs.plot(x,np.polyval(p2,x),label='deg 2',color='g')
axs.legend()

#plt.show()

#zad9

fig2,axs2=plt.subplots(1,2)
fig2.tight_layout()
fig2.supylabel('city-mpg')
axs2_twin=axs2[0].twinx()
axs2[0].set_xlabel('length')

axs2[0].tick_params(axis='y', labelcolor='b')
axs2_twin.tick_params(axis='y', labelcolor='r')

axs2[0].scatter(df['length'],df['city-mpg'],s=30,alpha=0.5,label='sample',color='b')
kernel=scipy.stats.gaussian_kde(df['length'])
axs2_twin.plot(x,kernel(x),label='density function',color='r')

axs2[0].legend()
axs2_twin.legend(loc = 'lower left')



#zad10
x2=np.linspace(df['width'].min(),df['width'].max(),100)
axs2_twin2=axs2[1].twinx()
axs2[1].set_xlabel('width')

axs2[1].tick_params(axis='y', labelcolor='b')
axs2_twin2.tick_params(axis='y', labelcolor='r')

axs2[1].scatter(df['width'],df['city-mpg'],s=30,alpha=0.5,label='sample',color='b')
kernel2=scipy.stats.gaussian_kde(df['width'])
axs2_twin2.plot(x2,kernel2(x2),label='density function',color='r')

axs2[1].legend()
axs2_twin2.legend(loc = 'lower left')


#Zad11
from mpl_toolkits import mplot3d

fig3 = plt.figure()
fig3.tight_layout()
ax3 = fig3.add_subplot(1, 2, 1,projection='3d')
ax3_2 = fig3.add_subplot(1, 2, 2,projection='3d')

ax3.set_xlabel('length')
ax3.set_ylabel('width')
ax3.set_zlabel('city-mpg')

ax3_2.set_xlabel('length')
ax3_2.set_ylabel('width')
ax3_2.set_zlabel('probability')

X,Y=np.meshgrid(x,x2)
positions = np.vstack([X.ravel(), Y.ravel()])
values=np.vstack([df['length'], df['width']])
kernel3=scipy.stats.gaussian_kde(values)

Z=kernel3(positions)
Z=np.reshape(Z,X.shape)

ax3.scatter3D(df['length'], df['width'], df['city-mpg'])
ax3.contour(X,Y,Z,cmap='viridis')

ax3_2.plot_surface(X,Y,Z,cmap='viridis')

plt.show()