import numpy as np

#tablice

a = np.array([1, 2, 3, 4, 5, 6, 7])
b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
b=np.transpose(b)

c=np.arange(0,100)
print(c)
d=np.linspace(0,10,10)
print(d)
e=np.arange(0,101,5)
print(e)

#liczby losowe
a=np.random.randn(1,20)
a=np.around(a,2)
print(a)
b=np.random.randint(1,1001,(1,100))
print(b)
c=np.zeros((3,2))
d=np.ones((3,2))
print(c,d)
e=np.random.randint(0,10,(5,5),dtype=np.int32)
print(e)

a=np.random.random((1,10))*10
print(a)
b=a.astype(int)
print(b)

#selekcja
b=np.array([[1,2,3,4,5], [6,7,8,9,10]],dtype=np.int32)
print('dim:',np.ndim(b),'size:',np.size(b))
print(b[0,1],b[0,3])
print(b[0,:])
print(b[:,0])
a=np.random.randint(0,101,(20,7))
print(a[:,0:4])

#Opercaje matematyczne i logiczne
a=np.random.randint(0,11,(3,3))
b=np.random.randint(0,11,(3,3))

try:
    print(a+b,a*b,a/b,a**b)
except:
    pass
print(np.all(a>=1),np.all(a<=4))
print(a.trace())

#Dane statystyczne
print('sum:',b.sum())
print('min:',b.min())
print('max:',b.max())
print('std:',b.std())
print('mean:',b.mean(axis=1))

#Rzutowanie wymiarów
a=np.linspace(1,50)
b=np.reshape(a,(10,5))
print(b)
a=np.resize(a,(10,5))
print(a.ravel())

a=np.arange(0,20)
b=np.arange(20,40)
a=np.reshape(a,(5,4))
b=np.reshape(b,(5,4))

print(np.vstack((a,b)))
print(a[np.newaxis,:])

#sortowanie
a=np.random.randn(5,5)
print(a)
print(np.sort(a,axis=1))
print(np.sort(a[::-1],axis=0))

#sortowanie
b=np.array([(1,'MZ','mazowieckie'),
(2,'ZP','zachodniopomorskie'),
(3,'ML','małopolskie')])
b=b[b[:, 1].argsort()]
print(b)
print(b[2,2])

#ZADANIA
#1
print("Zad1:")
A=np.random.randint(0,11,(10,5))
print(A)
print("Ślad macierzy:",A.trace())
print("Diagonalia:",A.diagonal())

#2
print("Zad2:")
A=np.random.randn(1,5)
B=np.random.randn(1,5)
print("A:\n",A)
print("B:\n",B)
print("A*B\n",A*B)

#3
print("Zad3:")
A=np.random.randint(1,101,(3,5))
B=np.random.randint(1,101,(3,5))
print("A:\n",A)
print("B:\n",B)
print("A+B:\n",A+B)

#4
print("Zad4:")
A=np.random.randint(1,101,(4,5))
B=np.random.randint(1,101,(5,4))
print("A:\n",A)
print("B:\n",B)
B.resize((4,5))
print("B resized:\n",B)
print("A+B:\n",A+B)

#5
print("Zad5:")
print(A[2]*B[3])

#6
print("Zad6:")
A=np.random.normal(0,1,size=(3,3))
B=np.random.uniform(0,1,size=(3,3))
print('A:\n',"mean:",A.mean(),"std:",A.std(),"var:",A.var())
print('B:\n',"mean:",B.mean(),"std:",B.std(),"var:",B.var())

#7
print("Zad7:")
A=np.random.randint(0,11,(2,2))
B=np.random.randint(0,11,(2,2))
print('A:\n',A)
print('B:\n',B)
print("A*B:\n",A*B)
print("A.*B:\n",np.dot(A,B))

#9
print("Zad9:")
A=np.random.randint(0,11,(2,2))
B=np.random.randint(0,11,(2,2))
print("Vstack:\n",np.vstack((A,B )))
print("Hstack:\n",np.hstack((A,B)))