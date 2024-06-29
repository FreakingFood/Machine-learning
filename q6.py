import numpy as np

m1 = np.array(np.random.randn(12).reshape(4,3))
m2 = np.array(np.random.randn(12).reshape(4,3))
print("\nFirst Matrix : ",m1)
print("\nSecond Matrix : ",m2)

print("\nAddition :\n",m1+m2)
print("\nSubtraction :\n",m1-m2)
print("\nMultiplication :\n",m1*m2)

m3  = np.array(np.random.randn(9).reshape(3,3))
print("\nMatrix multiplication: ",np.dot(m1,m3))


