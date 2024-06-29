import numpy as np

#1D array
arr1D = np.ones(10)
print("\n1D array with value 1 : \n",arr1D)
arr10D = np.zeros(10)
print("\n1D array with value 0 : \n",arr10D)
arrrandD = np.array(np.arange(10))
print("\n1D array with random value : \n",arrrandD)

#multidimensional array
arr2D = np.ones((10,8))
print("\n2D array with value 1 : \n",arr2D)
arr20D = np.zeros((10,8))
print("\n2D array with value 0 : \n",arr20D)
arrrand2D = np.array(np.random.randn(80).reshape(10,8))
print("\n2D array with random value : \n",arrrand2D)

#Diagonal matrix
arr3 = np.eye(4,3)
arr4 = np.identity(4)
arr5 = np.diag(np.full(5,3))
print("Array with diag value 1\n ",arr3)
print("Array with diag value 1\n ",arr4)
print("Array with diag value 3\n ",arr5)
