import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, svm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

houses = pd.read_csv("C:\\Users\\lab4\\Documents\\186_1464_420/house_new.csv")
print(houses.head())

# create x and y
feature_cols = 'Area' 


x = houses.values[:,0] # Get input values from first column
y = houses.values[:,1]

print("\nx-coordinate: \n",x)
print("\ny-coordinate: \n",y)

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2) 
# the test set will be 20% of the whole data set
m = len(x_train)
n = len(x_test)
print("X_train : ",m)
