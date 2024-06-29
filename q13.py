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

houses = pd.read_csv("C:\\Users\\lab4\\Documents\\186_1464_420/house.csv")
print(houses.head())

print(houses.describe())
houses['bedrooms'].value_counts().plot(kind='bar')
plt.title('No of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

plt.scatter(houses.price,houses.sqft_living)
plt.title("Price Vs Square Feet")
plt.show()

labels=houses['price']

train1=houses.drop(['id','date','price'],axis=1)
print(train1.head())

reg=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(train1,labels,test_size=0.10,random_state=2)#
x_train=x_train.replace((np.inf,-np.inf,np.nan),0).reset_index(drop=True)
x_test=x_test.replace((np.inf,-np.inf,np.nan),0).reset_index(drop=True)
reg.fit(x_train,y_train)#

print(reg.intercept_)
print(reg.coef_)

print(reg.score(x_test,y_test))#

y_pred=reg.predict(x_test)#

fig=plt.figure()
ax1 = fig.add_axes([0,0,1,1])
x=x_test
ax1.plot(x,y_test,'o',color='red')
ax2 = ax1.twinx()#
ax2.plot(x,y_pred,'+',color='yellow')
ax2.set_ylabel('Predicted')
fig.legend(labels=('Actual','Predicted'),loc='upper left')
plt.show()












