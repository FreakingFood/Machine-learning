import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

students_data = pd.read_csv("C:\\Users\\lab4\\Downloads\\Placement_Data_Full_Class.csv")
print("\nThe first 5 rows of placement data full class : \n",students_data.head())

print("\nThe description of data : \n",students_data.describe())

Y = students_data['status']
print("\nThe dependent data is : \n",Y)

X = students_data.drop(['sl_no','status','salary'],axis=1)
print("\nThe independent data is : \n",X)

X = pd.get_dummies(X,drop_first=True)#
print("\nThe dummy model of X : \n",X)
y = pd.get_dummies(Y,drop_first=True)#
print("\nThe dummy model of Y : \n",Y)

model = LogisticRegression()

scaler = MinMaxScaler()#
X = scaler.fit_transform(X)#
print("\nThe Scaler value of X : \n",X)

x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.10,random_state =2)

print("\nmodel.fit(x_train,y_train)\n",model.fit(x_train,y_train))#

print("\nModel Score of test data: \n",model.score(x_test,y_test))#

print("\nModel Score of training data: \n",model.score(x_train,y_train))#

y_pred = model.predict(x_test)#

cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix : \n",cm)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix")

sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')#
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()
plt.savefig('confusion_matrix.png')

