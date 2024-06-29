import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

students_data = pd.read_csv("C:\\Users\\lab4\\Downloads/Placement_Data_Full_Class")
print(students_data.head()

print(students_data.describe())
