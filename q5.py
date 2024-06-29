import numpy as np
import pandas as pd

file = pd.read_table("matrix.txt",sep=",",header=None)
print("\nFile : \n",file)
for i in range(3):
    print("\n Elements of Row ",i+1," is ",file.loc[i,].size)

file1 = file.T
for i in range(5):
    print("\n Elements of Column ",i+1," is ",file1.loc[i,].size)
