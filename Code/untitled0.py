# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:12:59 2020

@author: admin
"""
#%%
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from pandas import datetime
#%%
from pandas import DataFrame
df = DataFrame.from_csv("Data4.tsv", sep="\t")
#%%
#Creating a dataset variable and importing the dataset.csv file with it
dataset = pd.read_csv('Data5.tsv', delimiter="\t", header=0, encoding='utf-8')
#%%
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M')
dataset = pd.read_csv('Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)
#%%
dataset.plot()
dataset.show()
#%%
'''
Extra knowledge
pd.set_option('display.max_columns', 5) - To set displayed no. of columns

data = pd.read_csv('file1.csv', error_bad_lines=False) - To ignore error ridden lines
'''
#%%
#Creating separate matrices for x - indep, y- dep 
x = dataset.iloc[ : , 0].values
y = dataset.iloc[ : , 1:].values
#x is a matrix while y is a vector
#for defining x - specify a range so the resultant x is a matrix(10,1)
#for defining y - just specify the index of the reqd column directly to make it 
# a vector(10,)

#%%
#ELIMINATING THE MISSING VALUES

from sklearn.preprocessing import Imputer
#creating function variable using Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#attaching the variable imputer to our matrix y
imputer = imputer.fit(y[:,:])
#now we apply our imputer variable on matrix to fill in the
#missing values will be filled with the strategy we picked
#fit() used to apply changes on a temp var in memory
#transform() used to commit the changes to the said variable
#fit_transform() for doing both together
y=imputer.transform(y)

#%%


#%%
#Splitting dataset to training and test set
'''
Training Set- from which the model will learn from
Test -with which it will compare itself and check itself

'''

from sklearn.model_selection import train_test_split as tts
y_train,y_test = tts(y,test_size = 0.2, random_state=0)

#%%
#Feature Scaling- it scales the entries so that all columns are comparable to 
#same scale
from sklearn.preprocessing import StandardScaler as ss
sc_x = ss()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

