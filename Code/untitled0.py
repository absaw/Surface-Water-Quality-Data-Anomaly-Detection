# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:05:43 2020

@author: admin
"""
#%%
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from pandas import datetime
from sklearn.preprocessing import Imputer
from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split as tts
#%%

#def input_data():
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M')
dataset = pd.read_csv('./Dataset/Data7.csv',header=0, delimiter=',',index_col=0, parse_dates=[0], date_parser=parser)
dataset = dataset.fillna(method ='pad') 
turb = dataset.filter(['Turb(FNU)'], axis=1)

#dataset.fillna(method ='bfill') 

train_size,test_size = 1920, 3251#in paper given as 3169
turb_train,turb_test = tts(turb,train_size = train_size, random_state=0, shuffle=False)
X = turb_train.values
# walk-forward validation
history = [x for x in turb_train]
predictions = list()
for i in range(len(turb_test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(turb_test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()

#%%
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()